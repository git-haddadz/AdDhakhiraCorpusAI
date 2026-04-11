import json
import logging
import os
import re
from abc import ABC, abstractmethod
from typing import Dict, List, Optional

import requests

LOGGER = logging.getLogger(__name__)


class LLMBackend(ABC):
    @abstractmethod
    def generate_json(
        self,
        messages: List[Dict[str, str]],
        schema: Dict,
        max_tokens: int,
        temperature: float = 0.0,
        top_p: float = 1.0,
    ) -> Dict:
        pass

    @abstractmethod
    def generate_text(
        self,
        messages: List[Dict[str, str]],
        max_tokens: int,
        temperature: float = 0.0,
        top_p: float = 1.0,
    ) -> str:
        pass

    @abstractmethod
    def count_tokens(self, text: str) -> int:
        pass

    @abstractmethod
    def truncate_by_tokens(self, text: str, max_tokens: int) -> str:
        pass

    @abstractmethod
    def get_tokenizer(self):
        pass


class CustomBackend(LLMBackend):
    def __init__(self, model_path: str, num_gpus: int = 1, max_model_len: int = 1024):
        import torch
        from transformers import AutoTokenizer
        from vllm import LLM

        if torch.cuda.device_count() == 0:
            tensor_parallel_size = 1
        else:
            tensor_parallel_size = min(num_gpus, torch.cuda.device_count())

        self.tokenizer = AutoTokenizer.from_pretrained(
            model_path,
            cache_dir=model_path,
            local_files_only=True,
            trust_remote_code=True,
        )
        model_dtype = "float16" if "awq" in model_path.lower() else "bfloat16"
        self.model = LLM(
            model=model_path,
            tensor_parallel_size=tensor_parallel_size,
            max_model_len=max_model_len,
            max_num_batched_tokens=min(4096, max_model_len),
            max_num_seqs=1,
            gpu_memory_utilization=0.90,
            swap_space=0,
            enforce_eager=False,
            dtype=model_dtype,
            disable_custom_all_reduce=True,
        )

    def generate_json(
        self,
        messages: List[Dict[str, str]],
        schema: Dict,
        max_tokens: int,
        temperature: float = 0.0,
        top_p: float = 1.0,
    ) -> Dict:
        from vllm import SamplingParams
        from vllm.sampling_params import StructuredOutputsParams

        prompt = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )
        current_max_tokens = max(64, int(max_tokens))
        attempt_idx = 0
        while True:
            attempt_idx += 1
            sampling = SamplingParams(
                temperature=temperature,
                top_p=top_p,
                max_tokens=current_max_tokens,
                structured_outputs=StructuredOutputsParams(
                    json=schema,
                    disable_additional_properties=True,
                ),
            )
            output = self.model.generate(prompt, sampling)[0].outputs[0].text.strip()
            try:
                return json.loads(output)
            except json.JSONDecodeError as exc:
                LOGGER.error(
                    "JSON parsing failed in CustomBackend.generate_json (attempt %s, max_tokens=%s): %s at line=%s col=%s pos=%s",
                    attempt_idx,
                    current_max_tokens,
                    exc.msg,
                    exc.lineno,
                    exc.colno,
                    exc.pos,
                )
                LOGGER.error("---- RAW_MODEL_OUTPUT_ATTEMPT_%s_START ----", attempt_idx)
                LOGGER.error(output)
                LOGGER.error("---- RAW_MODEL_OUTPUT_ATTEMPT_%s_END ----", attempt_idx)
                current_max_tokens = max(current_max_tokens + 1, int(current_max_tokens * 1.5))

    def generate_text(
        self,
        messages: List[Dict[str, str]],
        max_tokens: int,
        temperature: float = 0.0,
        top_p: float = 1.0,
    ) -> str:
        from vllm import SamplingParams

        prompt = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )
        sampling = SamplingParams(
            temperature=temperature,
            top_p=top_p,
            max_tokens=max(16, int(max_tokens)),
        )
        return self.model.generate(prompt, sampling)[0].outputs[0].text.strip()

    def count_tokens(self, text: str) -> int:
        return len(self.tokenizer.encode(text or "", add_special_tokens=False))

    def truncate_by_tokens(self, text: str, max_tokens: int) -> str:
        token_ids = self.tokenizer.encode(text, add_special_tokens=False)
        if len(token_ids) <= max_tokens:
            return text
        return self.tokenizer.decode(token_ids[:max_tokens], skip_special_tokens=True)

    def get_tokenizer(self):
        return self.tokenizer


class GeminiBackend(LLMBackend):
    def __init__(self, model_name: str, api_key: Optional[str] = None):
        key = api_key or os.environ.get("GEMINI_API_KEY")
        if not key:
            raise ValueError("GEMINI_API_KEY is required when LLM_BACKEND='gemini_api'.")
        self.api_key = key
        self.model_name = model_name
        self.session = requests.Session()
        self.base_url = "https://generativelanguage.googleapis.com/v1beta/models"

    @staticmethod
    def _extract_json(raw: str) -> Dict:
        text = raw.strip()
        fenced = re.search(r"```(?:json)?\s*(\{.*\})\s*```", text, flags=re.DOTALL)
        if fenced:
            text = fenced.group(1).strip()
        if "{" in text and "}" in text:
            start = text.find("{")
            end = text.rfind("}")
            text = text[start : end + 1]
        return json.loads(text)

    @staticmethod
    def _messages_to_parts(messages: List[Dict[str, str]]) -> Dict[str, str]:
        system_parts: List[str] = []
        user_parts: List[str] = []
        assistant_parts: List[str] = []
        for msg in messages:
            role = msg.get("role", "")
            content = msg.get("content", "")
            if role == "system":
                system_parts.append(content)
            elif role == "assistant":
                assistant_parts.append(content)
            else:
                user_parts.append(content)
        if assistant_parts:
            user_parts.append("\n\nPrevious assistant context:\n" + "\n\n".join(assistant_parts))
        return {
            "system": "\n\n".join([p for p in system_parts if p.strip()]),
            "user": "\n\n".join([p for p in user_parts if p.strip()]),
        }

    def _generate_raw(
        self,
        messages: List[Dict[str, str]],
        max_tokens: int,
        temperature: float,
        top_p: float,
    ) -> str:
        parts = self._messages_to_parts(messages)
        url = f"{self.base_url}/{self.model_name}:generateContent?key={self.api_key}"
        payload = {
            "contents": [{"parts": [{"text": parts["user"]}]}],
            "generationConfig": {
                "temperature": float(temperature),
                "topP": float(top_p),
                "maxOutputTokens": int(max(16, max_tokens)),
            },
        }
        if parts["system"]:
            payload["systemInstruction"] = {"parts": [{"text": parts["system"]}]}

        resp = self.session.post(url, json=payload, timeout=180)
        if resp.status_code >= 400:
            raise RuntimeError(f"Gemini API error {resp.status_code}: {resp.text[:1200]}")
        data = resp.json()
        try:
            return data["candidates"][0]["content"]["parts"][0]["text"].strip()
        except Exception as exc:
            raise RuntimeError(f"Unexpected Gemini response: {json.dumps(data)[:1200]}") from exc

    def generate_json(
        self,
        messages: List[Dict[str, str]],
        schema: Dict,
        max_tokens: int,
        temperature: float = 0.0,
        top_p: float = 1.0,
    ) -> Dict:
        schema_note = {
            "role": "system",
            "content": (
                "Return valid JSON only. Do not use markdown fences.\n"
                "Follow this JSON Schema exactly:\n"
                f"{json.dumps(schema, ensure_ascii=False)}"
            ),
        }
        attempts = 0
        current_max_tokens = max(64, int(max_tokens))
        while True:
            attempts += 1
            raw = self._generate_raw(
                messages + [schema_note],
                max_tokens=current_max_tokens,
                temperature=temperature,
                top_p=top_p,
            )
            try:
                return self._extract_json(raw)
            except Exception:
                if attempts >= 4:
                    raise
                current_max_tokens = max(current_max_tokens + 1, int(current_max_tokens * 1.4))

    def generate_text(
        self,
        messages: List[Dict[str, str]],
        max_tokens: int,
        temperature: float = 0.0,
        top_p: float = 1.0,
    ) -> str:
        return self._generate_raw(messages, max_tokens=max_tokens, temperature=temperature, top_p=top_p)

    def count_tokens(self, text: str) -> int:
        return max(1, int(len(text or "") / 4))

    def truncate_by_tokens(self, text: str, max_tokens: int) -> str:
        max_chars = max(32, int(max_tokens) * 4)
        if len(text) <= max_chars:
            return text
        return text[:max_chars]

    def get_tokenizer(self):
        return None
