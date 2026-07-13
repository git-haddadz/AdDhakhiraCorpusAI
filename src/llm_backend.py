import json
import logging
import os
import re
import gc
import sys
from contextlib import contextmanager
from abc import ABC, abstractmethod
from typing import Dict, List, Optional

from src.config import (
    JSON_GENERATION_MAX_RETRIES,
    JSON_GENERATION_MAX_TOKEN_MULTIPLIER,
    VLLM_GPU_MEMORY_UTILIZATION,
    VLLM_MAX_NUM_BATCHED_TOKENS,
)

LOGGER = logging.getLogger(__name__)


class JSONGenerationError(RuntimeError):
    """Expose failed JSON generation details to the optional HTML diagnostic."""

    def __init__(
        self,
        message: str,
        *,
        provider: str,
        attempts: int,
        raw_response: Optional[str] = None,
        cause: Optional[BaseException] = None,
    ):
        super().__init__(message)
        self.provider = provider
        self.attempts = attempts
        self.raw_response = raw_response
        self.cause_type = type(cause).__name__ if cause is not None else None
        self.cause_message = str(cause) if cause is not None else None


@contextmanager
def _stdio_with_fileno_for_vllm():
    """Colab/IPython stdout has no fileno(), but vLLM V1 expects one."""
    def has_fileno(stream) -> bool:
        try:
            stream.fileno()
            return True
        except Exception:
            return False

    if has_fileno(sys.stdout) and has_fileno(sys.stderr):
        yield
        return

    old_stdout, old_stderr = sys.stdout, sys.stderr
    stdout = os.fdopen(os.dup(1), "w", buffering=1)
    stderr = os.fdopen(os.dup(2), "w", buffering=1)
    try:
        sys.stdout = stdout
        sys.stderr = stderr
        yield
    finally:
        sys.stdout = old_stdout
        sys.stderr = old_stderr
        stdout.close()
        stderr.close()


def _extract_json_object(raw: str) -> Dict:
    text = (raw or "").strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    fenced = re.search(r"```(?:json)?\s*(\{.*\})\s*```", text, flags=re.DOTALL)
    if fenced:
        try:
            return _extract_json_object(fenced.group(1))
        except json.JSONDecodeError:
            pass

    decoder = json.JSONDecoder()
    best = None
    best_end = -1
    last_error: Optional[json.JSONDecodeError] = None
    for match in re.finditer(r"\{", text):
        try:
            candidate, end = decoder.raw_decode(text[match.start() :])
        except json.JSONDecodeError as exc:
            last_error = exc
            continue
        if isinstance(candidate, dict):
            absolute_end = match.start() + end
            if absolute_end > best_end:
                best = candidate
                best_end = absolute_end
    if best is not None:
        return best
    if last_error is not None:
        raise last_error
    return json.loads(text)


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

    def close(self) -> None:
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
        max_num_batched_tokens = (
            int(VLLM_MAX_NUM_BATCHED_TOKENS)
            if VLLM_MAX_NUM_BATCHED_TOKENS is not None
            else min(4096, max_model_len)
        )
        with _stdio_with_fileno_for_vllm():
            self.model = LLM(
                model=model_path,
                tensor_parallel_size=tensor_parallel_size,
                max_model_len=max_model_len,
                max_num_batched_tokens=max_num_batched_tokens,
                max_num_seqs=1,
                gpu_memory_utilization=float(VLLM_GPU_MEMORY_UTILIZATION),
                swap_space=0,
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
        token_limit = max(current_max_tokens, int(max_tokens) * int(JSON_GENERATION_MAX_TOKEN_MULTIPLIER))
        attempt_idx = 0
        last_error = None
        while attempt_idx < int(JSON_GENERATION_MAX_RETRIES):
            attempt_idx += 1
            sampling = SamplingParams(
                temperature=temperature,
                top_p=top_p,
                max_tokens=min(current_max_tokens, token_limit),
                structured_outputs=StructuredOutputsParams(
                    json=schema,
                    disable_additional_properties=True,
                ),
            )
            output = self.model.generate(prompt, sampling)[0].outputs[0].text.strip()
            try:
                return _extract_json_object(output)
            except json.JSONDecodeError as exc:
                last_error = exc
                LOGGER.error(
                    "JSON parsing failed in CustomBackend.generate_json (attempt %s, max_tokens=%s): %s at line=%s col=%s pos=%s",
                    attempt_idx,
                    min(current_max_tokens, token_limit),
                    exc.msg,
                    exc.lineno,
                    exc.colno,
                    exc.pos,
                )
                LOGGER.error("---- RAW_MODEL_OUTPUT_ATTEMPT_%s_START ----", attempt_idx)
                LOGGER.error(output)
                LOGGER.error("---- RAW_MODEL_OUTPUT_ATTEMPT_%s_END ----", attempt_idx)
                current_max_tokens = min(
                    token_limit,
                    max(current_max_tokens + 1, int(current_max_tokens * 1.5)),
                )
        raise RuntimeError(
            "Model did not return valid JSON after "
            f"{JSON_GENERATION_MAX_RETRIES} attempts; last error: {last_error}"
        )

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

    def close(self) -> None:
        try:
            engine = getattr(self.model, "llm_engine", None)
            if engine is not None:
                executor = getattr(engine, "model_executor", None)
                if executor is not None and hasattr(executor, "shutdown"):
                    executor.shutdown()
                if hasattr(engine, "shutdown"):
                    engine.shutdown()
        except Exception as exc:
            LOGGER.warning("vLLM engine shutdown raised: %s", exc)
        try:
            del self.model
        except AttributeError:
            pass
        try:
            import torch

            if torch.distributed.is_available() and torch.distributed.is_initialized():
                torch.distributed.destroy_process_group()
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()
        except Exception as exc:
            LOGGER.warning("CUDA cleanup raised: %s", exc)
        try:
            from vllm.distributed.parallel_state import destroy_distributed_environment, destroy_model_parallel

            destroy_model_parallel()
            destroy_distributed_environment()
        except Exception:
            pass
        gc.collect()


class GeminiBackend(LLMBackend):
    def __init__(self, model_name: str, api_key: Optional[str] = None):
        from google import genai

        key = api_key or os.environ.get("GEMINI_API_KEY")
        if not key:
            raise ValueError("GEMINI_API_KEY is required when LLM_BACKEND='gemini_api'.")
        self.model_name = model_name
        self.client = genai.Client(api_key=key)

    @staticmethod
    def _extract_json(raw: str) -> Dict:
        return _extract_json_object(raw)

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
        response_schema: Optional[Dict] = None,
    ) -> str:
        parts = self._messages_to_parts(messages)
        config = {
            "temperature": float(temperature),
            "top_p": float(top_p),
            "max_output_tokens": int(max(16, max_tokens)),
        }
        if parts["system"]:
            config["system_instruction"] = parts["system"]
        if response_schema is not None:
            config["response_mime_type"] = "application/json"
            config["response_json_schema"] = response_schema

        response = self.client.models.generate_content(
            model=self.model_name,
            contents=parts["user"],
            config=config,
        )
        if not response.text:
            raise RuntimeError("Gemini SDK returned an empty text response.")
        return response.text.strip()

    def generate_json(
        self,
        messages: List[Dict[str, str]],
        schema: Dict,
        max_tokens: int,
        temperature: float = 0.0,
        top_p: float = 1.0,
    ) -> Dict:
        current_max_tokens = max(64, int(max_tokens))
        token_limit = max(
            current_max_tokens,
            int(max_tokens) * int(JSON_GENERATION_MAX_TOKEN_MULTIPLIER),
        )
        attempt_idx = 0
        last_error = None
        last_raw = None
        while attempt_idx < int(JSON_GENERATION_MAX_RETRIES):
            attempt_idx += 1
            try:
                raw = self._generate_raw(
                    messages,
                    max_tokens=min(current_max_tokens, token_limit),
                    temperature=temperature,
                    top_p=top_p,
                    response_schema=schema,
                )
                last_raw = raw
            except Exception as exc:
                raise JSONGenerationError(
                    f"Gemini request failed during JSON generation: {exc}",
                    provider="Gemini",
                    attempts=attempt_idx,
                    cause=exc,
                ) from exc
            try:
                return self._extract_json(raw)
            except json.JSONDecodeError as exc:
                last_error = exc
                LOGGER.error(
                    "JSON parsing failed in GeminiBackend.generate_json (attempt %s, max_tokens=%s): %s at line=%s col=%s pos=%s",
                    attempt_idx,
                    min(current_max_tokens, token_limit),
                    exc.msg,
                    exc.lineno,
                    exc.colno,
                    exc.pos,
                )
                LOGGER.error("---- RAW_GEMINI_OUTPUT_ATTEMPT_%s_START ----", attempt_idx)
                LOGGER.error(raw)
                LOGGER.error("---- RAW_GEMINI_OUTPUT_ATTEMPT_%s_END ----", attempt_idx)
                current_max_tokens = min(
                    token_limit,
                    max(current_max_tokens + 1, int(current_max_tokens * 1.5)),
                )
        raise JSONGenerationError(
            "Gemini did not return valid JSON after "
            f"{JSON_GENERATION_MAX_RETRIES} attempts; last error: {last_error}",
            provider="Gemini",
            attempts=attempt_idx,
            raw_response=last_raw,
            cause=last_error,
        )

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

    def close(self) -> None:
        self.client.close()


class OpenAIBackend(LLMBackend):
    def __init__(self, model_name: str, api_key: Optional[str] = None):
        from openai import OpenAI

        key = api_key or os.environ.get("OPENAI_API_KEY")
        if not key:
            raise ValueError("OPENAI_API_KEY is required when LLM_BACKEND='openai_api'.")
        self.model_name = model_name
        self.client = OpenAI(api_key=key, timeout=180.0)

    def _generate_raw(
        self,
        messages: List[Dict[str, str]],
        max_tokens: int,
        temperature: float,
        top_p: float,
        response_format: Optional[Dict] = None,
    ) -> str:
        params = {
            "model": self.model_name,
            "messages": messages,
            "temperature": float(temperature),
            "top_p": float(top_p),
            "max_tokens": int(max(16, max_tokens)),
        }
        if response_format:
            params["response_format"] = response_format
        completion = self.client.chat.completions.create(**params)
        content = completion.choices[0].message.content
        if not content:
            raise RuntimeError("OpenAI SDK returned an empty text response.")
        return content.strip()

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
        current_max_tokens = max(64, int(max_tokens))
        token_limit = max(
            current_max_tokens,
            int(max_tokens) * int(JSON_GENERATION_MAX_TOKEN_MULTIPLIER),
        )
        attempt_idx = 0
        last_error = None
        last_raw = None
        while attempt_idx < int(JSON_GENERATION_MAX_RETRIES):
            attempt_idx += 1
            try:
                raw = self._generate_raw(
                    messages + [schema_note],
                    max_tokens=min(current_max_tokens, token_limit),
                    temperature=temperature,
                    top_p=top_p,
                    response_format={"type": "json_object"},
                )
                last_raw = raw
            except Exception as exc:
                raise JSONGenerationError(
                    f"OpenAI request failed during JSON generation: {exc}",
                    provider="OpenAI",
                    attempts=attempt_idx,
                    cause=exc,
                ) from exc
            try:
                return _extract_json_object(raw)
            except json.JSONDecodeError as exc:
                last_error = exc
                LOGGER.error(
                    "JSON parsing failed in OpenAIBackend.generate_json (attempt %s, max_tokens=%s): %s at line=%s col=%s pos=%s",
                    attempt_idx,
                    min(current_max_tokens, token_limit),
                    exc.msg,
                    exc.lineno,
                    exc.colno,
                    exc.pos,
                )
                LOGGER.error("---- RAW_OPENAI_OUTPUT_ATTEMPT_%s_START ----", attempt_idx)
                LOGGER.error(raw)
                LOGGER.error("---- RAW_OPENAI_OUTPUT_ATTEMPT_%s_END ----", attempt_idx)
                current_max_tokens = min(
                    token_limit,
                    max(current_max_tokens + 1, int(current_max_tokens * 1.5)),
                )
        raise JSONGenerationError(
            "OpenAI did not return valid JSON after "
            f"{JSON_GENERATION_MAX_RETRIES} attempts; last error: {last_error}",
            provider="OpenAI",
            attempts=attempt_idx,
            raw_response=last_raw,
            cause=last_error,
        )

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

    def close(self) -> None:
        self.client.close()


class AnthropicBackend(LLMBackend):
    def __init__(self, model_name: str, api_key: Optional[str] = None):
        from anthropic import Anthropic

        key = api_key or os.environ.get("ANTHROPIC_API_KEY")
        if not key:
            raise ValueError("ANTHROPIC_API_KEY is required when LLM_BACKEND='anthropic_api'.")
        self.model_name = model_name
        self.client = Anthropic(api_key=key, timeout=180.0)

    @staticmethod
    def _split_messages(messages: List[Dict[str, str]]) -> Dict[str, object]:
        system_parts = []
        chat_messages = []
        for msg in messages:
            role = msg.get("role", "")
            content = msg.get("content", "")
            if role == "system":
                system_parts.append(content)
            elif role == "assistant":
                chat_messages.append({"role": "assistant", "content": content})
            else:
                chat_messages.append({"role": "user", "content": content})
        return {
            "system": "\n\n".join([p for p in system_parts if p.strip()]),
            "messages": chat_messages,
        }

    def _generate_raw(
        self,
        messages: List[Dict[str, str]],
        max_tokens: int,
        temperature: float,
        top_p: float,
    ) -> str:
        parts = self._split_messages(messages)
        params = {
            "model": self.model_name,
            "messages": parts["messages"],
            "max_tokens": int(max(16, max_tokens)),
            "temperature": float(temperature),
            "top_p": float(top_p),
        }
        if parts["system"]:
            params["system"] = parts["system"]
        message = self.client.messages.create(**params)
        text = "".join(
            block.text
            for block in message.content
            if getattr(block, "type", None) == "text"
        ).strip()
        if not text:
            raise RuntimeError("Anthropic SDK returned an empty text response.")
        return text

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
        current_max_tokens = max(64, int(max_tokens))
        token_limit = max(
            current_max_tokens,
            int(max_tokens) * int(JSON_GENERATION_MAX_TOKEN_MULTIPLIER),
        )
        attempt_idx = 0
        last_error = None
        last_raw = None
        while attempt_idx < int(JSON_GENERATION_MAX_RETRIES):
            attempt_idx += 1
            try:
                raw = self._generate_raw(
                    messages + [schema_note],
                    max_tokens=min(current_max_tokens, token_limit),
                    temperature=temperature,
                    top_p=top_p,
                )
                last_raw = raw
            except Exception as exc:
                raise JSONGenerationError(
                    f"Anthropic request failed during JSON generation: {exc}",
                    provider="Anthropic",
                    attempts=attempt_idx,
                    cause=exc,
                ) from exc
            try:
                return _extract_json_object(raw)
            except json.JSONDecodeError as exc:
                last_error = exc
                LOGGER.error(
                    "JSON parsing failed in AnthropicBackend.generate_json (attempt %s, max_tokens=%s): %s at line=%s col=%s pos=%s",
                    attempt_idx,
                    min(current_max_tokens, token_limit),
                    exc.msg,
                    exc.lineno,
                    exc.colno,
                    exc.pos,
                )
                LOGGER.error("---- RAW_ANTHROPIC_OUTPUT_ATTEMPT_%s_START ----", attempt_idx)
                LOGGER.error(raw)
                LOGGER.error("---- RAW_ANTHROPIC_OUTPUT_ATTEMPT_%s_END ----", attempt_idx)
                current_max_tokens = min(
                    token_limit,
                    max(current_max_tokens + 1, int(current_max_tokens * 1.5)),
                )
        raise JSONGenerationError(
            "Anthropic did not return valid JSON after "
            f"{JSON_GENERATION_MAX_RETRIES} attempts; last error: {last_error}",
            provider="Anthropic",
            attempts=attempt_idx,
            raw_response=last_raw,
            cause=last_error,
        )

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

    def close(self) -> None:
        self.client.close()
