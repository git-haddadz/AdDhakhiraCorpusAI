[العربية](README.md) | [Français](README.fr.md) | English

# AdDhakhiraCorpusAI

## 1) Why is the repository named `AdDhakhiraCorpusAI`?

The name has two intentions:

- It is a tribute to Imam al-Qarafi (رحمه الله تعالى), one of the great scholars of the Maliki school, and to his major work **al-Dhakhira (الذخيرة)**, which we warmly recommend students to discover.
- Linguistically, **dhakhira** means a stored reserve, repository, treasury, or provision kept for future need. In that sense, the project is conceived as a place where knowledge and books are gathered and organized for study.

Analogy with this project: **AdDhakhiraCorpusAI** acts as a digital *dhakhira* for students, a curated reserve of Maliki references that can be searched, retrieved, and studied efficiently.

## 2) Which bibliography is this based on?

Current corpus (from the `database/` folder):

- **شرح الخرشي على مختصر خليل - ومعه حاشية العدوي** (*Sharh al-Kharashi 'ala Mukhtasar Khalil - wa ma'ahu Hashiyat al-'Adawi*) — **الخرشي = الخراشي; العدوي** (*al-Kharashi; al-'Adawi*)
- **مواهب الجليل في شرح مختصر خليل** (*Mawahib al-Jalil fi Sharh Mukhtasar Khalil*) — **الحطاب** (*al-Hattab*)
- **الذخيرة للقرافي** (*al-Dhakhira li-l-Qarafi*) — **القرافي** (*al-Qarafi*)
- **القوانين الفقهية** (*al-Qawanin al-Fiqhiyyah*) — **ابن جزي الكلبي** (*Ibn Juzayy al-Kalbi*)
- **الثمر الداني شرح رسالة ابن أبي زيد القيرواني** (*al-Thamar al-Dani, Sharh Risalat Ibn Abi Zayd al-Qayrawani*) — **صالح بن عبد السميع الأزهري** (*Salih ibn 'Abd al-Sami' al-Azhari*)
- **المدخل لابن الحاج** (*al-Madkhal li-Ibn al-Hajj*) — **ابن الحاج** (*Ibn al-Hajj*)
- **شرح الزرقاني على مختصر خليل وحاشية البناني** (*Sharh al-Zurqani 'ala Mukhtasar Khalil wa Hashiyat al-Bannani*) — **الزرقاني، عبد الباقي** (*al-Zurqani, 'Abd al-Baqi*)
- **التوضيح في شرح مختصر ابن الحاجب** (*al-Tawdih fi Sharh Mukhtasar Ibn al-Hajib*) — **خليل بن إسحاق الجندي** (*Khalil ibn Ishaq al-Jundi*)
- **شرح زروق على متن الرسالة** (*Sharh Zarruq 'ala Matn al-Risalah*) — **زروق** (*Zarruq*)
- **التبصرة للخمي** (*al-Tabsirah li-l-Lakhmi*) — **اللخمي، أبو الحسن** (*al-Lakhmi, Abu al-Hasan*)
- **الفواكه الدواني على رسالة ابن أبي زيد القيرواني** (*al-Fawakih al-Dawani 'ala Risalat Ibn Abi Zayd al-Qayrawani*) — **النفراوي** (*al-Nafrawi*)
- **حاشية العدوي على كفاية الطالب الرباني** (*Hashiyat al-'Adawi 'ala Kifayat al-Talib al-Rabbani*) — **العدوي** (*al-'Adawi*)
- **حاشية الصاوي على الشرح الصغير = بلغة السالك لأقرب المسالك** (*Hashiyat al-Sawi 'ala al-Sharh al-Saghir = Bulghat al-Salik li-Aqrab al-Masalik*) — **أحمد الصاوي** (*Ahmad al-Sawi*)
- **التاج والإكليل لمختصر خليل** (*al-Taj wa-l-Iklil li-Mukhtasar Khalil*) — **محمد بن يوسف المواق** (*Muhammad ibn Yusuf al-Mawwaq*)
- **منح الجليل شرح مختصر خليل** (*Manh al-Jalil, Sharh Mukhtasar Khalil*) — **محمد بن أحمد عليش** (*Muhammad ibn Ahmad 'Illish*)
- **المقدمات الممهدات** (*al-Muqaddimat al-Mumahhidat*) — **ابن رشد الجد** (*Ibn Rushd al-Jadd*)
- **شرح التلقين** (*Sharh al-Talqin*) — **المازري** (*al-Maziri*)
- **تحبير المختصر وهو الشرح الوسط لبهرام على مختصر خليل** (*Tahbir al-Mukhtasar, wa huwa al-Sharh al-Wasat li-Bahram 'ala Mukhtasar Khalil*) — **بهرام الدميري** (*Bahram al-Damiri*)

May Allah preserve all the scholars who enriched the Maliki school.

## 3) Project scope and purpose

**A bibliographic research assistant for Maliki fiqh using RAG and LLMs. This tool does not issue fatwas.**

This is an AI-assisted bibliographic research tool for students and researchers. Its role is to help navigate sources, surface relevant passages, and support structured study.

## 4) Inference modes and default models

The pipeline keeps the same RAG flow across all inference modes:

- Arabic keyword extraction
- dense retrieval over the corpus pages, enabled by default but optional
- answer generation from retrieved pages
- HTML output

Only the two LLM calls change depending on the selected inference mode.

### `default`

This is the recommended local setup selected by the benchmark:

- Extractor: `gemma-4-12B-it`
- Reasoner: `Qwen3.6-35B-A3B`
- Embedding: `Qwen/Qwen3-Embedding-4B`
- Retrieval: dense
- Vector backend: FAISS

### `lite_version`

This mode is intended for easier Colab or smaller-GPU testing:

- Extractor: `Qwen/Qwen2.5-7B-Instruct-AWQ`
- Reasoner: `Qwen/Qwen2.5-7B-Instruct-AWQ`
- Embedding/retrieval/output: identical to `default`

It is lighter and easier to run, but less accurate than `default`.

### API modes

The notebook can also replace only the two LLMs with an API backend while keeping the same embedding, retrieval, retrieved pages, and HTML output.

- `gemini_api`: uses Gemini through `GEMINI_API_KEY`; requires a Google AI Studio account and an API key ([official API key guide](https://ai.google.dev/gemini-api/docs/api-key)).
- `openai_api`: uses ChatGPT/OpenAI models through `OPENAI_API_KEY`; requires an OpenAI Platform account and an API key ([official quickstart](https://developers.openai.com/api/docs/quickstart)).
- `anthropic_api`: uses Claude/Anthropic models through `ANTHROPIC_API_KEY`; requires a Claude Platform account and an API key ([official API overview](https://platform.claude.com/docs/en/api/overview)).

These modes are useful for users who do not have enough local GPU memory but do have an API account.

All model IDs are configurable in the notebook and in `src/config.py`. For local runs, the model values must point to model directories available on the machine, or to models already available in the configured Hugging Face/cache environment. Dense retrieval is recommended; `USE_DENSE_RETRIEVAL` in the web UI can disable it only as a fallback when the machine cannot run dense retrieval.

## 5) Run locally

1. Clone the repository

```bash
git clone https://github.com/git-haddadz/AdDhakhiraCorpusAI.git
cd AdDhakhiraCorpusAI
```

2. Build and start the Docker environment

The local setup runs inside the provided Docker environment.

Build the image:

```bash
docker compose build
```

Start an interactive container:

```bash
docker compose run --rm projet_rag-dev
```

All following commands in this section are run inside the container.

3. Create your local config and folders

Copy the configuration template:

```bash
cp src/config_template.py src/config.py
```

Create the local folders used by the default setup:

```bash
mkdir -p models outputs database/vector_indexes
```

4. Download the default models

The `default` configuration uses:

- Extractor: `google/gemma-4-12B-it`
- Reasoner: `Qwen/Qwen3.6-35B-A3B`
- Embedding: `Qwen/Qwen3-Embedding-4B`

Download them into the local `models/` folder:

```bash
hf download google/gemma-4-12B-it \
  --local-dir models/gemma-4-12B-it
```

```bash
hf download Qwen/Qwen3.6-35B-A3B \
  --local-dir models/Qwen3.6-35B-A3B
```

```bash
hf download Qwen/Qwen3-Embedding-4B \
  --local-dir models/Qwen__Qwen3-Embedding-4B
```

5. Edit `src/config.py`

For the `default` configuration, set:

```python
LLM_BACKEND = "default"

MODEL_EXTRACTOR_PATH = "models/gemma-4-12B-it"
MODEL_REASONER_PATH = "models/Qwen3.6-35B-A3B"

EMBEDDING_MODEL = "models/Qwen__Qwen3-Embedding-4B"

NUM_GPUS_EXTRACTOR = 1
NUM_GPUS_REASONER = 1

ENABLE_DENSE_RETRIEVAL = True
VECTOR_INDEX_BACKEND = "faiss"
```

Adjust `NUM_GPUS_EXTRACTOR` and `NUM_GPUS_REASONER` if needed for your machine.

For API modes, set:

- `LLM_BACKEND`
- `MODEL_EXTRACTOR_PATH`
- `MODEL_REASONER_PATH`
- the corresponding API key field

6. Prepare the dense FAISS index

The default pipeline uses dense retrieval with `Qwen/Qwen3-Embedding-4B`. Build the index once before running the app. Re-running this command reuses the existing compatible index.

Use the same embedding model path as `EMBEDDING_MODEL` in `src/config.py`:

```bash
python3 -m src.vector_index \
  --model "models/Qwen__Qwen3-Embedding-4B" \
  --backend faiss \
  --json-input ./database \
  --output-dir ./database/vector_indexes \
  --show-progress
```

7. Run the main entrypoint

```bash
python3 -u main.py \
  --question "Write your question here" \
  --output "./outputs/output_local.html" \
  --diagnostic-coherence
```

The generated HTML file will be written to:

```text
./outputs/output_local.html
```

## 6) Demo - Get Started

For Google Colab usage, run the web app notebook:

- Google Colab: [AdDhakhira_WebApp.ipynb](https://colab.research.google.com/github/git-haddadz/AdDhakhiraCorpusAI/blob/main/AdDhakhira_WebApp.ipynb)

The notebook exposes a dropdown for:

- `default`
- `lite_version`
- Gemini API
- ChatGPT/OpenAI API
- Claude/Anthropic API

When an API mode is selected, the corresponding API key field appears. API modes replace only the extractor and reasoner LLMs; the embedding model, retrieval, selected pages, output format, Drive save, and local HTML download stay the same.

You can edit the model fields in the notebook before initialization if you want to test different local models, API models, or embedding models. `MODEL_STORAGE` controls whether downloaded models are kept persistently in Drive (`drive`) or only in the current Colab runtime (`colab_session`).

## 7) Is the project modifiable?

Yes.

The project is open-source and designed to be forked, improved, and adapted, as long as derivative use remains free and open-source. Adapting the tool to a corpus from another madhhab is explicitly encouraged.
