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

## 4) Default models (LLM + RAG embedding)

From `src/config.py` and retrieval logic:

- Default backend: `custom` (the models run locally through vLLM on your machine/GPUs).
- Alternative backend supported: `gemini_api` (the same pipeline flow is kept, but model inference is delegated to Gemini API).
- Default extractor model path: `Qwen/Qwen3-Next-80B-A3B-Instruct-FP8`
- Default reasoner model path: `Qwen/Qwen3-Next-80B-A3B-Instruct-FP8`
- Default embedding model: `sentence-transformers/paraphrase-multilingual-mpnet-base-v2`

Model choices are configurable in `src/config.py`.

## 5) Run locally

1. Clone the repository
```bash
git clone https://github.com/git-haddadz/AdDhakhiraCorpusAI.git
cd AdDhakhiraCorpusAI
```

2. Create and activate a Python environment
```bash
python -m venv .venv
source .venv/bin/activate
pip install -U pip
pip install -r requirements.txt
```

3. Run the main entrypoint
```bash
python -u main.py \
  --question "كيف تكون صلاة الجنازة عند المالكية؟" \
  --output "./outputs/output_local.html" \
  --diagnostic-coherence
```

## 6) Demo - Get Started (Gemini)

If you cannot run the full local stack, start with the Gemini notebook:

- Google Colab: [AdDhakhira_WebApp_Gemini.ipynb](https://colab.research.google.com/github/git-haddadz/AdDhakhiraCorpusAI/blob/main/AdDhakhira_WebApp_Gemini.ipynb)

This notebook keeps the same pipeline flow while using the Gemini backend to reduce local hardware constraints.

## 7) Demo - Get Started (Custom)

For Google Colab free usage, the custom notebook defaults to a smaller quantized model for easier testing:

- Google Colab: [AdDhakhira_WebApp_Custom.ipynb](https://colab.research.google.com/github/git-haddadz/AdDhakhiraCorpusAI/blob/main/AdDhakhira_WebApp_Custom.ipynb)

This default model is less accurate than the project’s optimal setup. If you have enough compute (local machine or stronger Colab setup), switch the notebook models back to `Qwen/Qwen3-Next-80B-A3B-Instruct-FP8` to reproduce the project’s optimal conditions.

## 8) Is the project modifiable?

Yes.

The project is open-source and designed to be forked, improved, and adapted, as long as derivative use remains free and open-source. Adapting the tool to a corpus from another madhhab is explicitly encouraged.
