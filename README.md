العربية | [Français](README.fr.md) | [English](README.en.md)

# AdDhakhiraCorpusAI

## ١) لماذا سُمّي هذا المستودع `AdDhakhiraCorpusAI`؟

للاسم مقصدان:

- هو إشارة وتقدير للإمام القرافي رحمه الله تعالى، أحد كبار علماء المذهب المالكي، ولعمله الجليل **الذخيرة**، وهو كتاب ننصح طلبة العلم بالتعرّف عليه.
- ومن جهة اللغة، فالذخيرة هي ما يُدّخر وينتفع به عند الحاجة. ومن هذا المعنى صُمّم المشروع ليكون موضعاً تُجمع فيه الكتب والمعارف وتُنظّم لأجل الدراسة والبحث.

وبهذا المعنى، يعمل **AdDhakhiraCorpusAI** كذخيرة رقمية للطلاب والباحثين: مجموعة منظّمة من المراجع المالكية يمكن البحث فيها واسترجاع مواضعها ودراستها بكفاءة أكبر.

## ٢) ما هي المراجع التي يعتمد عليها المشروع؟

يعتمد corpus الحالي على ملفات `database/` ويضم:

- **شرح الخرشي على مختصر خليل - ومعه حاشية العدوي** — **الخرشي = الخراشي; العدوي**
- **مواهب الجليل في شرح مختصر خليل** — **الحطاب**
- **الذخيرة للقرافي** — **القرافي**
- **القوانين الفقهية** — **ابن جزي الكلبي**
- **الثمر الداني شرح رسالة ابن أبي زيد القيرواني** — **صالح بن عبد السميع الأزهري**
- **المدخل لابن الحاج** — **ابن الحاج**
- **شرح الزرقاني على مختصر خليل وحاشية البناني** — **الزرقاني، عبد الباقي**
- **التوضيح في شرح مختصر ابن الحاجب** — **خليل بن إسحاق الجندي**
- **شرح زروق على متن الرسالة** — **زروق**
- **التبصرة للخمي** — **اللخمي، أبو الحسن**
- **الفواكه الدواني على رسالة ابن أبي زيد القيرواني** — **النفراوي**
- **حاشية العدوي على كفاية الطالب الرباني** — **العدوي**
- **حاشية الصاوي على الشرح الصغير = بلغة السالك لأقرب المسالك** — **أحمد الصاوي**
- **التاج والإكليل لمختصر خليل** — **محمد بن يوسف المواق**
- **منح الجليل شرح مختصر خليل** — **محمد بن أحمد عليش**
- **المقدمات الممهدات** — **ابن رشد الجد**
- **شرح التلقين** — **المازري**
- **تحبير المختصر وهو الشرح الوسط لبهرام على مختصر خليل** — **بهرام الدميري**

نسأل الله أن يحفظ العلماء الذين خدموا المذهب المالكي وأثروا تراثه.

## ٣) نطاق المشروع وهدفه

**هذا مساعد بحثي ببليوغرافي للفقه المالكي يعتمد على RAG وLLM. لا يُصدر هذا النظام فتاوى.**

الغرض منه مساعدة الطلاب والباحثين على تصفح المصادر، واستخراج المواضع ذات الصلة، وتنظيم الدراسة حول النصوص.

## ٤) أوضاع الاستدلال والنماذج الافتراضية

تظل pipeline البحث نفسها في جميع الأوضاع:

- استخراج كلمات مفتاحية عربية؛
- استرجاع dense من صفحات corpus، وهو مفعّل افتراضياً لكنه اختياري؛
- توليد جواب اعتماداً على الصفحات المسترجعة؛
- إخراج HTML.

الذي يتغير فقط هو النموذجان اللغويان: extractor وreasoner.

### `default`

هذا هو الإعداد المحلي الموصى به وفق نتائج benchmark:

- Extractor: `gemma-4-12B-it`
- Reasoner: `Qwen3.6-35B-A3B`
- Embedding: `Qwen/Qwen3-Embedding-4B`
- Retrieval: dense
- Vector backend: FAISS

### `lite_version`

وضع أخف للتجربة على Colab أو GPU أصغر:

- Extractor: `Qwen/Qwen2.5-7B-Instruct-AWQ`
- Reasoner: `Qwen/Qwen2.5-7B-Instruct-AWQ`
- embedding/retrieval/output: مثل `default`

هذا الوضع أسهل في التشغيل لكنه أقل دقة من `default`.

### أوضاع API

يمكن في notebook استبدال النموذجين اللغويين فقط بخدمة API، مع بقاء embedding وretrieval والصفحات المسترجعة وإخراج HTML كما هي.

- `gemini_api`: يستخدم Gemini عبر `GEMINI_API_KEY`؛ ويتطلب حساباً في Google AI Studio ومفتاح API ([الدليل الرسمي لمفاتيح API](https://ai.google.dev/gemini-api/docs/api-key?hl=ar)).
- `openai_api`: يستخدم نماذج ChatGPT/OpenAI عبر `OPENAI_API_KEY`؛ ويتطلب حساباً في OpenAI Platform ومفتاح API ([الدليل الرسمي للبدء، بالإنجليزية](https://developers.openai.com/api/docs/quickstart)).
- `anthropic_api`: يستخدم Claude/Anthropic عبر `ANTHROPIC_API_KEY`؛ ويتطلب حساباً في Claude Platform ومفتاح API ([النظرة الرسمية على API، بالإنجليزية](https://platform.claude.com/docs/en/api/overview)).

هذه الأوضاع مفيدة لمن لا يملك ذاكرة GPU كافية محلياً لكنه يملك حساب API.

يمكن تعديل جميع نماذج المشروع من notebook أو من `src/config.py`. في التشغيل المحلي يجب أن تشير القيم إلى مجلدات نماذج موجودة على الجهاز أو في cache Hugging Face. يبقى استعمال dense retrieval هو الموصى به؛ ويمكن تعطيله عبر `USE_DENSE_RETRIEVAL` في واجهة الويب فقط كحل احتياطي عندما لا تستطيع الآلة تشغيل dense retrieval.

## ٥) التشغيل المحلي

1. استنساخ المستودع

```bash
git clone https://github.com/git-haddadz/AdDhakhiraCorpusAI.git
cd AdDhakhiraCorpusAI
```

2. بناء وتشغيل بيئة Docker

التشغيل المحلي يتم داخل بيئة Docker المرفقة بالمشروع.

ابنِ الصورة:

```bash
docker compose build
```

شغّل حاوية تفاعلية:

```bash
docker compose run --rm addhakhira-ai-dev
```

كل الأوامر التالية في هذا القسم تُنفّذ داخل الحاوية.

3. إنشاء ملف الإعداد المحلي والمجلدات

انسخ قالب الإعداد:

```bash
cp src/config_template.py src/config.py
```

أنشئ المجلدات المحلية التي يستعملها الإعداد الافتراضي:

```bash
mkdir -p models outputs database/vector_indexes
```

4. تنزيل النماذج الافتراضية

يستعمل إعداد `default` النماذج التالية:

- Extractor: `google/gemma-4-12B-it`
- Reasoner: `Qwen/Qwen3.6-35B-A3B`
- Embedding: `Qwen/Qwen3-Embedding-4B`

نزّل النماذج داخل المجلد المحلي `models/`:

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

5. تعديل `src/config.py`

في إعداد `default`، اضبط القيم التالية:

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

عدّل `NUM_GPUS_EXTRACTOR` و `NUM_GPUS_REASONER` عند الحاجة حسب جهازك.

في أوضاع API اضبط:

- `LLM_BACKEND`
- `MODEL_EXTRACTOR_PATH`
- `MODEL_REASONER_PATH`
- مفتاح API الموافق

6. بناء فهرس FAISS dense

الوضع الافتراضي يستعمل retrieval dense مع `Qwen/Qwen3-Embedding-4B`. يُبنى الفهرس مرة واحدة قبل تشغيل التطبيق، ثم يُعاد استعماله إذا كان متوافقاً.

استعمل نفس مسار نموذج embedding الموجود في `EMBEDDING_MODEL` داخل `src/config.py`:

```bash
python3 -m src.vector_index \
  --model "models/Qwen__Qwen3-Embedding-4B" \
  --backend faiss \
  --json-input ./database \
  --output-dir ./database/vector_indexes \
  --show-progress
```

7. التشغيل محلياً

### الخيار أ - التشغيل من الطرفية

```bash
python3 -u main.py \
  --question "Write your question here" \
  --output "./outputs/output_local.html" \
  --diagnostic-coherence
```

سيُكتب ملف HTML الناتج في:

```text
./outputs/output_local.html
```

### الخيار ب - واجهة محادثة محلية

يمكن أيضاً تشغيل واجهة محادثة محلية من نفس بيئة Docker:

```bash
python3 -m src.web_app --host 0.0.0.0 --port 7860
```

مع إعداد Docker Compose المرفق، يستعمل التطبيق شبكة المضيف. افتح الواجهة المحلية على:

```text
http://localhost:7860
```

تقرأ الواجهة مسارات النماذج المحلية والإعدادات الافتراضية من `src/config.py`. كما توفر قائمة لاختيار inference المحلي أو Gemini API أو ChatGPT/OpenAI API أو Claude/Anthropic API. يمكن إدخال مفاتيح API مباشرة في الواجهة، ويمكن تعطيل retrieval dense بالـ embeddings من خانة الاختيار عند الحاجة.

## ٦) تجربة سريعة

لاستعمال Google Colab، افتح notebook التطبيق:

- Google Colab: [AdDhakhira_WebApp.ipynb](https://colab.research.google.com/github/git-haddadz/AdDhakhiraCorpusAI/blob/main/AdDhakhira_WebApp.ipynb)

يوفر notebook قائمة اختيار:

- `default`
- `lite_version`
- Gemini API
- ChatGPT/OpenAI API
- Claude/Anthropic API

عند اختيار وضع API يظهر حقل المفتاح المناسب. أوضاع API تستبدل فقط extractor وreasoner، بينما embedding وretrieval والصفحات المختارة وصيغة الإخراج والحفظ في Drive وتحميل HTML محلياً تبقى كما هي.

يمكن تعديل حقول النماذج في notebook قبل التهيئة. يتحكم `MODEL_STORAGE` في مكان حفظ النماذج: حفظ دائم في Drive عبر `drive` أو حفظ مؤقت داخل جلسة Colab عبر `colab_session`.

## ٧) هل يمكن تعديل المشروع؟

نعم.

المشروع open-source ومصمم ليُنسخ ويُحسّن ويُكيّف، بشرط أن تبقى الاستعمالات المشتقة حرة ومفتوحة المصدر. كما أن تكييف الأداة مع corpus لمذهب آخر أمر مرحّب به.
