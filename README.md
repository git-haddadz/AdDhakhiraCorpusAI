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

- Default LLM (extractor/reasoner): `Qwen/Qwen3-Next-80B-A3B-Instruct-FP8`
- Embedding model:
  - `EMBEDDING_MODEL = None` by default in config.
  - If no custom local path is provided, retrieval attempts a local cached sentence-transformers multilingual model (`sentence-transformers/paraphrase-multilingual-mpnet-base-v2`) when available.

Model choices are configurable and can be adapted to other models through `vLLM` by editing the project configuration.

## 5) Is the project modifiable?

Yes.

The project is open-source and designed to be forked, improved, and adapted, as long as derivative use remains free and open-source. Adapting the tool to a corpus from another madhhab is explicitly encouraged.
