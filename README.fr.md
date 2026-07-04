[العربية](README.md) | Français | [English](README.en.md)

# AdDhakhiraCorpusAI

## 1) Pourquoi le dépôt s'appelle-t-il `AdDhakhiraCorpusAI` ?

Le nom a deux intentions :

- Il rend hommage à l'imam al-Qarafi (رحمه الله تعالى), l'un des grands savants de l'école malikite, ainsi qu'à son œuvre majeure **al-Dhakhira (الذخيرة)**, que nous recommandons vivement aux étudiants de découvrir.
- Linguistiquement, **dhakhira** désigne une réserve, un dépôt, un trésor ou une provision conservée pour un besoin futur. Le projet est conçu dans cet esprit : un lieu où des connaissances et des livres sont rassemblés et organisés pour l'étude.

Par analogie, **AdDhakhiraCorpusAI** agit comme une *dhakhira* numérique pour les étudiants : une réserve structurée de références malikites que l'on peut rechercher, interroger et étudier plus efficacement.

## 2) Sur quelle bibliographie le projet s'appuie-t-il ?

Corpus actuel, depuis le dossier `database/` :

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

Qu'Allah préserve les savants qui ont enrichi l'école malikite.

## 3) Objectif du projet

**Assistant de recherche bibliographique pour le fiqh malikite avec RAG et LLMs. Cet outil ne délivre pas de fatwas.**

Il s'agit d'un outil d'aide à la recherche pour étudiants et chercheurs. Son rôle est d'aider à naviguer dans les sources, faire remonter des passages pertinents et soutenir une étude structurée.

## 4) Modes d'inférence et modèles par défaut

La pipeline garde le même flux RAG dans tous les modes :

- extraction de mots-clés arabes ;
- retrieval dense sur les pages du corpus ;
- génération de réponse à partir des pages récupérées ;
- sortie HTML.

Seuls les deux appels LLM changent selon le mode choisi.

### `default`

Configuration locale recommandée par le benchmark :

- Extractor : `gemma-4-12B-it`
- Reasoner : `Qwen3.6-35B-A3B`
- Embedding : `Qwen/Qwen3-Embedding-4B`
- Retrieval : dense
- Backend vectoriel : FAISS

### `lite_version`

Mode prévu pour Colab ou des GPU plus modestes :

- Extractor : `Qwen/Qwen2.5-7B-Instruct-AWQ`
- Reasoner : `Qwen/Qwen2.5-7B-Instruct-AWQ`
- Embedding/retrieval/sortie : identiques à `default`

Ce mode est plus léger, mais moins précis que `default`.

### Modes API

Le notebook peut remplacer uniquement les deux LLM par une API tout en gardant le même embedding, le même retrieval, les mêmes pages récupérées et la même sortie HTML.

- `gemini_api` : Gemini via `GEMINI_API_KEY` ; nécessite un compte Google AI Studio et une clé API ([guide officiel des clés API](https://ai.google.dev/gemini-api/docs/api-key?hl=fr)).
- `openai_api` : ChatGPT/OpenAI via `OPENAI_API_KEY` ; nécessite un compte OpenAI Platform et une clé API ([quickstart officiel, en anglais](https://developers.openai.com/api/docs/quickstart)).
- `anthropic_api` : Claude/Anthropic via `ANTHROPIC_API_KEY` ; nécessite un compte Claude Platform et une clé API ([aperçu officiel de l'API](https://platform.claude.com/docs/fr/api/overview)).

Ces modes sont utiles si l'utilisateur n'a pas assez de mémoire GPU locale mais dispose d'un compte API.

Tous les IDs de modèles sont configurables dans le notebook et dans `src/config.py`. En usage local, les chemins doivent pointer vers des modèles disponibles sur la machine ou dans le cache Hugging Face configuré.

## 5) Lancer localement

1. Cloner le dépôt

```bash
git clone https://github.com/git-haddadz/AdDhakhiraCorpusAI.git
cd AdDhakhiraCorpusAI
```

2. Créer et activer un environnement Python

```bash
python -m venv .venv
source .venv/bin/activate
pip install -U pip
pip install -r requirements.txt
```

3. Créer la configuration locale

Copier le template puis l'adapter à la machine :

```bash
cp src/config_template.py src/config.py
```

Pour `default`, renseigner :

- `MODEL_EXTRACTOR_PATH`
- `MODEL_REASONER_PATH`
- `EMBEDDING_MODEL`
- `NUM_GPUS_EXTRACTOR`
- `NUM_GPUS_REASONER`

Pour les modes API, renseigner :

- `LLM_BACKEND`
- `MODEL_EXTRACTOR_PATH`
- `MODEL_REASONER_PATH`
- la clé API correspondante

4. Préparer l'index dense FAISS

La pipeline par défaut utilise le retrieval dense avec `Qwen/Qwen3-Embedding-4B`. Construire l'index une fois avant de lancer l'application. Relancer la commande réutilise l'index compatible existant.

```bash
python -m src.vector_index   --model "/path/to/Qwen__Qwen3-Embedding-4B"   --backend faiss   --json-input ./database   --output-dir ./database/vector_indexes
```

Utiliser le même chemin ou ID de modèle que `EMBEDDING_MODEL` dans `src/config.py`.

5. Lancer le point d'entrée principal

```bash
python -u main.py   --question "كيف تكون صلاة الجنازة عند المالكية؟"   --output "./outputs/output_local.html"   --diagnostic-coherence
```

## 6) Démo - démarrage rapide

Pour Google Colab, utiliser le notebook web app :

- Google Colab : [AdDhakhira_WebApp.ipynb](https://colab.research.google.com/github/git-haddadz/AdDhakhiraCorpusAI/blob/main/AdDhakhira_WebApp.ipynb)

Le notebook propose une liste déroulante :

- `default`
- `lite_version`
- Gemini API
- ChatGPT/OpenAI API
- Claude/Anthropic API

Quand un mode API est sélectionné, le champ de clé correspondant apparaît. Les modes API remplacent seulement l'extractor et le reasoner ; l'embedding, le retrieval, les pages sélectionnées, le format de sortie, la sauvegarde Drive et le téléchargement local HTML restent identiques.

Les champs modèles peuvent être modifiés dans le notebook avant l'initialisation.

## 7) Le projet est-il modifiable ?

Oui.

Le projet est open-source et conçu pour être forké, amélioré et adapté, tant que les usages dérivés restent libres et open-source. Adapter l'outil à un corpus d'un autre madhhab est explicitement encouragé.
