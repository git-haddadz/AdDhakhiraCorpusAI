import argparse
import html
import importlib
import os
import queue
import sys
import tempfile
import threading
import time
import traceback
import urllib.parse
from datetime import datetime
from pathlib import Path
from typing import Dict, List

from src import config as base_config
from src.reporting import write_output_with_timing


BACKEND_CHOICES = [
    ("Default", "default"),
    ("Gemini API", "gemini_api"),
    ("ChatGPT / OpenAI API", "openai_api"),
    ("Claude / Anthropic API", "anthropic_api"),
]

API_KEY_BY_BACKEND = {
    "gemini_api": "GEMINI_API_KEY",
    "openai_api": "OPENAI_API_KEY",
    "anthropic_api": "ANTHROPIC_API_KEY",
}

DEFAULT_API_MODELS = {
    "gemini_api": "gemini-2.5-flash",
    "openai_api": "gpt-4.1",
    "anthropic_api": "claude-sonnet-4-20250514",
}

PIPELINE_MODULE_PREFIXES = (
    "src.pipeline",
    "src.llm_ops",
    "src.llm_backend",
    "src.retrieval",
)

_pipeline_lock = threading.Lock()


def _source_pairs(pages: List[Dict[str, object]]) -> List[Dict[str, str]]:
    seen = set()
    sources = []
    for page in pages:
        author = str(page.get("author") or "Auteur inconnu").strip()
        title = str(page.get("book_title") or "Livre inconnu").strip()
        page_number = str(page.get("page_number") or "").strip()
        key = (author, title, page_number)
        if key in seen:
            continue
        seen.add(key)
        sources.append({"author": author, "title": title, "page_number": page_number})
    return sources


def _authors_summary(pages: List[Dict[str, object]]) -> str:
    authors = []
    for page in pages:
        author = str(page.get("author") or "Auteur inconnu").strip()
        if author and author not in authors:
            authors.append(author)
        if len(authors) >= 3:
            break
    return ", ".join(authors) if authors else "les auteurs retrouvés"


def _idle_status_message(stage: str, sources: List[Dict[str, str]], index: int) -> str:
    if not sources:
        if stage == "queued":
            fillers = [
                "Votre question est actuellement en queue. Veuillez laisser cette page ouverte ; une notification pourra vous prévenir quand l'assistant commencera à la traiter.",
                "Une autre analyse est en cours. Votre demande est conservée et démarrera automatiquement ensuite ; gardez cette page ouverte.",
                "La file d'attente protège le budget GPU : une seule analyse tourne à la fois.",
                "Votre question attend son tour pour éviter de lancer plusieurs GPU en parallèle.",
                "Merci de garder cette page ouverte : l'assistant démarrera automatiquement dès que possible.",
                "La requête précédente utilise encore le modèle ; votre question prendra la suite.",
                "Votre demande est bien reçue. Elle reste en attente tant que l'analyse précédente n'est pas terminée.",
                "Je garde votre question en attente pour ne pas multiplier les coûts GPU.",
                "Si votre navigateur l'autorise, une notification apparaîtra quand l'assistant commencera votre question.",
                "Vous pouvez patienter sur cette page : l'assistant prendra votre question dès que la place sera libre.",
            ]
        elif stage in {"startup", "init", "started"}:
            fillers = [
                "Bienvenue. L'assistant démarre, prépare l'environnement puis charge les modèles. Merci de patienter...",
                "Je réserve les ressources nécessaires et je prépare le chargement des modèles.",
                "Je charge les composants de recherche et les paramètres de l'instance.",
                "Je prépare le modèle chargé de comprendre la question.",
                "Je vérifie que l'environnement est prêt avant de lancer l'analyse.",
                "Je mets en place la session de travail pour traiter la question.",
                "Je laisse les modèles se charger correctement avant de commencer la recherche.",
                "Je prépare la mémoire et les outils nécessaires à l'analyse.",
                "Je configure le moteur d'inférence pour cette requête.",
                "Je stabilise l'environnement avant de passer à l'interprétation de la question.",
                "Je prépare les bibliothèques nécessaires au raisonnement.",
                "Je vérifie que les chemins des modèles et de l'index sont accessibles.",
                "Je mets en place le contexte d'exécution de la requête.",
                "Je charge progressivement les poids du modèle en mémoire.",
                "Je prépare le moteur qui servira à extraire les premiers indices.",
                "Je synchronise les composants de recherche avant de commencer.",
                "Je m'assure que les ressources GPU sont disponibles pour cette question.",
                "Je prépare la session pour éviter d'interrompre l'analyse en cours.",
                "Je rassemble les paramètres de génération et de récupération.",
                "Je prépare l'environnement Python utilisé par l'assistant.",
                "Je charge les outils qui liront la question puis interrogeront le corpus.",
                "Je vérifie la configuration de l'instance avant de lancer le traitement.",
                "Je prépare le passage entre l'interface et l'assistant de recherche.",
                "Je laisse le modèle terminer son initialisation.",
                "Je prépare les caches utiles au traitement de la question.",
                "Je vérifie que l'index et les modèles locaux répondent correctement.",
                "Je prépare la requête pour qu'elle soit traitée dans le bon ordre.",
                "Je charge les composants nécessaires sans encore interroger les textes.",
                "Je mets en place les outils de lecture et de sélection des passages.",
                "Je patiente le temps que le moteur soit entièrement prêt.",
                "Je prépare la chaîne d'analyse avant la première interprétation.",
                "Je vérifie que la session dispose des modèles attendus.",
                "Je charge le modèle d'extraction avant de chercher dans le corpus.",
                "Je prépare l'espace de travail de cette réponse.",
                "Je termine la préparation technique avant l'analyse de la question.",
            ]
        elif stage == "question":
            fillers = [
                "Je prépare l'interprétation de la question pour la recherche en arabe.",
                "Je cherche les formulations arabes les plus utiles pour interroger les textes.",
                "Je transforme la question en axes exploitables pour la recherche.",
                "Je repère les notions principales avant de lancer la récupération des passages.",
                "Je prépare les mots-clés qui guideront la recherche dans le corpus.",
                "Je clarifie la demande pour éviter une recherche trop large.",
                "Je distingue les termes importants des détails moins utiles pour la recherche.",
                "Je prépare une formulation qui corresponde mieux aux textes arabes.",
                "Je cherche les notions juridiques qui structurent la question.",
                "Je reformule la demande en indices exploitables par la recherche.",
                "Je vérifie que l'interprétation conserve le sens de la question.",
                "Je prépare la transition entre la question et les mots-clés arabes.",
                "Je cherche les termes qui auront le plus de chances d'apparaître dans le corpus.",
                "Je réduis la question à ses éléments de recherche principaux.",
                "Je garde les conditions importantes de la question pendant l'interprétation.",
                "Je prépare une recherche assez précise pour éviter les passages hors sujet.",
                "Je transforme la demande en pistes de recherche textuelle.",
                "Je repère les termes qui peuvent varier selon les auteurs.",
            ]
        elif stage == "keywords":
            fillers = [
                "Je prépare la recherche à partir des mots-clés identifiés.",
                "Je pondère les axes de recherche avant d'interroger l'index.",
                "Je vérifie que les mots-clés sont assez précis pour retrouver les bons passages.",
                "Je passe des mots-clés à la recherche dans le corpus.",
                "Je trie les mots-clés pour mieux guider la récupération des pages.",
                "Je prépare la combinaison entre recherche lexicale et recherche dense.",
                "Je vérifie que les termes retenus couvrent bien la question.",
                "Je transforme les mots-clés en requête de recherche.",
                "Je prépare les variantes utiles avant de parcourir l'index.",
                "Je m'assure que les mots-clés restent proches du sujet demandé.",
                "Je prépare les signaux qui serviront à classer les passages.",
                "Je relie les mots-clés entre eux pour orienter la recherche.",
                "Je prépare la recherche dans les titres, sections et pages.",
                "Je vérifie que les mots-clés ne tirent pas la recherche trop loin du sujet.",
                "Je prépare le classement initial des passages candidats.",
                "Je passe de l'analyse de la question à la recherche documentaire.",
            ]
        elif stage == "retrieval":
            fillers = [
                "Je parcours l'index pour faire remonter les passages les plus proches.",
                "Je compare les scores des passages candidats.",
                "Je filtre les résultats pour garder les pages les plus utiles.",
                "Je rapproche la question des textes indexés.",
                "Je cherche les pages qui répondent le plus directement à la question.",
                "Je compare la recherche par mots-clés avec les résultats de similarité.",
                "Je fais remonter les passages les mieux alignés avec les axes de recherche.",
                "Je regroupe les passages proches pour éviter les doublons inutiles.",
                "Je prépare la sélection des pages qui seront lues par le modèle de raisonnement.",
                "Je vérifie que les résultats retrouvés ne sont pas seulement des correspondances superficielles.",
                "Je classe les passages candidats selon leur proximité avec la question.",
                "Je filtre les pages éditoriales ou peu utiles quand elles apparaissent.",
                "Je cherche les extraits qui donnent une base textuelle exploitable.",
                "Je prépare un petit dossier de sources avant la rédaction.",
                "Je compare les pages candidates pour garder les plus informatives.",
                "Je sélectionne les passages qui méritent d'être analysés plus longuement.",
                "Je vérifie que les pages retenues couvrent bien les termes importants.",
                "Je rassemble les résultats les plus solides avant de passer au raisonnement.",
            ]
        else:
            fillers = [
                "Je poursuis le traitement de la question.",
                "Je prépare l'étape suivante de l'analyse.",
                "Je laisse le calcul se terminer avant d'afficher le résultat.",
            ]
        return fillers[index % len(fillers)]

    source = sources[index % len(sources)]
    other = sources[(index + 1) % len(sources)]
    author = source["author"]
    title = source["title"]
    page = source.get("page_number")
    other_author = other["author"]
    other_title = other["title"]
    page_suffix = f", page {page}" if page else ""
    templates = [
        f"J'analyse maintenant un passage de {title}{page_suffix}.",
        f"Je vérifie comment {author} aborde ce point.",
        f"Je condense un passage de {author} dans la réponse.",
        f"Je compare ce que j'ai noté avec {other_title} de {other_author}.",
        f"Je relie les extraits de {author} au reste des sources sélectionnées.",
        f"Je contrôle que la formulation reste fidèle à {title}.",
        "Je compare les extraits retenus avant de rédiger la réponse.",
        "Je trie les éléments les plus utiles pour éviter une réponse trop large.",
        f"Je relis {title}{page_suffix} pour isoler ce qui répond directement à la question.",
        f"Je vérifie si le passage de {author} permet une réponse ferme ou seulement conditionnelle.",
        f"Je compare la formulation de {author} avec les autres passages retenus.",
        f"Je cherche la citation la plus utile dans {title}.",
        f"Je prépare une synthèse à partir du passage de {author}.",
        f"Je vérifie que la page de {title} ne dit pas plus que ce que la réponse peut affirmer.",
        f"Je rapproche ce passage de {title} des autres sources sélectionnées.",
        f"Je garde le passage de {author} comme point de contrôle pour la réponse.",
        f"Je reformule les éléments utiles de {title} en français clair.",
        f"Je vérifie que la réponse reste compatible avec {other_title}.",
        f"Je compare les passages de {author} et {other_author}.",
        f"Je cherche comment intégrer la page {page or 'retenue'} sans surcharger la réponse.",
        f"Je distingue la règle utile des détails secondaires dans {title}.",
        f"Je vérifie que les citations retenues restent attachées à leurs pages.",
        f"Je prépare les sources qui seront visibles dans le rapport final.",
        f"Je contrôle la cohérence entre les passages de {title} et de {other_title}.",
        f"Je rassemble les points vraiment exploitables chez {author}.",
        f"Je vérifie que l'analyse ne dépasse pas ce que dit {title}.",
        f"Je compare les indications de {author} avec la question posée.",
        f"Je prépare une réponse courte avant de détailler les preuves.",
        f"Je cherche l'équilibre entre précision juridique et lisibilité.",
        f"Je vérifie que chaque affirmation importante pourra être rattachée à une source.",
        f"Je trie les citations pour garder celles qui portent le mieux la réponse.",
        f"Je relis les passages retenus pour éviter une conclusion trop rapide.",
        f"Je vérifie les limites de ce que les extraits permettent d'affirmer.",
        f"Je prépare la structure de la réponse à partir des sources retrouvées.",
        f"Je compare les éléments concordants avant de formuler la conclusion.",
        f"Je garde les passages de {author} en référence pendant la rédaction.",
        f"Je vérifie que les détails de la question sont bien pris en compte.",
        f"Je condense les éléments utiles sans ajouter de connaissance extérieure.",
        f"Je prépare la réponse finale en respectant les pages récupérées.",
        f"Je vérifie que le raisonnement reste aligné avec les extraits.",
        f"Je relie la citation arabe à son explication en français.",
        f"Je compare les preuves disponibles avant de choisir la formulation finale.",
    ]
    return templates[index % len(templates)]


def _config_value(name: str, default=None):
    return getattr(base_config, name, default)


def _model_for_backend(backend: str) -> str:
    current_backend = _config_value("LLM_BACKEND", "default")
    if current_backend == backend and backend != "default":
        return str(_config_value("MODEL_REASONER_PATH", DEFAULT_API_MODELS[backend]))
    return DEFAULT_API_MODELS[backend]


def _config_api_key_for_backend(backend: str) -> str:
    key_name = API_KEY_BY_BACKEND.get(backend)
    if not key_name:
        return ""
    return str(_config_value(key_name, "") or "").strip()


def _output_dir() -> Path:
    directory = Path(_config_value("REPO_ROOT", Path.cwd())) / "outputs" / "web_app"
    directory.mkdir(parents=True, exist_ok=True)
    return directory


def _allowed_paths() -> List[str]:
    output_dir = _output_dir()
    gradio_tmp = Path(tempfile.gettempdir()) / "gradio"
    gradio_tmp.mkdir(parents=True, exist_ok=True)
    return [str(output_dir), str(gradio_tmp)]


def _download_button_html(file_path: object) -> str:
    if not file_path:
        return ""
    path = str(file_path)
    file_url = f"/gradio_api/file={urllib.parse.quote(path, safe='/:')}"
    return (
        '<a class="download-synthesis-link" '
        f'href="{html.escape(file_url, quote=True)}" download>'
        "Télécharger la synthèse bibliographique"
        "</a>"
    )


def _api_key_for_backend(backend: str, api_key: str) -> str:
    key_name = API_KEY_BY_BACKEND.get(backend)
    if not key_name:
        return ""
    return (
        (api_key or "").strip()
        or str(_config_value(key_name, "") or "").strip()
        or os.environ.get(key_name, "").strip()
    )


def _runtime_config(
    backend: str,
    api_key: str,
    dense_retrieval: bool,
) -> Dict[str, object]:
    cfg = {
        "LLM_BACKEND": backend,
        "ENABLE_DENSE_RETRIEVAL": bool(dense_retrieval),
        "GEMINI_API_KEY": "",
        "OPENAI_API_KEY": "",
        "ANTHROPIC_API_KEY": "",
        "MODEL_EXTRACTOR_PATH": _config_value("MODEL_EXTRACTOR_PATH"),
        "MODEL_REASONER_PATH": _config_value("MODEL_REASONER_PATH"),
    }
    if backend != "default":
        model_name = _model_for_backend(backend)
        key_name = API_KEY_BY_BACKEND[backend]
        cfg.update(
            {
                key_name: _api_key_for_backend(backend, api_key),
                "MODEL_EXTRACTOR_PATH": model_name,
                "MODEL_REASONER_PATH": model_name,
            }
        )
    return cfg


def _apply_runtime_config(cfg: Dict[str, object]) -> None:
    for key, value in cfg.items():
        setattr(base_config, key, value)


def _clear_pipeline_modules() -> None:
    for name in list(sys.modules):
        if name == "src.config":
            continue
        if name.startswith(PIPELINE_MODULE_PREFIXES):
            sys.modules.pop(name, None)


def _run_question(
    backend: str,
    api_key: str,
    dense_retrieval: bool,
    question: str,
):
    import gradio as gr

    question = (question or "").strip()
    if not question:
        yield "Saisis une question.", "", ""
        return
    if backend != "default" and not _api_key_for_backend(backend, api_key):
        yield "La clé API est requise pour ce backend.", "", ""
        return

    cfg = _runtime_config(
        backend=backend,
        api_key=api_key,
        dense_retrieval=dense_retrieval,
    )
    events: "queue.Queue[Dict[str, object]]" = queue.Queue()
    result: Dict[str, object] = {}

    def progress_callback(event: Dict[str, object]) -> None:
        events.put(event)

    def run_pipeline() -> None:
        # The pipeline imports config values as module constants. Keep one run at a
        # time so requests cannot overwrite each other's runtime backend/key.
        with _pipeline_lock:
            progress_callback(
                {
                    "stage": "started",
                    "message": "Votre question démarre maintenant. L'assistant commence son analyse.",
                }
            )
            try:
                _apply_runtime_config(cfg)
                _clear_pipeline_modules()
                pipeline = importlib.import_module("src.pipeline")

                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                output_path = _output_dir() / f"answer_{timestamp}.html"
                t0 = time.time()
                report = pipeline.build_final_report(
                    question=question,
                    translate_to_french=bool(_config_value("TRANSLATE_TOP_CHUNKS_TO_FRENCH", False)),
                    diagnostic_coherence=True,
                    auto_translate_question_to_arabic=bool(
                        _config_value("AUTO_TRANSLATE_QUESTION_TO_ARABIC", False)
                    ),
                    progress_callback=progress_callback,
                )
                elapsed = time.time() - t0
                write_output_with_timing(
                    report,
                    elapsed_seconds=elapsed,
                    output_path=str(output_path),
                )
                result["status"] = f"Terminé en {elapsed:.1f}s avec {backend}. Synthèse prête au téléchargement."
                result["report"] = report
                result["output_path"] = str(output_path)
            except Exception as exc:
                traceback.print_exc()
                result["status"] = f"Erreur: {exc}"
                result["report"] = ""
                result["output_path"] = None

    sources: List[Dict[str, str]] = []
    filler_index = 0
    worker = None

    def start_worker() -> threading.Thread:
        thread = threading.Thread(target=run_pipeline, daemon=True)
        thread.start()
        return thread

    if _pipeline_lock.locked():
        current_stage = "queued"
        yield (
            "Votre question est actuellement en queue. Veuillez laisser cette page ouverte ; une notification pourra vous prévenir quand l'assistant commencera à la traiter.",
            "",
            "",
        )
        while _pipeline_lock.locked():
            time.sleep(8)
            filler_index += 1
            yield _idle_status_message(current_stage, sources, filler_index), "", ""
        worker = start_worker()
    else:
        current_stage = "startup"
        worker = start_worker()
        yield (
            "Bienvenue. L'assistant démarre, prépare l'environnement puis charge les modèles. Merci de patienter...",
            "",
            "",
        )

    while worker.is_alive() or not events.empty():
        try:
            event = events.get(timeout=8)
        except queue.Empty:
            filler_index += 1
            yield _idle_status_message(current_stage, sources, filler_index), "", ""
            continue

        pages = event.get("top_pages")
        if isinstance(pages, list):
            sources = _source_pairs(pages)

        current_stage = str(event.get("stage") or current_stage)
        message = str(event.get("message") or "").strip()
        if event.get("stage") == "pages_found" and isinstance(pages, list) and pages:
            message = f"J'ai trouvé des passages pertinents chez {_authors_summary(pages)}."
        elif event.get("stage") == "generation" and sources:
            message = _idle_status_message(current_stage, sources, filler_index)
            filler_index += 1

        if message:
            yield message, "", ""

    worker.join()
    yield (
        str(result.get("status") or "Réponse terminée."),
        str(result.get("report") or ""),
        _download_button_html(result.get("output_path")),
    )


def _toggle_backend_fields(backend: str):
    import gradio as gr

    is_api = backend != "default"
    return gr.update(visible=is_api, value=_config_api_key_for_backend(backend) if is_api else "")


def build_demo():
    import gradio as gr

    initial_backend = str(_config_value("LLM_BACKEND", "default"))
    if initial_backend not in {value for _, value in BACKEND_CHOICES}:
        initial_backend = "default"

    notify_js = """
(backend, apiKey, denseRetrieval, question) => {
    const startText = "Votre question démarre maintenant";
    const doneText = "Synthèse prête au téléchargement";
    const startNotificationTitle = "Votre question démarre";
    const startNotificationBody = "L'assistant commence maintenant son analyse.";
    const doneNotificationTitle = "Synthèse prête";
    const doneNotificationBody = "La synthèse bibliographique est prête au téléchargement.";

    if ("Notification" in window && Notification.permission === "default") {
        Notification.requestPermission().catch(() => {});
    }

    if (!window.__addhakhiraStatusObserver) {
        window.__addhakhiraStartNotificationSent = false;
        window.__addhakhiraDoneNotificationSent = false;
        const notifyWhenReady = () => {
            const status = document.querySelector("#run-status");
            if (!status) {
                return false;
            }
            window.__addhakhiraStatusObserver = new MutationObserver(() => {
                const text = status.innerText || status.textContent || "";
                if (!window.__addhakhiraStartNotificationSent && text.includes(startText)) {
                    window.__addhakhiraStartNotificationSent = true;
                    if ("Notification" in window && Notification.permission === "granted") {
                        new Notification(startNotificationTitle, { body: startNotificationBody });
                    }
                }
                if (!window.__addhakhiraDoneNotificationSent && text.includes(doneText)) {
                    window.__addhakhiraDoneNotificationSent = true;
                    if ("Notification" in window && Notification.permission === "granted") {
                        new Notification(doneNotificationTitle, { body: doneNotificationBody });
                    }
                }
            });
            window.__addhakhiraStatusObserver.observe(status, {
                childList: true,
                subtree: true,
                characterData: true,
            });
            return true;
        };

        if (!notifyWhenReady()) {
            setTimeout(notifyWhenReady, 1000);
        }
    } else {
        window.__addhakhiraStartNotificationSent = false;
        window.__addhakhiraDoneNotificationSent = false;
    }

    return [backend, apiKey, denseRetrieval, question];
}
"""

    css = """
.download-synthesis-link {
    display: flex;
    align-items: center;
    min-height: 64px;
    width: 100%;
    justify-content: center;
    font-size: 1.05rem;
    font-weight: 700;
    border-radius: 8px;
    color: white !important;
    background: #0f766e;
    text-decoration: none !important;
    margin: 12px 0 18px;
}
.download-synthesis-link:hover {
    background: #115e59;
}
"""

    with gr.Blocks(title="AdDhakhiraCorpusAI", css=css) as demo:
        gr.Markdown("## Assistant de recherche Malikite")
        gr.Markdown(
            """
### Options d'inférence

Vous pouvez écrire votre question indifféremment en arabe ou en français.

- `Default` : fonctionnement par défaut du chat. Il utilise les modèles locaux configurés pour cette instance.
- `Gemini API` : utilise Gemini. Pour connecter votre compte Google à cette interface, suivez le guide officiel, créez une clé `GEMINI_API_KEY`, copiez-la, puis collez-la dans le champ `Clé API` ([guide officiel des clés API](https://ai.google.dev/gemini-api/docs/api-key?hl=fr)).
- `ChatGPT / OpenAI API` : utilise ChatGPT. Pour connecter votre compte ChatGPT à cette interface, suivez le quickstart officiel, créez une clé `OPENAI_API_KEY`, copiez-la, puis collez-la dans le champ `Clé API` ([quickstart officiel, en anglais](https://developers.openai.com/api/docs/quickstart)).
- `Claude / Anthropic API` : utilise Claude. Pour connecter votre compte Claude à cette interface, suivez l'aperçu officiel de l'API, créez une clé `ANTHROPIC_API_KEY`, copiez-la, puis collez-la dans le champ `Clé API` ([aperçu officiel de l'API](https://platform.claude.com/docs/fr/api/overview)).

### Option de recherche

`Retrieval dense` : il est recommandé de laisser cette case cochée. Elle aide l'assistant à retrouver des passages proches du sens de votre question, même si les mots exacts ne sont pas les mêmes. Si vous la décochez, la recherche devient plus classique, surtout basée sur les mots-clés et les correspondances lexicales ; cela peut servir de comparaison, mais c'est généralement moins efficace.

Code source et explications détaillées : [AdDhakhiraCorpusAI](https://github.com/git-haddadz/AdDhakhiraCorpusAI). Projet expérimental de recherche assistée par IA ; consultez le dépôt pour les prérequis, les limites et les options de configuration.
"""
        )
        with gr.Row():
            backend = gr.Dropdown(
                choices=BACKEND_CHOICES,
                value=initial_backend,
                label="Backend",
            )
            dense_retrieval = gr.Checkbox(
                label="Retrieval dense",
                value=bool(_config_value("ENABLE_DENSE_RETRIEVAL", True)),
            )

        api_key = gr.Textbox(
            label="Clé API",
            value=_config_api_key_for_backend(initial_backend) if initial_backend != "default" else "",
            visible=(initial_backend != "default"),
        )

        question = gr.Textbox(label="Question", lines=4)
        submit = gr.Button("Envoyer", variant="primary")
        status = gr.Markdown(elem_id="run-status")
        download = gr.HTML()
        answer = gr.HTML()

        backend.change(
            _toggle_backend_fields,
            inputs=backend,
            outputs=api_key,
        )
        submit.click(
            _run_question,
            inputs=[
                backend,
                api_key,
                dense_retrieval,
                question,
            ],
            outputs=[status, answer, download],
            js=notify_js,
        )

    return demo


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Interface Gradio pour AdDhakhiraCorpusAI.")
    parser.add_argument("--host", default="0.0.0.0", help="Adresse d'écoute Gradio.")
    parser.add_argument("--port", type=int, default=7860, help="Port d'écoute Gradio.")
    parser.add_argument("--share", action="store_true", help="Créer un lien public Gradio temporaire.")
    parser.add_argument("--debug", action="store_true", help="Activer le mode debug Gradio.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    demo = build_demo()
    demo.queue(default_concurrency_limit=1)
    demo.launch(
        server_name=args.host,
        server_port=args.port,
        share=args.share,
        debug=args.debug,
        allowed_paths=_allowed_paths(),
    )


if __name__ == "__main__":
    main()
