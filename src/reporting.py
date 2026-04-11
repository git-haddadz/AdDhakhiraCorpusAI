import html
import re
from typing import Dict, List, Optional


def format_duration(seconds: float) -> str:
    total = int(round(seconds))
    minutes = total // 60
    secs = total % 60
    return f"{minutes:02d}min {secs:02d} s"


def build_timing_message(seconds: float) -> str:
    return f"L'Assistant a produit la recherche en {format_duration(seconds)}"


def _h(value: object) -> str:
    return html.escape("" if value is None else str(value), quote=True)


def _render_consistency_html(consistency_diagnostic: Optional[Dict[str, str]]) -> str:
    if not consistency_diagnostic:
        return ""
    rows = []
    keys = [
        ("Traduction question activée", "question_translation_enabled"),
        ("Traduction question appliquée", "question_translation_applied"),
        ("Question utilisée par la pipeline", "question_pipeline_used"),
        ("Verdict initial", "initial_verdict"),
        ("Génération structurée échouée", "generation_failed"),
        ("Verdict initial (primaire)", "initial_primary_verdict"),
        ("Verdict initial (adversarial)", "initial_adversarial_verdict"),
        ("Verdict après régénération", "retry_verdict"),
        ("Verdict retry (primaire)", "retry_primary_verdict"),
        ("Verdict retry (adversarial)", "retry_adversarial_verdict"),
        ("Verdict passe finale", "final_pass_verdict"),
        ("Verdict passe finale (primaire)", "final_pass_primary_verdict"),
        ("Verdict passe finale (adversarial)", "final_pass_adversarial_verdict"),
        ("Fallback not_enough_context", "fallback_used"),
    ]
    for label, key in keys:
        if key in consistency_diagnostic and consistency_diagnostic.get(key) is not None:
            rows.append(
                f"<tr><th>{_h(label)}</th><td>{_h(consistency_diagnostic.get(key))}</td></tr>"
            )

    issue_blocks = []
    for label, key in [
        ("Issues initiales", "initial_issues"),
        ("Issues après régénération", "retry_issues"),
        ("Issues passe finale", "final_pass_issues"),
    ]:
        issues = consistency_diagnostic.get(key, [])
        if issues:
            items = "".join(f"<li>{_h(issue)}</li>" for issue in issues)
            issue_blocks.append(f"<h4>{_h(label)}</h4><ul>{items}</ul>")

    issue_html = "".join(issue_blocks) if issue_blocks else "<p>Aucune issue.</p>"
    return f"""
    <section class="tab-panel" id="panel-debug" role="tabpanel">
      <div class="card">
        <h3>Diagnostic Cohérence</h3>
        <div class="table-wrap">
          <table class="diag-table">
            <tbody>
              {''.join(rows)}
            </tbody>
          </table>
        </div>
        <div class="issues-block">
          {issue_html}
        </div>
      </div>
    </section>
    """


def _normalize_digits(text: str) -> str:
    arabic_indic = "٠١٢٣٤٥٦٧٨٩"
    eastern_arabic_indic = "۰۱۲۳۴۵۶۷۸۹"
    trans = {}
    for i, d in enumerate(arabic_indic):
        trans[ord(d)] = ord(str(i))
    for i, d in enumerate(eastern_arabic_indic):
        trans[ord(d)] = ord(str(i))
    return text.translate(trans)


def _build_source_indexes(source_page_map: Optional[Dict[str, Dict[str, str]]]):
    by_pair: Dict[str, Dict[str, str]] = {}
    by_page_id: Dict[str, Dict[str, str]] = {}
    by_page_number: Dict[str, Dict[str, str]] = {}
    if not source_page_map:
        return by_pair, by_page_id, by_page_number

    for key, info in source_page_map.items():
        normalized_key = _normalize_digits(key.strip())
        by_pair[normalized_key] = info
        page_id = _normalize_digits(str(info.get("page_id", "")).strip())
        page_number = _normalize_digits(str(info.get("page_number", "")).strip())
        if page_id and page_id not in by_page_id:
            by_page_id[page_id] = info
        if page_number and page_number not in by_page_number:
            by_page_number[page_number] = info
    return by_pair, by_page_id, by_page_number


def _parse_source_refs(source_text: str) -> List[Dict[str, str]]:
    if not source_text:
        return []
    normalized = _normalize_digits(source_text)
    page_pattern = re.compile(
        r"Page\s*([0-9]+)\s*[\(\[]\s*page_id\s*=\s*([^\)\]\s,;]+)\s*[\)\]]",
        flags=re.IGNORECASE,
    )
    kv_pattern = re.compile(
        r"page_number\s*=\s*([0-9]+)\s+page_id\s*=\s*([^\s,;]+)",
        flags=re.IGNORECASE,
    )
    refs = []
    for page_number, page_id in page_pattern.findall(normalized):
        refs.append({"page_number": page_number.strip(), "page_id": page_id.strip()})
    for page_number, page_id in kv_pattern.findall(normalized):
        refs.append({"page_number": page_number.strip(), "page_id": page_id.strip()})
    return refs


def _resolve_ref(
    ref: Dict[str, str],
    source_page_map: Optional[Dict[str, Dict[str, str]]],
) -> Optional[Dict[str, str]]:
    by_pair, by_page_id, by_page_number = _build_source_indexes(source_page_map)
    page_number = _normalize_digits(str(ref.get("page_number", "")).strip())
    page_id = _normalize_digits(str(ref.get("page_id", "")).strip())

    exact = by_pair.get(f"{page_number}|{page_id}")
    if exact:
        return exact
    if page_id and page_id in by_page_id:
        return by_page_id[page_id]
    if page_number and page_number in by_page_number:
        return by_page_number[page_number]
    return None


def _infer_point_source_info(source_text: str, source_page_map: Optional[Dict[str, Dict[str, str]]]) -> Optional[Dict[str, str]]:
    if not source_text or not source_page_map:
        return None

    refs = _parse_source_refs(source_text)
    for ref in refs:
        info = _resolve_ref(ref, source_page_map)
        if info is not None:
            return info
    return None


def render_source_reference(source_text: str, source_page_map: Optional[Dict[str, Dict[str, str]]]) -> str:
    if not source_text:
        return "Référence indisponible."
    if not source_page_map:
        return source_text

    refs = _parse_source_refs(source_text)
    if not refs:
        return source_text

    rendered = []
    seen = set()
    for ref in refs:
        page_number = ref["page_number"]
        page_id = ref["page_id"]
        key = f"{page_number}|{page_id}"
        if key in seen:
            continue
        seen.add(key)
        page_info = _resolve_ref(ref, source_page_map)
        if page_info:
            section_suffix = (
                f", Section: {page_info['section_path']}" if page_info.get("section_path") else ""
            )
            source_prefix = f"[{page_info['source_id']}] " if page_info.get("source_id") else ""
            rendered.append(
                f"{source_prefix}{page_info['title']} ({page_info['author']}), "
                f"Page {page_info['page_number']} (page_id={page_info['page_id']}){section_suffix}"
            )
        else:
            rendered.append(f"Page {page_number} (page_id={page_id}, hors-contexte récupéré)")
    return " | ".join(rendered)


def format_fatwa(
    answer: Dict,
    source_page_map: Optional[Dict[str, Dict[str, str]]] = None,
) -> str:
    status = answer.get("status", "not_enough_context")
    reponse_courte = answer.get("reponse_courte", "")
    points = answer.get("points", [])
    limites = answer.get("limites", "")
    author = "Auteur inconnu"
    title = "Livre inconnu"

    lines = []
    lines.append("Avis synthétique")
    lines.append(reponse_courte or "Aucune conclusion n'a pu être formulée.")
    lines.append("")

    if status == "not_enough_context":
        lines.append("Niveau de certitude")
        lines.append("Les extraits retrouvés ne suffisent pas pour trancher de façon explicite.")
        lines.append("")

    lines.append("Preuves textuelles")
    if points:
        for i, p in enumerate(points, start=1):
            citation_text = p.get("citation_arabe", "")
            point_source_info = _infer_point_source_info(p.get("source", ""), source_page_map)
            point_author = author
            point_title = title
            if point_source_info:
                point_author = point_source_info.get("author", point_author)
                point_title = point_source_info.get("title", point_title)
            lines.append(f"{i}. {p.get('titre', 'Point')}")
            lines.append(f"   Citation arabe: {point_author} dit dans {point_title} : {citation_text}")
            lines.append(f"   Explication: {p.get('explication_fr', '')}")
            lines.append(f"   Référence: {render_source_reference(p.get('source', ''), source_page_map)}")
    else:
        lines.append("Aucune preuve textuelle explicite n'a été retournée.")
    lines.append("")

    lines.append("Limites de la réponse")
    lines.append(limites or "Aucune limite supplémentaire signalée.")
    return "\n".join(lines)


def print_final(
    question: str,
    keywords: List[str],
    top_pages: List[Dict],
    answer: Dict,
    source_page_map: Optional[Dict[str, Dict[str, str]]] = None,
    consistency_diagnostic: Optional[Dict[str, str]] = None,
) -> str:
    status = answer.get("status", "not_enough_context")
    reponse_courte = answer.get("reponse_courte", "")
    points = answer.get("points", []) or []
    limites = answer.get("limites", "")

    keyword_html = "".join(f"<span class='chip'>{_h(kw)}</span>" for kw in keywords)
    certitude_html = (
        "<div class='notice'>Niveau de certitude: les extraits sont insuffisants pour trancher explicitement.</div>"
        if status == "not_enough_context"
        else ""
    )

    points_html = ""
    if points:
        blocks = []
        for i, p in enumerate(points, start=1):
            point_source_info = _infer_point_source_info(p.get("source", ""), source_page_map)
            point_author = point_source_info.get("author", "Auteur inconnu") if point_source_info else "Auteur inconnu"
            point_title = point_source_info.get("title", "Livre inconnu") if point_source_info else "Livre inconnu"
            blocks.append(
                f"""
                <article class="proof">
                  <h4>{i}. {_h(p.get('titre', 'Point'))}</h4>
                  <p><strong>Citation arabe:</strong> {_h(point_author)} dit dans {_h(point_title)} :</p>
                  <blockquote dir="rtl" lang="ar">{_h(p.get('citation_arabe', ''))}</blockquote>
                  <p><strong>Explication:</strong> {_h(p.get('explication_fr', ''))}</p>
                  <p><strong>Référence:</strong> {_h(render_source_reference(p.get('source', ''), source_page_map))}</p>
                </article>
                """
            )
        points_html = "".join(blocks)
    else:
        points_html = "<p>Aucune preuve textuelle explicite n'a été retournée.</p>"

    if top_pages:
        options = "".join(
            f"<option value='page-{i}'>Page {i+1} · {_h(p.get('book_title', 'Livre inconnu'))}</option>"
            for i, p in enumerate(top_pages)
        )
        page_cards = []
        for i, p in enumerate(top_pages):
            page_cards.append(
                f"""
                <article class="page-card {'active' if i == 0 else ''}" id="page-{i}">
                  <div class="meta-grid">
                    <div><span>Auteur</span><strong>{_h(p.get('author', 'Auteur inconnu'))}</strong></div>
                    <div><span>Livre</span><strong>{_h(p.get('book_title', 'Livre inconnu'))}</strong></div>
                    <div><span>Source</span><strong>{_h(p.get('source_id', '-'))}</strong></div>
                    <div><span>Section</span><strong>{_h(p.get('section_path', '-'))}</strong></div>
                    <div><span>Page</span><strong>{_h(p.get('page_number'))} | id {_h(p.get('page_id'))}</strong></div>
                    <div><span>Score</span><strong>{_h(f"{p.get('score', 0):.4f}")}</strong></div>
                  </div>
                  <details open>
                    <summary>Texte arabe</summary>
                    <pre dir="rtl" lang="ar">{_h(p.get('text', ''))}</pre>
                  </details>
                </article>
                """
            )
        pages_html = f"""
        <section class="tab-panel" id="panel-pages" role="tabpanel">
          <div class="card">
            <h3>Pages Récupérées</h3>
            <label class="select-label" for="page-select">Choisir une page</label>
            <select id="page-select">{options}</select>
            <div class="pages-wrap">
              {''.join(page_cards)}
            </div>
          </div>
        </section>
        """
    else:
        pages_html = """
        <section class="tab-panel" id="panel-pages" role="tabpanel">
          <div class="card"><h3>Pages Récupérées</h3><p>Aucune page récupérée.</p></div>
        </section>
        """

    debug_tab = ""
    debug_panel = ""
    if consistency_diagnostic:
        debug_tab = "<button class='tab-btn' data-tab='debug'>Debug</button>"
        debug_panel = _render_consistency_html(consistency_diagnostic)

    return f"""<!doctype html>
<html lang="fr">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Rapport Fiqh RAG</title>
  <style>
    :root {{
      --bg: #f5f7fb; --card: #ffffff; --text: #0f172a; --muted: #64748b;
      --border: #e2e8f0; --accent: #0ea5e9; --accent-soft: #e0f2fe; --ok: #10b981;
    }}
    * {{ box-sizing: border-box; }}
    body {{ margin: 0; font-family: ui-sans-serif, system-ui, -apple-system, Segoe UI, Roboto, Arial, sans-serif; background: var(--bg); color: var(--text); }}
    .app {{ max-width: 980px; margin: 0 auto; padding: 16px; }}
    .hero {{ background: linear-gradient(135deg, #082f49 0%, #0f766e 100%); color: #fff; border-radius: 16px; padding: 16px; margin-bottom: 14px; }}
    .hero h1 {{ margin: 0 0 10px; font-size: 1.15rem; }}
    .hero p {{ margin: 0; opacity: .95; line-height: 1.45; }}
    .chips {{ display: flex; flex-wrap: wrap; gap: 8px; margin-top: 10px; }}
    .chip {{ background: rgba(255,255,255,.2); border: 1px solid rgba(255,255,255,.25); border-radius: 999px; padding: 4px 10px; font-size: .8rem; }}
    .tabs {{ display: flex; gap: 8px; margin-bottom: 10px; position: sticky; top: 0; padding: 8px 0; background: var(--bg); z-index: 20; }}
    .tab-btn {{ border: 1px solid var(--border); background: #fff; color: var(--text); border-radius: 10px; padding: 8px 12px; font-weight: 600; font-size: .9rem; }}
    .tab-btn.active {{ background: var(--accent-soft); border-color: var(--accent); color: #075985; }}
    .tab-panel {{ display: none; }}
    .tab-panel.active {{ display: block; }}
    .card {{ background: var(--card); border: 1px solid var(--border); border-radius: 14px; padding: 14px; margin-bottom: 12px; }}
    h3 {{ margin: 0 0 10px; font-size: 1rem; }}
    .notice {{ background: #fff7ed; color: #9a3412; border: 1px solid #fdba74; border-radius: 10px; padding: 10px; margin-bottom: 10px; font-size: .9rem; }}
    .proof {{ border: 1px solid var(--border); border-radius: 12px; padding: 10px; margin-bottom: 10px; }}
    .proof h4 {{ margin: 0 0 8px; font-size: .95rem; }}
    blockquote {{ margin: 8px 0; padding: 10px; border-radius: 10px; background: #f8fafc; border-right: 3px solid var(--accent); line-height: 1.8; }}
    .select-label {{ display: block; font-size: .82rem; color: var(--muted); margin-bottom: 6px; }}
    select {{ width: 100%; border: 1px solid var(--border); border-radius: 10px; padding: 10px; font-size: .95rem; margin-bottom: 10px; background: #fff; }}
    .page-card {{ display: none; }}
    .page-card.active {{ display: block; }}
    .meta-grid {{ display: grid; grid-template-columns: 1fr 1fr; gap: 8px; margin-bottom: 10px; }}
    .meta-grid span {{ display: block; font-size: .75rem; color: var(--muted); }}
    .meta-grid strong {{ font-size: .85rem; display: block; line-height: 1.3; }}
    details {{ border: 1px solid var(--border); border-radius: 10px; overflow: hidden; }}
    summary {{ cursor: pointer; padding: 8px 10px; background: #f8fafc; font-weight: 600; font-size: .9rem; }}
    pre {{ margin: 0; padding: 10px; white-space: pre-wrap; word-wrap: break-word; line-height: 1.75; font-size: .92rem; }}
    .table-wrap {{ overflow-x: auto; }}
    .diag-table {{ width: 100%; border-collapse: collapse; }}
    .diag-table th, .diag-table td {{ border: 1px solid var(--border); padding: 8px; text-align: left; font-size: .88rem; vertical-align: top; }}
    .diag-table th {{ width: 42%; background: #f8fafc; }}
    .issues-block h4 {{ margin: 12px 0 6px; font-size: .9rem; }}
    .issues-block ul {{ margin: 0; padding-left: 18px; }}
    @media (max-width: 640px) {{
      .app {{ padding: 10px; }}
      .hero {{ border-radius: 12px; }}
      .meta-grid {{ grid-template-columns: 1fr; }}
      .tabs {{ overflow-x: auto; }}
      .tab-btn {{ white-space: nowrap; }}
    }}
  </style>
</head>
<body>
  <main class="app">
    <header class="hero">
      <h1>Assistant IA de recherche bibliographique en Fiqh Malikite par RAG et LLM</h1>
      <p><strong>Question:</strong> {_h(question)}</p>
      <div class="chips">{keyword_html}</div>
    </header>

    <nav class="tabs" role="tablist" aria-label="Sections rapport">
      <button class="tab-btn active" data-tab="answer">Réponse</button>
      <button class="tab-btn" data-tab="pages">Pages</button>
      {debug_tab}
    </nav>

    <section class="tab-panel active" id="panel-answer" role="tabpanel">
      <div class="card">
        <h3>Avis Synthétique</h3>
        <p>{_h(reponse_courte or "Aucune conclusion n'a pu être formulée.")}</p>
        {certitude_html}
      </div>
      <div class="card">
        <h3>Preuves Textuelles</h3>
        {points_html}
      </div>
      <div class="card">
        <h3>Limites de la Réponse</h3>
        <p>{_h(limites or "Aucune limite supplémentaire signalée.")}</p>
      </div>
    </section>

    {pages_html}
    {debug_panel}
  </main>
  <script>
    (function() {{
      const tabButtons = Array.from(document.querySelectorAll('.tab-btn'));
      const panels = {{
        answer: document.getElementById('panel-answer'),
        pages: document.getElementById('panel-pages'),
        debug: document.getElementById('panel-debug')
      }};
      tabButtons.forEach((btn) => {{
        btn.addEventListener('click', () => {{
          tabButtons.forEach((b) => b.classList.remove('active'));
          btn.classList.add('active');
          const tab = btn.getAttribute('data-tab');
          Object.keys(panels).forEach((key) => {{
            if (!panels[key]) return;
            panels[key].classList.toggle('active', key === tab);
          }});
        }});
      }});

      const pageSelect = document.getElementById('page-select');
      if (pageSelect) {{
        const showPage = (id) => {{
          document.querySelectorAll('.page-card').forEach((card) => {{
            card.classList.toggle('active', card.id === id);
          }});
        }};
        pageSelect.addEventListener('change', (e) => showPage(e.target.value));
        showPage(pageSelect.value);
      }}
    }})();
  </script>
</body>
</html>"""


def print_startup_message() -> None:
    print(
        "Bienvenue. L'assistant démarre, prépare l'environnement puis charge les modèles. Merci de patienter...",
        flush=True,
    )


def write_output_file(content: str, output_path: str = "output.txt") -> None:
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(content)


def write_output_with_timing(content: str, elapsed_seconds: float, output_path: str = "output.txt") -> None:
    timing_message = build_timing_message(elapsed_seconds)

    if "<!doctype html>" in content and "</header>" in content:
        timing_badge = (
            f"<p style=\"margin:10px 0 0;font-size:.86rem;opacity:.92;\">"
            f"<strong>Temps de traitement:</strong> {_h(format_duration(elapsed_seconds))}"
            f"</p>"
        )
        content = content.replace("</header>", f"      {timing_badge}\n    </header>", 1)
    else:
        content = f"{content}\n\n{timing_message}\n"

    write_output_file(content, output_path=output_path)
