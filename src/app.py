import argparse
import time

from src.config import AUTO_TRANSLATE_QUESTION_TO_ARABIC, TRANSLATE_TOP_CHUNKS_TO_FRENCH
from src.pipeline import build_final_report
from src.reporting import print_startup_message, write_output_with_timing


def parse_cli_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Assistant de recherche fiqh.")
    parser.add_argument(
        "--question",
        required=True,
        help="Question utilisateur à traiter.",
    )
    parser.add_argument(
        "--output",
        required=True,
        help="Chemin du fichier de sortie (.html recommandé).",
    )
    parser.add_argument(
        "--traduction",
        dest="traduction",
        action="store_true",
        help="Afficher la traduction française des pages.",
    )
    parser.add_argument(
        "--diagnostic-coherence",
        dest="diagnostic_coherence",
        action="store_true",
        help="Afficher le diagnostic du contrôle de cohérence (verdict/issues/retry).",
    )
    parser.add_argument(
        "--traduction-question-arabe",
        dest="traduction_question_arabe",
        action="store_true",
        help="Traduire automatiquement la question en arabe pour la pipeline interne.",
    )
    parser.set_defaults(traduction=TRANSLATE_TOP_CHUNKS_TO_FRENCH)
    parser.set_defaults(traduction_question_arabe=AUTO_TRANSLATE_QUESTION_TO_ARABIC)
    return parser.parse_args()


def main() -> None:
    args = parse_cli_args()
    start_time = time.time()
    print_startup_message()
    final_report = build_final_report(
        question=args.question,
        translate_to_french=args.traduction,
        diagnostic_coherence=args.diagnostic_coherence,
        auto_translate_question_to_arabic=args.traduction_question_arabe,
    )
    elapsed_seconds = time.time() - start_time
    write_output_with_timing(final_report, elapsed_seconds=elapsed_seconds, output_path=args.output)
