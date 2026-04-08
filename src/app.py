import argparse
import time

from src.config import TRANSLATE_TOP_CHUNKS_TO_FRENCH
from src.pipeline import build_final_report
from src.reporting import build_timing_message, print_startup_message, write_output_with_timing


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
    parser.set_defaults(traduction=TRANSLATE_TOP_CHUNKS_TO_FRENCH)
    return parser.parse_args()


def main() -> None:
    args = parse_cli_args()
    start_time = time.time()
    print_startup_message()
    final_report = build_final_report(
        question=args.question,
        translate_to_french=args.traduction,
        diagnostic_coherence=args.diagnostic_coherence,
    )
    elapsed_seconds = time.time() - start_time
    write_output_with_timing(final_report, elapsed_seconds=elapsed_seconds, output_path=args.output)
    print(final_report, flush=True)
    print(f"\n{build_timing_message(elapsed_seconds)}", flush=True)
