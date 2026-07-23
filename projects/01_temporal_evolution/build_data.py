from __future__ import annotations

import argparse

from pipeline.corpus import build_corpus


def main() -> None:
    parser = argparse.ArgumentParser(description="Build the compact PubMed semantic corpus for the artwork.")
    parser.add_argument("--papers-per-year", type=int, default=104)
    parser.add_argument("--targeted-per-form", type=int, default=350)
    parser.add_argument(
        "--targeted-ai",
        type=int,
        default=1_600,
        help="Dedicated neuroscience and AI sample size; kept separate because this literature is much larger.",
    )
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--refresh-search", action="store_true", help="Refresh PubMed ID searches; unchanged embeddings are still cached.")
    args = parser.parse_args()
    build_corpus(
        papers_per_year=args.papers_per_year,
        targeted_per_form=args.targeted_per_form,
        targeted_ai=args.targeted_ai,
        force=args.force,
        refresh_search=args.refresh_search,
    )


if __name__ == "__main__":
    main()
