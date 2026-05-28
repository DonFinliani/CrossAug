import argparse
import json
import re
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parent
DEFAULT_INPUT_JSONL = Path("/home/GDM/LiteraryQA_data/jsonl/test.jsonl")
DEFAULT_OUTPUT_DIR = REPO_ROOT / "reproduce" / "dataset" / "literaryqa"
WORD_RE = re.compile(r"\S+")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Convert LiteraryQA JSONL into per-book HippoRAG datasets. Each input "
            "line becomes one independent HippoRAG dataset so each book can be "
            "indexed and evaluated with its own graph."
        )
    )
    parser.add_argument(
        "--input_jsonl",
        type=Path,
        default=DEFAULT_INPUT_JSONL,
        help="LiteraryQA JSONL file. Defaults to /home/GDM/LiteraryQA_data/jsonl/test.jsonl.",
    )
    parser.add_argument(
        "--output_dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Directory for HippoRAG dataset JSON files. Defaults to HippoRAG-main/reproduce/dataset/literaryqa.",
    )
    parser.add_argument(
        "--dataset_prefix",
        type=str,
        default="literaryqa",
        help="Prefix for generated HippoRAG dataset names.",
    )
    parser.add_argument(
        "--chunk_words",
        type=int,
        default=500,
        help="Target number of whitespace-delimited words per text chunk.",
    )
    parser.add_argument(
        "--chunk_overlap_words",
        type=int,
        default=80,
        help="Number of overlapping words between adjacent chunks.",
    )
    parser.add_argument(
        "--min_final_chunk_words",
        type=int,
        default=80,
        help="If the final chunk is shorter than this, merge it into the previous chunk.",
    )
    parser.add_argument(
        "--gold_docs",
        choices=["empty", "all_chunks"],
        default="empty",
        help=(
            "How to fill each QA sample's paragraphs field. Use 'empty' because "
            "LiteraryQA does not provide supporting chunk labels. Use 'all_chunks' "
            "only if a downstream script requires non-empty gold_docs; it makes "
            "dataset files much larger and recall metrics less meaningful."
        ),
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Optional maximum number of books to convert.",
    )
    parser.add_argument(
        "--start_index",
        type=int,
        default=0,
        help="Skip books before this zero-based input line index.",
    )
    parser.add_argument(
        "--include_unanswerable",
        action="store_true",
        help="Keep QA rows with no reference answers. By default those rows are skipped.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing generated JSON files.",
    )
    parser.add_argument(
        "--pretty",
        action="store_true",
        help="Write indented JSON. The default compact JSON is smaller for book chunks.",
    )
    return parser.parse_args()


def validate_args(args: argparse.Namespace) -> None:
    if args.chunk_words <= 0:
        raise ValueError("--chunk_words must be positive.")
    if args.chunk_overlap_words < 0:
        raise ValueError("--chunk_overlap_words must be non-negative.")
    if args.chunk_overlap_words >= args.chunk_words:
        raise ValueError("--chunk_overlap_words must be smaller than --chunk_words.")
    if args.min_final_chunk_words < 0:
        raise ValueError("--min_final_chunk_words must be non-negative.")
    if args.limit is not None and args.limit < 0:
        raise ValueError("--limit must be non-negative.")
    if args.start_index < 0:
        raise ValueError("--start_index must be non-negative.")
    if not args.input_jsonl.exists():
        raise FileNotFoundError(f"Input JSONL not found: {args.input_jsonl}")


def slugify(value: Any, fallback: str) -> str:
    text = "" if value is None else str(value)
    text = re.sub(r"[^A-Za-z0-9]+", "_", text).strip("_").lower()
    return text or fallback


def clean_title(title: Any, fallback: str) -> str:
    title_text = "" if title is None else str(title)
    title_text = " ".join(title_text.split())
    return title_text or fallback


def unique_preserve_order(values: list[Any]) -> list[str]:
    seen = set()
    output = []
    for value in values:
        if value is None:
            continue
        text = str(value).strip()
        if not text or text in seen:
            continue
        seen.add(text)
        output.append(text)
    return output


def split_text_into_word_chunks(
    text: str,
    chunk_words: int,
    overlap_words: int,
    min_final_chunk_words: int,
) -> list[dict[str, Any]]:
    matches = list(WORD_RE.finditer(text))
    if not matches:
        return []

    chunks: list[dict[str, Any]] = []
    step = chunk_words - overlap_words
    word_start = 0
    while word_start < len(matches):
        word_end = min(word_start + chunk_words, len(matches))
        char_start = matches[word_start].start()
        char_end = matches[word_end - 1].end()
        chunk_text = text[char_start:char_end].strip()

        chunks.append(
            {
                "text": chunk_text,
                "word_start": word_start,
                "word_end": word_end,
                "char_start": char_start,
                "char_end": char_end,
            }
        )

        if word_end == len(matches):
            break
        word_start += step

    if len(chunks) >= 2:
        final_word_count = chunks[-1]["word_end"] - chunks[-1]["word_start"]
        if final_word_count < min_final_chunk_words:
            previous = chunks[-2]
            final = chunks[-1]
            previous["text"] = text[previous["char_start"] : final["char_end"]].strip()
            previous["word_end"] = final["word_end"]
            previous["char_end"] = final["char_end"]
            chunks.pop()

    return chunks


def build_corpus(record: dict[str, Any], dataset_name: str, chunks: list[dict[str, Any]]) -> list[dict[str, Any]]:
    book_title = clean_title(record.get("title"), fallback=dataset_name)
    corpus = []
    num_chunks = len(chunks)

    for chunk_index, chunk in enumerate(chunks):
        chunk_title = f"{book_title} [chunk {chunk_index + 1:04d}]"
        corpus.append(
            {
                "title": chunk_title,
                "text": chunk["text"],
                "document_id": record.get("document_id"),
                "gutenberg_id": record.get("gutenberg_id"),
                "split": record.get("split"),
                "book_title": book_title,
                "chunk_id": f"{dataset_name}__chunk_{chunk_index:04d}",
                "chunk_index": chunk_index,
                "num_chunks": num_chunks,
                "word_start": chunk["word_start"],
                "word_end": chunk["word_end"],
                "char_start": chunk["char_start"],
                "char_end": chunk["char_end"],
            }
        )

    return corpus


def build_paragraphs(corpus: list[dict[str, Any]], gold_docs_mode: str) -> list[dict[str, Any]]:
    if gold_docs_mode == "empty":
        return []

    return [
        {
            "idx": item["chunk_index"],
            "title": item["title"],
            "paragraph_text": item["text"],
            "is_supporting": True,
        }
        for item in corpus
    ]


def build_samples(
    record: dict[str, Any],
    dataset_name: str,
    corpus: list[dict[str, Any]],
    gold_docs_mode: str,
    include_unanswerable: bool,
) -> list[dict[str, Any]]:
    book_title = clean_title(record.get("title"), fallback=dataset_name)
    document_id = str(record.get("document_id") or dataset_name)
    paragraphs = build_paragraphs(corpus, gold_docs_mode)
    qas = record.get("qas") or []
    metadata = record.get("metadata") or {}
    samples = []

    for qa_index, qa in enumerate(qas):
        question = str(qa.get("question") or "").strip()
        answers = unique_preserve_order(qa.get("answers") or [])

        if not question:
            continue
        if not answers and not include_unanswerable:
            continue

        samples.append(
            {
                "id": f"{document_id}__q{qa_index:03d}",
                "document_id": record.get("document_id"),
                "gutenberg_id": record.get("gutenberg_id"),
                "split": record.get("split"),
                "title": book_title,
                "question": question,
                "answer": answers[0] if answers else "",
                "answer_aliases": answers[1:],
                "answerable": bool(answers),
                "paragraphs": paragraphs,
                "metadata": {
                    "qa_index": qa_index,
                    "is_question_modified": qa.get("is_question_modified"),
                    "is_answer_modified": qa.get("is_answer_modified"),
                    "author": metadata.get("author"),
                    "publication_date": metadata.get("publication_date"),
                    "genre_tags": metadata.get("genre_tags"),
                    "text_url": metadata.get("text_url") or metadata.get("text_urls"),
                    "summary_url": metadata.get("summary_url") or metadata.get("summary_urls"),
                },
            }
        )

    return samples


def write_json(path: Path, payload: Any, pretty: bool, overwrite: bool) -> None:
    if path.exists() and not overwrite:
        raise FileExistsError(f"Refusing to overwrite existing file without --overwrite: {path}")

    with path.open("w", encoding="utf-8") as f:
        if pretty:
            json.dump(payload, f, ensure_ascii=False, indent=2)
        else:
            json.dump(payload, f, ensure_ascii=False, separators=(",", ":"))


def write_lines(path: Path, lines: list[str], overwrite: bool) -> None:
    if path.exists() and not overwrite:
        raise FileExistsError(f"Refusing to overwrite existing file without --overwrite: {path}")

    with path.open("w", encoding="utf-8") as f:
        for line in lines:
            f.write(line + "\n")


def manifest_path_string(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(REPO_ROOT))
    except ValueError:
        return str(path.resolve())


def make_dataset_name(prefix: str, input_index: int, record: dict[str, Any], used_names: set[str]) -> str:
    split = slugify(record.get("split"), fallback="split")
    gutenberg_id = slugify(record.get("gutenberg_id"), fallback="")
    document_id = slugify(str(record.get("document_id") or "")[:12], fallback=f"doc_{input_index:04d}")
    prefix_slug = slugify(prefix, fallback="literaryqa")

    base_parts = [prefix_slug]
    if split and prefix_slug != split and not prefix_slug.endswith(f"_{split}"):
        base_parts.append(split)
    base_parts.append(f"{input_index:04d}")
    base_parts.append(gutenberg_id or document_id)
    base_name = "_".join(part for part in base_parts if part)

    dataset_name = base_name
    suffix = 2
    while dataset_name in used_names:
        dataset_name = f"{base_name}_{suffix}"
        suffix += 1
    used_names.add(dataset_name)
    return dataset_name


def convert(args: argparse.Namespace) -> list[dict[str, Any]]:
    args.output_dir.mkdir(parents=True, exist_ok=True)
    books_dir = args.output_dir / "books"
    books_dir.mkdir(parents=True, exist_ok=True)
    manifest = []
    used_dataset_names: set[str] = set()
    converted = 0

    with args.input_jsonl.open("r", encoding="utf-8") as f:
        for input_index, line in enumerate(f):
            if input_index < args.start_index:
                continue
            if args.limit is not None and converted >= args.limit:
                break
            if not line.strip():
                continue

            record = json.loads(line)
            text = record.get("text") or ""
            if not isinstance(text, str) or not text.strip():
                print(f"Skipping line {input_index}: empty text")
                continue

            dataset_name = make_dataset_name(args.dataset_prefix, input_index, record, used_dataset_names)
            chunks = split_text_into_word_chunks(
                text=text,
                chunk_words=args.chunk_words,
                overlap_words=args.chunk_overlap_words,
                min_final_chunk_words=args.min_final_chunk_words,
            )
            if not chunks:
                print(f"Skipping line {input_index}: no chunks produced")
                continue

            corpus = build_corpus(record, dataset_name, chunks)
            samples = build_samples(
                record=record,
                dataset_name=dataset_name,
                corpus=corpus,
                gold_docs_mode=args.gold_docs,
                include_unanswerable=args.include_unanswerable,
            )
            if not samples:
                print(f"Skipping line {input_index}: no QA samples produced")
                continue

            corpus_path = books_dir / f"{dataset_name}_corpus.json"
            samples_path = books_dir / f"{dataset_name}.json"
            write_json(corpus_path, corpus, pretty=args.pretty, overwrite=args.overwrite)
            write_json(samples_path, samples, pretty=args.pretty, overwrite=args.overwrite)

            manifest.append(
                {
                    "dataset": dataset_name,
                    "document_id": record.get("document_id"),
                    "gutenberg_id": record.get("gutenberg_id"),
                    "split": record.get("split"),
                    "title": record.get("title"),
                    "num_chunks": len(corpus),
                    "num_qas": len(samples),
                    "text_chars": len(text),
                    "text_words": len(WORD_RE.findall(text)),
                    "corpus_path": manifest_path_string(corpus_path),
                    "samples_path": manifest_path_string(samples_path),
                    "book_output_subdir": dataset_name,
                }
            )
            converted += 1

    manifest_payload = {
        "format": "hipporag_literaryqa_multibook_v1",
        "dataset": slugify(args.dataset_prefix, fallback="literaryqa"),
        "source_jsonl": str(args.input_jsonl),
        "gold_docs": args.gold_docs,
        "chunk_words": args.chunk_words,
        "chunk_overlap_words": args.chunk_overlap_words,
        "min_final_chunk_words": args.min_final_chunk_words,
        "num_books": len(manifest),
        "num_total_chunks": sum(item["num_chunks"] for item in manifest),
        "num_total_qas": sum(item["num_qas"] for item in manifest),
        "books": manifest,
        "main_command": f"python main.py --dataset {slugify(args.dataset_prefix, fallback='literaryqa')}",
    }

    manifest_path = args.output_dir / "manifest.json"
    commands_main_path = args.output_dir / "run_main_command.txt"

    write_json(manifest_path, manifest_payload, pretty=True, overwrite=args.overwrite)
    write_lines(
        commands_main_path,
        [manifest_payload["main_command"]],
        overwrite=args.overwrite,
    )

    return manifest


def main() -> None:
    args = parse_args()
    validate_args(args)
    manifest = convert(args)
    print(f"Converted {len(manifest)} LiteraryQA books into per-book HippoRAG datasets.")
    print(f"Output directory: {args.output_dir}")
    if manifest:
        print("First dataset:", manifest[0]["dataset"])
        print("Example command from HippoRAG root:")
        print(" ", f"python main.py --dataset {slugify(args.dataset_prefix, fallback='literaryqa')}")


if __name__ == "__main__":
    main()
