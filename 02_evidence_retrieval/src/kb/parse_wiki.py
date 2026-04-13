import json
import os
import sys
import zipfile
from dataclasses import dataclass, asdict

# Add shared utils to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "..", "shared", "utils"))
from cleaning import remove_special_characters


@dataclass
class SentenceRecord:
    sentence_id: str       # "{article_title}_{line_number}" e.g. "Barack_Obama_3"
    article_title: str
    line_number: int
    text: str


def parse_wiki_pages(zip_path, min_length=10, verbose=True):
    """
    Extract and clean all sentences from FEVER's wiki-pages.zip.

    The zip contains ~109 JSONL files (wiki-001.jsonl through wiki-109.jsonl).
    Each JSON line has:
        - id: article title (e.g. "Barack_Obama")
        - text: raw article text
        - lines: newline-separated entries formatted as "{line_num}\\t{sentence_text}"

    Returns a list of SentenceRecord objects.
    """
    records = []
    articles_processed = 0

    with zipfile.ZipFile(zip_path, "r") as zf:
        jsonl_files = sorted(
            [n for n in zf.namelist() if n.endswith(".jsonl")]
        )

        if verbose:
            print(f"Found {len(jsonl_files)} JSONL files in {zip_path}")

        for file_idx, jsonl_name in enumerate(jsonl_files):
            with zf.open(jsonl_name) as f:
                for raw_line in f:
                    raw_line = raw_line.strip()
                    if not raw_line:
                        continue

                    try:
                        page = json.loads(raw_line)
                    except json.JSONDecodeError:
                        continue
                    title = page.get("id", "")
                    lines_blob = page.get("lines", "")

                    if not title or not lines_blob:
                        continue

                    for entry in lines_blob.split("\n"):
                        entry = entry.strip()
                        if not entry or "\t" not in entry:
                            continue

                        parts = entry.split("\t", 1)
                        if len(parts) != 2:
                            continue

                        line_num_str, sentence_text = parts

                        try:
                            line_num = int(line_num_str)
                        except ValueError:
                            continue

                        sentence_text = sentence_text.strip()
                        if not sentence_text:
                            continue

                        cleaned = remove_special_characters(sentence_text)
                        if len(cleaned) < min_length:
                            continue

                        sid = f"{title}_{line_num}"
                        records.append(SentenceRecord(
                            sentence_id=sid,
                            article_title=title,
                            line_number=line_num,
                            text=cleaned,
                        ))

                    articles_processed += 1

            if verbose:
                print(
                    f"  [{file_idx + 1}/{len(jsonl_files)}] {jsonl_name} — "
                    f"{articles_processed:,} articles, {len(records):,} sentences so far"
                )

    if verbose:
        print(
            f"\nParsing complete: {articles_processed:,} articles, "
            f"{len(records):,} sentences"
        )

    return records


def save_records(records, output_path):
    """Persist sentence records to a JSONL file."""
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        for rec in records:
            f.write(json.dumps(asdict(rec), ensure_ascii=False) + "\n")
    print(f"Saved {len(records):,} records to {output_path}")


def load_records(input_path):
    """Load sentence records from a JSONL file."""
    records = []
    with open(input_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                d = json.loads(line)
                records.append(SentenceRecord(**d))
    print(f"Loaded {len(records):,} records from {input_path}")
    return records


if __name__ == "__main__":
    import yaml

    config_path = os.path.join(os.path.dirname(__file__), "..", "config.yaml")
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    zip_path = config["corpus"]["wiki_zip_path"]
    records_path = config["corpus"]["records_path"]
    min_len = config["corpus"].get("min_sentence_length", 10)

    # Resolve relative paths from config.yaml's directory
    base_dir = os.path.dirname(os.path.abspath(config_path))
    zip_path = os.path.normpath(os.path.join(base_dir, zip_path))
    records_path = os.path.normpath(os.path.join(base_dir, records_path))

    print(f"Parsing wiki pages from: {zip_path}")
    records = parse_wiki_pages(zip_path, min_length=min_len)
    save_records(records, records_path)
