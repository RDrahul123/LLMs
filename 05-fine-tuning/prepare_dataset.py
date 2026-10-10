"""
Dataset preparation tool for OpenAI fine-tuning.

Loads data from JSON, CSV, or plain text, converts to JSONL format,
validates, splits into train/validation, and performs quality checks.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import random
from pathlib import Path
from typing import Any, Optional


def load_json_data(file_path: str) -> list[dict[str, Any]]:
    """Load data from a JSON file.

    Supports both array format [{...}] and object format {"data": [...]}.

    Args:
        file_path: Path to the JSON file.

    Returns:
        List of data records.
    """
    with open(file_path, encoding="utf-8") as f:
        data = json.load(f)

    if isinstance(data, list):
        return data
    if isinstance(data, dict):
        for key in ("data", "examples", "samples", "records"):
            if key in data and isinstance(data[key], list):
                return data[key]
        raise ValueError(f"Cannot find data array in JSON. Keys: {list(data.keys())}")

    raise ValueError(f"Unexpected JSON root type: {type(data)}")


def load_csv_data(file_path: str) -> list[dict[str, Any]]:
    """Load data from a CSV file.

    Args:
        file_path: Path to the CSV file.

    Returns:
        List of data records as dictionaries.
    """
    records = []
    with open(file_path, encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            records.append(dict(row))
    return records


def load_text_data(file_path: str, delimiter: str = "\n\n") -> list[dict[str, Any]]:
    """Load data from a plain text file.

    Each paragraph (separated by delimiter) becomes a record with
    a 'text' field.

    Args:
        file_path: Path to the text file.
        delimiter: Separator between records.

    Returns:
        List of data records.
    """
    with open(file_path, encoding="utf-8") as f:
        content = f.read()

    paragraphs = [p.strip() for p in content.split(delimiter) if p.strip()]
    return [{"text": p} for p in paragraphs]


def detect_format(file_path: str) -> str:
    """Detect the file format from extension.

    Args:
        file_path: Path to the file.

    Returns:
        Format string: 'json', 'csv', or 'text'.
    """
    ext = Path(file_path).suffix.lower()
    format_map = {
        ".json": "json",
        ".jsonl": "json",
        ".csv": "csv",
        ".tsv": "csv",
        ".txt": "text",
        ".md": "text",
    }
    return format_map.get(ext, "json")


def load_data(file_path: str, fmt: Optional[str] = None) -> list[dict[str, Any]]:
    """Load data from a file, auto-detecting format.

    Args:
        file_path: Path to the data file.
        fmt: Explicit format ('json', 'csv', 'text'). Auto-detected if None.

    Returns:
        List of data records.
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File not found: {file_path}")

    if fmt is None:
        fmt = detect_format(file_path)

    loaders = {
        "json": load_json_data,
        "csv": load_csv_data,
        "text": load_text_data,
    }

    loader = loaders.get(fmt)
    if loader is None:
        raise ValueError(f"Unsupported format: {fmt}. Use json, csv, or text.")

    return loader(file_path)


def convert_to_openai_jsonl(
    records: list[dict[str, Any]],
    output_path: str,
    system_prompt: str = "",
    user_field: str = "prompt",
    assistant_field: str = "completion",
) -> int:
    """Convert records to OpenAI JSONL fine-tuning format.

    Args:
        records: List of data records.
        output_path: Output JSONL file path.
        system_prompt: Optional system prompt to include.
        user_field: Field name for the user message.
        assistant_field: Field name for the assistant message.

    Returns:
        Number of records written.
    """
    count = 0
    with open(output_path, "w", encoding="utf-8") as f:
        for record in records:
            messages = []
            if system_prompt:
                messages.append({"role": "system", "content": system_prompt})

            user_content = record.get(user_field, record.get("text", ""))
            assistant_content = record.get(assistant_field, record.get("output", ""))

            if not user_content:
                continue

            messages.append({"role": "user", "content": str(user_content)})
            if assistant_content:
                messages.append({"role": "assistant", "content": str(assistant_content)})

            jsonl_record = {"messages": messages}
            f.write(json.dumps(jsonl_record) + "\n")
            count += 1

    return count


def validate_jsonl(file_path: str) -> dict[str, Any]:
    """Validate a JSONL file for OpenAI fine-tuning format.

    Args:
        file_path: Path to the JSONL file.

    Returns:
        Validation report dictionary.
    """
    report = {
        "valid": True,
        "total_lines": 0,
        "valid_lines": 0,
        "errors": [],
        "warnings": [],
    }

    with open(file_path, encoding="utf-8") as f:
        for line_num, line in enumerate(f, 1):
            report["total_lines"] += 1
            line = line.strip()
            if not line:
                continue

            try:
                record = json.loads(line)
            except json.JSONDecodeError as e:
                report["errors"].append(f"Line {line_num}: Invalid JSON - {e}")
                report["valid"] = False
                continue

            if "messages" not in record:
                report["errors"].append(f"Line {line_num}: Missing 'messages' key")
                report["valid"] = False
                continue

            messages = record["messages"]
            if not isinstance(messages, list) or len(messages) == 0:
                report["errors"].append(f"Line {line_num}: 'messages' must be a non-empty list")
                report["valid"] = False
                continue

            valid_roles = {"system", "user", "assistant"}
            has_user = False
            has_assistant = False

            for msg in messages:
                if not isinstance(msg, dict):
                    report["errors"].append(f"Line {line_num}: Message must be a dict")
                    report["valid"] = False
                    continue

                role = msg.get("role")
                if role not in valid_roles:
                    report["errors"].append(f"Line {line_num}: Invalid role '{role}'")
                    report["valid"] = False
                    continue

                if role == "user":
                    has_user = True
                elif role == "assistant":
                    has_assistant = True

                content = msg.get("content", "")
                if not content or not isinstance(content, str):
                    report["warnings"].append(f"Line {line_num}: Empty or non-string content for role '{role}'")

            if not has_user:
                report["errors"].append(f"Line {line_num}: No user message found")
                report["valid"] = False
            if not has_assistant:
                report["warnings"].append(f"Line {line_num}: No assistant message found")

            report["valid_lines"] += 1

    return report


def split_dataset(
    records: list[dict[str, Any]],
    train_ratio: float = 0.8,
    seed: int = 42,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Split dataset into train and validation sets.

    Args:
        records: List of data records.
        train_ratio: Fraction of data for training.
        seed: Random seed for reproducibility.

    Returns:
        Tuple of (train_records, validation_records).
    """
    random.seed(seed)
    shuffled = records.copy()
    random.shuffle(shuffled)

    split_idx = int(len(shuffled) * train_ratio)
    return shuffled[:split_idx], shuffled[split_idx:]


def compute_statistics(records: list[dict[str, Any]]) -> dict[str, Any]:
    """Compute basic dataset statistics.

    Args:
        records: List of data records.

    Returns:
        Statistics dictionary.
    """
    stats: dict[str, Any] = {
        "total_records": len(records),
        "fields": {},
    }

    if not records:
        return stats

    # Field analysis
    all_fields: dict[str, int] = {}
    for record in records:
        for key in record.keys():
            all_fields[key] = all_fields.get(key, 0) + 1

    stats["fields"] = all_fields

    # Text length analysis
    text_lengths = []
    for record in records:
        for value in record.values():
            if isinstance(value, str):
                text_lengths.append(len(value))

    if text_lengths:
        stats["text_length"] = {
            "min": min(text_lengths),
            "max": max(text_lengths),
            "mean": sum(text_lengths) / len(text_lengths),
            "median": sorted(text_lengths)[len(text_lengths) // 2],
        }

    return stats


def quality_checks(records: list[dict[str, Any]]) -> dict[str, Any]:
    """Perform quality checks on the dataset.

    Args:
        records: List of data records.

    Returns:
        Quality report dictionary.
    """
    report: dict[str, Any] = {
        "total_records": len(records),
        "issues": [],
        "duplicates": 0,
        "empty_records": 0,
        "format_issues": [],
    }

    # Check for duplicates
    seen = set()
    for i, record in enumerate(records):
        record_str = json.dumps(record, sort_keys=True)
        if record_str in seen:
            report["duplicates"] += 1
            report["issues"].append(f"Record {i}: Duplicate found")
        seen.add(record_str)

    # Check for empty records
    for i, record in enumerate(records):
        values = [v for v in record.values() if isinstance(v, str)]
        if not values or all(not v.strip() for v in values):
            report["empty_records"] += 1
            report["issues"].append(f"Record {i}: Empty or whitespace-only content")

    # Check format consistency
    if records:
        first_keys = set(records[0].keys())
        for i, record in enumerate(records):
            if set(record.keys()) != first_keys:
                report["format_issues"].append(
                    f"Record {i}: Inconsistent fields (expected {first_keys}, got {set(record.keys())})"
                )

    report["clean"] = (
        report["duplicates"] == 0
        and report["empty_records"] == 0
        and len(report["format_issues"]) == 0
    )

    return report


def run_pipeline(
    input_file: str,
    output_dir: str,
    system_prompt: str = "",
    user_field: str = "prompt",
    assistant_field: str = "completion",
    train_ratio: float = 0.8,
    seed: int = 42,
) -> None:
    """Run the complete dataset preparation pipeline.

    Args:
        input_file: Path to input data file.
        output_dir: Directory for output files.
        system_prompt: Optional system prompt for all examples.
        user_field: Field name for user messages.
        assistant_field: Field name for assistant messages.
        train_ratio: Train/validation split ratio.
        seed: Random seed.
    """
    print("=" * 60)
    print("Dataset Preparation Pipeline")
    print("=" * 60)

    # Load data
    print(f"\nLoading data from: {input_file}")
    records = load_data(input_file)
    print(f"Loaded {len(records)} records")

    # Compute statistics
    stats = compute_statistics(records)
    print(f"\nDataset Statistics:")
    print(f"  Total records: {stats['total_records']}")
    print(f"  Fields: {stats['fields']}")
    if "text_length" in stats:
        tl = stats["text_length"]
        print(f"  Text lengths - min: {tl['min']}, max: {tl['max']}, mean: {tl['mean']:.1f}")

    # Quality checks
    quality = quality_checks(records)
    print(f"\nQuality Report:")
    print(f"  Clean: {quality['clean']}")
    print(f"  Duplicates: {quality['duplicates']}")
    print(f"  Empty records: {quality['empty_records']}")
    if quality["issues"]:
        print(f"  Issues found: {len(quality['issues'])}")
        for issue in quality["issues"][:5]:
            print(f"    - {issue}")

    # Create output directory
    os.makedirs(output_dir, exist_ok=True)

    # Split dataset
    train_records, val_records = split_dataset(records, train_ratio, seed)
    print(f"\nSplit: {len(train_records)} train, {len(val_records)} validation")

    # Convert to JSONL
    base_name = Path(input_file).stem
    train_path = os.path.join(output_dir, f"{base_name}_train.jsonl")
    val_path = os.path.join(output_dir, f"{base_name}_val.jsonl")

    train_count = convert_to_openai_jsonl(
        train_records, train_path, system_prompt, user_field, assistant_field
    )
    val_count = convert_to_openai_jsonl(
        val_records, val_path, system_prompt, user_field, assistant_field
    )

    print(f"\nGenerated JSONL files:")
    print(f"  Training:   {train_path} ({train_count} records)")
    print(f"  Validation: {val_path} ({val_count} records)")

    # Validate output
    print("\nValidating training file...")
    train_report = validate_jsonl(train_path)
    print(f"  Valid: {train_report['valid']}, Lines: {train_report['valid_lines']}/{train_report['total_lines']}")

    print("Validating validation file...")
    val_report = validate_jsonl(val_path)
    print(f"  Valid: {val_report['valid']}, Lines: {val_report['valid_lines']}/{val_report['total_lines']}")

    if not train_report["valid"] or not val_report["valid"]:
        print("\nWarnings/Errors:")
        for error in train_report["errors"][:5]:
            print(f"  Train: {error}")
        for error in val_report["errors"][:5]:
            print(f"  Val: {error}")

    print("\n" + "=" * 60)
    print("Pipeline complete!")
    print("=" * 60)


def main() -> None:
    """CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Prepare datasets for OpenAI fine-tuning"
    )
    parser.add_argument(
        "input_file",
        help="Input data file (JSON, CSV, or text)",
    )
    parser.add_argument(
        "--output-dir", "-o",
        default="./prepared_data",
        help="Output directory (default: ./prepared_data)",
    )
    parser.add_argument(
        "--system-prompt", "-s",
        default="",
        help="System prompt for all examples",
    )
    parser.add_argument(
        "--user-field",
        default="prompt",
        help="Field name for user messages (default: prompt)",
    )
    parser.add_argument(
        "--assistant-field",
        default="completion",
        help="Field name for assistant messages (default: completion)",
    )
    parser.add_argument(
        "--train-ratio",
        type=float,
        default=0.8,
        help="Train/validation split ratio (default: 0.8)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed (default: 42)",
    )
    parser.add_argument(
        "--validate-only",
        action="store_true",
        help="Only validate an existing JSONL file",
    )

    args = parser.parse_args()

    if args.validate_only:
        report = validate_jsonl(args.input_file)
        print(json.dumps(report, indent=2))
        sys.exit(0 if report["valid"] else 1)

    run_pipeline(
        input_file=args.input_file,
        output_dir=args.output_dir,
        system_prompt=args.system_prompt,
        user_field=args.user_field,
        assistant_field=args.assistant_field,
        train_ratio=args.train_ratio,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()
