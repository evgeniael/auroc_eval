from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any, Dict, List


def load_items(path: Path) -> List[Dict[str, Any]]:
    with path.open("r") as f:
        return json.load(f)


def load_existing_annotations(path: Path) -> Dict[str, Any]:
    if not path.exists():
        return {}
    with path.open("r") as f:
        data = json.load(f)
    # Support both list-of-dicts (same structure as generations) or id->annotation dict
    if isinstance(data, list):
        annotations = {}
        for item in data:
            qid = item.get("question_id")
            if qid is not None:
                annotations[qid] = item
        return annotations
    elif isinstance(data, dict):
        return data
    else:
        raise ValueError("Unsupported annotation file format.")


def save_annotations(path: Path, annotations: Dict[str, Any]) -> None:
    # Store as dict keyed by question_id for efficient resume
    with path.open("w") as f:
        json.dump(annotations, f, indent=2, ensure_ascii=False)


def prompt_label(question: str, answers_aliases: List[str], prediction: str) -> Dict[str, Any]:
    print("\n" + "-" * 80)
    print(f"Question:\n  {question}\n")
    print("Plausible answers:")
    for i, ans in enumerate(answers_aliases):
        print(f"  [{i}] {ans}")
    print("\nModel answer:")
    print(f"  {prediction}\n")
    print("Label options:")
    print("  [y] correct")
    print("  [n] incorrect")
    print("  [u] unsure / ambiguous")
    print("  [s] skip (no label, move on)")

    while True:
        choice = input("Your label [y/n/u/s]: ").strip().lower()
        if choice in {"y", "n", "u", "s"}:
            break
        print("Please enter one of: y, n, u, s.")

    if choice == "s":
        return {}

    note = input("Optional comment (enter to skip): ").strip()

    return {
        "label": {"y": "correct", "n": "incorrect", "u": "unsure"}[choice],
        "comment": note,
    }


def annotate_file(input_path: Path, output_path: Path, start_index: int | None = None) -> None:
    items = load_items(input_path)
    annotations = load_existing_annotations(output_path)

    # Build index over items for deterministic ordering and resume
    total = len(items)
    print(f"Loaded {total} items from {input_path}")

    # If start_index is provided, use it; otherwise resume from existing annotations
    if start_index is not None:
        current_idx = max(0, min(start_index, total - 1))
    else:
        # Find first index whose question_id is not annotated yet
        current_idx = 0
        while current_idx < total:
            qid = items[current_idx].get("question_id")
            if qid is None or qid not in annotations:
                break
            current_idx += 1

    print(f"Starting from index {current_idx} (0-based).")

    for idx in range(current_idx, total):
        item = items[idx]
        qid = item.get("question_id", f"idx_{idx}")

        # Skip if already annotated
        if qid in annotations and isinstance(annotations[qid], dict) and "human_greedy_label" in annotations[qid]:
            continue

        question = item.get("question", "")
        answers_aliases = item.get("answers_aliases", [])
        greedy_pred = item.get("greedy_prediction", {}).get("greedy_string", "")

        label_info = prompt_label(question, answers_aliases, greedy_pred)

        if not label_info:
            # Skipped
            continue

        # Store annotation; keep minimal necessary info
        annotations[qid] = {
            "question_id": qid,
            "question": question,
            "answers_aliases": answers_aliases,
            "greedy_prediction": greedy_pred,
            "human_greedy_label": label_info["label"],
            "human_comment": label_info["comment"],
        }

        # Save after each annotation to avoid losing work
        save_annotations(output_path, annotations)
        print(f"Saved annotation {idx + 1}/{total} to {output_path}")

        # Allow user to stop early
        cont = input("Press Enter to continue, or type 'q' to quit: ").strip().lower()
        if cont == "q":
            print("Stopping early. Progress saved.")
            break

    print("Annotation session finished.")
    print(f"Annotated {len(annotations)} items so far. Output at: {output_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Interactive annotation tool for greedy predictions.")
    parser.add_argument(
        "--input",
        type=str,
        required=True,
        help="Path to generations JSON file (list of dicts).",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Path to output annotations JSON file. "
        "Defaults to replacing 'generations' with 'annotations' in the input path.",
    )
    parser.add_argument(
        "--start_index",
        type=int,
        default=None,
        help="Optional 0-based index to start from (overrides automatic resume).",
    )
    args = parser.parse_args()

    input_path = Path(args.input).expanduser().resolve()
    if args.output is not None:
        output_path = Path(args.output).expanduser().resolve()
    else:
        # Reasonable default: results/annotations/<dataset>_<model>_annotations.json
        parent = input_path.parent
        name = input_path.name
        if "generations" in str(parent):
            out_parent = parent.parent / "annotations"
        else:
            out_parent = parent
        out_parent.mkdir(parents=True, exist_ok=True)
        stem = name.rsplit(".", 1)[0]
        output_path = out_parent / f"{stem}_annotations.json"

    if not input_path.exists():
        raise FileNotFoundError(f"Input file not found: {input_path}")

    print(f"Input:  {input_path}")
    print(f"Output: {output_path}")

    annotate_file(input_path, output_path, start_index=args.start_index)


if __name__ == "__main__":
    main()

