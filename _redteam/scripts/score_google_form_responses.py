#!/usr/bin/env python3
import argparse
import csv
import re
from pathlib import Path
from typing import Dict, Tuple

import matplotlib.pyplot as plt
import matplotlib as mpl


# ---------------------------------------------------------
# Global Plot Style (NeurIPS/ICLR-ready)
# ---------------------------------------------------------
mpl.rcParams.update({
    "figure.dpi": 160,
    "font.size": 15,
    "font.family": "serif",
    "axes.labelsize": "large",
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": True,
    "grid.alpha": 0.3,
    "lines.linewidth": 2,
    "legend.frameon": False,
    'xtick.labelsize': 10
})


ID_PATTERN = re.compile(r"ID:(E\d+)\s*\|")


def parse_truth(value: str) -> bool:
    val = str(value).strip().lower()
    if val in {"true", "1", "yes"}:
        return True
    if val in {"false", "0", "no"}:
        return False
    raise ValueError(f"Unrecognized truth value: {value!r}")


def parse_answer(value: str):
    val = str(value).strip().lower()
    if val == "yes":
        return True
    if val == "no":
        return False
    return None


def load_ground_truth(path: Path) -> Dict[str, bool]:
    truth: Dict[str, bool] = {}
    with path.open("r", encoding="utf-8", newline="") as f:
        for row in csv.DictReader(f):
            truth[row["entry_id"]] = parse_truth(row["jailbreak"])
    return truth


def score_response_row(response_row: Dict[str, str], truth: Dict[str, bool]) -> Tuple[int, int, int, int, int, int, int]:
    """
    Score a single response row against ground truth.
    
    Returns:
        (total, answered, correct, tp, fp, tn, fn)
        - total: number of questions detected
        - answered: number of non-empty answers provided
        - correct: number of answers matching ground truth
        - tp: true positives (predicted True, actually True)
        - fp: false positives (predicted True, actually False)
        - tn: true negatives (predicted False, actually False)
        - fn: false negatives (predicted False, actually True)
    """
    total = 0
    answered = 0
    correct = 0
    tp = 0
    fp = 0
    tn = 0
    fn = 0

    for header, value in response_row.items():
        m = ID_PATTERN.search(header or "")
        if not m:
            continue
        entry_id = m.group(1)
        if entry_id not in truth:
            continue

        total += 1
        parsed = parse_answer(value)
        ground_truth = truth[entry_id]
        
        if parsed is None:
            # Unanswered treated as negative for confusion matrix
            if ground_truth:
                fn += 1  # Should have said True, said nothing
            else:
                tn += 1  # Correctly didn't answer for False case
            continue

        answered += 1
        if parsed == ground_truth:
            correct += 1
        
        # Update confusion matrix
        if parsed and ground_truth:
            tp += 1  # Correctly predicted True
        elif parsed and not ground_truth:
            fp += 1  # Incorrectly predicted True
        elif not parsed and not ground_truth:
            tn += 1  # Correctly predicted False
        else:  # not parsed and ground_truth
            fn += 1  # Incorrectly predicted False

    return total, answered, correct, tp, fp, tn, fn


def compute_majority_vote_truth(responses_csv: Path, valid_ids: set[str]) -> Dict[str, bool]:
    """Build per-entry human ground truth from majority vote across all response rows."""
    votes: Dict[str, Dict[str, int]] = {entry_id: {"yes": 0, "no": 0} for entry_id in valid_ids}

    with responses_csv.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            for header, value in row.items():
                m = ID_PATTERN.search(header or "")
                if not m:
                    continue
                entry_id = m.group(1)
                if entry_id not in votes:
                    continue
                parsed = parse_answer(value)
                if parsed is None:
                    continue
                if parsed:
                    votes[entry_id]["yes"] += 1
                else:
                    votes[entry_id]["no"] += 1

    majority_truth: Dict[str, bool] = {}
    ties = 0
    no_votes = 0
    for entry_id, c in votes.items():
        yes_count = c["yes"]
        no_count = c["no"]
        if yes_count == 0 and no_count == 0:
            no_votes += 1
            continue
        if yes_count == no_count:
            ties += 1
            continue
        majority_truth[entry_id] = yes_count > no_count

    print(f"Human-majority ground truth built for {len(majority_truth)} entries")
    if ties:
        print(f"Skipped {ties} tied entries")
    if no_votes:
        print(f"Skipped {no_votes} entries with no human votes")

    return majority_truth


def compute_confusion(predictions: Dict[str, bool], truth: Dict[str, bool]) -> Tuple[int, int, int, int, int]:
    """Return (n, tp, fp, tn, fn) for predictions against truth on intersection IDs."""
    tp = fp = tn = fn = 0
    n = 0
    for entry_id, gt in truth.items():
        if entry_id not in predictions:
            continue
        pred = predictions[entry_id]
        n += 1
        if pred and gt:
            tp += 1
        elif pred and not gt:
            fp += 1
        elif not pred and not gt:
            tn += 1
        else:
            fn += 1
    return n, tp, fp, tn, fn


def normalize_response_value(value: str) -> str:
    """Normalize response values to yes/no/empty for consistency."""
    val = str(value).strip().lower() if value else ""
    if val in {"yes", "y", "true", "1"}:
        return "Yes"
    elif val in {"no", "n", "false", "0"}:
        return "No"
    else:
        return ""


def convert_human_judgement_responses(input_csv: Path, output_csv: Path) -> None:
    """
    Convert human_judgement_responses.csv to desired response format.
    
    Validates header format, normalizes Yes/No values, removes empty rows,
    and produces a clean CSV suitable for score_google_form_responses.py.
    
    Args:
        input_csv: Path to human_judgement_responses.csv
        output_csv: Path to write converted CSV
    """
    if not input_csv.exists():
        raise FileNotFoundError(f"Input file not found: {input_csv}")
    
    rows_read = 0
    rows_written = 0
    invalid_headers = []
    
    with input_csv.open("r", encoding="utf-8", newline="") as infile, \
         output_csv.open("w", encoding="utf-8", newline="") as outfile:
        
        reader = csv.DictReader(infile)
        if not reader.fieldnames:
            raise ValueError("Input CSV has no headers")
        
        # Validate and prepare headers
        headers = list(reader.fieldnames)
        for header in headers:
            if header and header != "Timestamp":
                if not ID_PATTERN.search(header):
                    invalid_headers.append(header)
        
        if invalid_headers:
            print(f"WARNING: {len(invalid_headers)} headers don't match expected ID pattern:")
            for h in invalid_headers[:5]:
                print(f"  - {h}")
        
        writer = csv.DictWriter(outfile, fieldnames=headers)
        writer.writeheader()
        
        for row in reader:
            rows_read += 1
            
            # Normalize all response values
            normalized_row = {}
            has_data = False
            
            for key, value in row.items():
                if key == "Timestamp":
                    normalized_row[key] = value or ""
                else:
                    normalized_value = normalize_response_value(value)
                    normalized_row[key] = normalized_value
                    if normalized_value:
                        has_data = True
            
            # Skip entirely empty responses
            if not has_data and rows_read > 1:
                print(f"Skipping empty row {rows_read}")
                continue
            
            writer.writerow(normalized_row)
            rows_written += 1
    
    print(f"Conversion complete:")
    print(f"  Input rows: {rows_read}")
    print(f"  Output rows: {rows_written}")
    print(f"  Output file: {output_csv}")


def save_confusion_matrix_figure(tp: int, fp: int, tn: int, fn: int, output_path: Path) -> None:
    """Save a confusion matrix figure.

    Ground truth axis corresponds to majority-vote human labels.
    Prediction axis corresponds to judge-model labels.
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)
    print(output_path)

    # Rows: ground truth (human majority vote), Cols: prediction (judge model)
    matrix = [[tn, fp], [fn, tp]]

    # husl_cmap = sns.color_palette("husl", as_cmap=True)

    fig, ax = plt.subplots(figsize=(6.5, 5.5))
    im = ax.imshow(matrix)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    ax.set_xticks([0, 1])
    ax.set_yticks([0, 1])
    ax.set_xticklabels(["No", "Yes"])
    ax.set_yticklabels(["No", "Yes"])
    ax.set_xlabel("Prediction (Judge Model)", fontsize=14)
    ax.set_ylabel("Ground Truth (Human Majority Vote)", fontsize=14)
    ax.set_title("Confusion Matrix", fontsize=16)

    for i in range(2):
        for j in range(2):
            ax.text(j, i, str(matrix[i][j]), ha="center", va="center", color="black", fontsize=14)

    fig.tight_layout()
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Score Google Form responses against redteam jailbreak ground truth, or convert raw responses to desired format."
    )
    
    subparsers = parser.add_subparsers(dest="command", help="Command to run")
    
    # Conversion subcommand
    convert_parser = subparsers.add_parser("convert", help="Convert human_judgement_responses.csv to desired format")
    convert_parser.add_argument("--input", required=True, type=Path, help="Input CSV file (human_judgement_responses.csv)")
    convert_parser.add_argument("--output", required=True, type=Path, help="Output CSV file (converted responses)")
    
    # Scoring subcommand
    score_parser = subparsers.add_parser("score", help="Compare judge labels against majority-vote human ground truth")
    score_parser.add_argument("--responses-csv", required=True, type=Path, help="CSV with per-annotator human responses")
    score_parser.add_argument("--ground-truth-csv", required=True, type=Path)
    score_parser.add_argument(
        "--output-csv",
        type=Path,
        default=None,
        help="Optional output of per-entry majority-vote comparison summary",
    )
    score_parser.add_argument(
        "--confusion-matrix-fig",
        type=Path,
        default=None,
        help="Optional output path for confusion matrix figure (PNG)",
    )
    
    args = parser.parse_args()
    
    # Handle conversion command
    if args.command == "convert":
        convert_human_judgement_responses(args.input, args.output)
        return
    
    # Handle scoring command (or default if no command specified)
    if args.command == "score" or (not args.command and hasattr(args, 'responses_csv')):
        # Support legacy command-line style
        if not hasattr(args, 'responses_csv'):
            parser.print_help()
            return
        
        # Judge model outputs from google_form_ground_truth.csv
        judge_predictions = load_ground_truth(args.ground_truth_csv)
        # Human-majority labels from all annotator responses
        majority_truth = compute_majority_vote_truth(args.responses_csv, set(judge_predictions.keys()))

        # Restrict counts to evaluated overlap only (typically 200 entries).
        evaluated_ids = [entry_id for entry_id in majority_truth if entry_id in judge_predictions]
        judge_true_in_evaluated = sum(1 for entry_id in evaluated_ids if judge_predictions[entry_id])
        human_true_in_evaluated = sum(1 for entry_id in evaluated_ids if majority_truth[entry_id])

        evaluated_count, total_tp, total_fp, total_tn, total_fn = compute_confusion(
            predictions=judge_predictions,
            truth=majority_truth,
        )

        summaries = []
        for entry_id, human_gt in majority_truth.items():
            if entry_id not in judge_predictions:
                continue
            judge_pred = judge_predictions[entry_id]
            if judge_pred and human_gt:
                label = "TP"
            elif judge_pred and not human_gt:
                label = "FP"
            elif not judge_pred and not human_gt:
                label = "TN"
            else:
                label = "FN"
            summaries.append(
                {
                    "entry_id": entry_id,
                    "human_majority_ground_truth": str(human_gt),
                    "judge_prediction": str(judge_pred),
                    "confusion_cell": label,
                }
            )

        if args.output_csv:
            args.output_csv.parent.mkdir(parents=True, exist_ok=True)
            with args.output_csv.open("w", encoding="utf-8", newline="") as f:
                writer = csv.DictWriter(
                    f,
                    fieldnames=[
                        "entry_id",
                        "human_majority_ground_truth",
                        "judge_prediction",
                        "confusion_cell",
                    ],
                )
                writer.writeheader()
                writer.writerows(summaries)

        # Compute and display aggregate confusion matrix and metrics
        print("\n" + "="*70)
        print("CONFUSION MATRIX (Judge Model vs Human Majority Vote)")
        print("="*70)
        print(f"Evaluated entries:      {evaluated_count}")
        print(f"Judge true (evaluated): {judge_true_in_evaluated}")
        print(f"Human true (evaluated): {human_true_in_evaluated}")
        print(f"True Positives (TP):  {total_tp}")
        print(f"False Positives (FP): {total_fp}")
        print(f"True Negatives (TN):  {total_tn}")
        print(f"False Negatives (FN): {total_fn}")
        
        # Compute metrics
        total_predicted_true = total_tp + total_fp
        total_predicted_false = total_tn + total_fn
        total_actual_true = total_tp + total_fn
        total_actual_false = total_fp + total_tn
        
        precision = (total_tp / total_predicted_true) if total_predicted_true > 0 else 0.0
        recall = (total_tp / total_actual_true) if total_actual_true > 0 else 0.0
        specificity = (total_tn / total_actual_false) if total_actual_false > 0 else 0.0
        f1 = (2 * precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
        accuracy = (total_tp + total_tn) / (total_tp + total_fp + total_tn + total_fn) if (total_tp + total_fp + total_tn + total_fn) > 0 else 0.0
        
        print("\nMETRICS:")
        print(f"Precision (TP/(TP+FP)):     {precision:.6f}")
        print(f"Recall (TP/(TP+FN)):        {recall:.6f}")
        print(f"Specificity (TN/(TN+FP)):   {specificity:.6f}")
        print(f"F1-Score:                   {f1:.6f}")
        print(f"Overall Accuracy:           {accuracy:.6f}")
        print("="*70)

        if args.confusion_matrix_fig:
            save_confusion_matrix_figure(
                tp=total_tp,
                fp=total_fp,
                tn=total_tn,
                fn=total_fn,
                output_path=args.confusion_matrix_fig,
            )
            print(f"Confusion matrix figure saved to: {args.confusion_matrix_fig}")


if __name__ == "__main__":
    main()
