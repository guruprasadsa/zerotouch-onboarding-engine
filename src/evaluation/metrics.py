import json
import os
from typing import Dict
import torch

from src.core.extractor import DocumentExtractor


# Paths
DATA_DIR = "data"
GT_DIR = os.path.join(DATA_DIR, "ground_truth")

PAN_DIR = os.path.join(DATA_DIR, "pan")
GST_DIR = os.path.join(DATA_DIR, "gst")
BANK_DIR = os.path.join(DATA_DIR, "bank")


# Helpers
def load_json(path):
    with open(path, "r") as f:
        return json.load(f)


def normalize(text):
    if not text:
        return ""
    return text.strip().upper()


def ocr_normalize(text):
    """Normalize common OCR confusions for structured fields like GSTIN, IFSC, PAN."""
    if not text:
        return ""
    replacements = {
        "O": "0", "I": "1", "L": "1", "S": "5",
        "B": "8", "Z": "2", "G": "6"
    }
    result = text.strip().upper()
    return "".join(replacements.get(c, c) for c in result)


def relaxed_match(gt: str, pred: str, min_overlap: float = 0.7) -> bool:
    if not gt or not pred:
        return False

    gt_tokens = set(gt.split())
    pred_tokens = set(pred.split())

    if not gt_tokens or not pred_tokens:
        return False

    overlap = len(gt_tokens & pred_tokens) / len(gt_tokens)
    return overlap >= min_overlap


def ocr_match(gt: str, pred: str) -> bool:
    return ocr_normalize(gt) == ocr_normalize(pred)


# Evaluation Logic
def evaluate_doc_type(
    image_dir: str,
    gt_file: str,
    field_map: Dict[str, Dict],
    extractor: DocumentExtractor
):
    gt_data = load_json(gt_file)

    stats = {
        gt_key: {"TP": 0, "FP": 0, "FN": 0}
        for gt_key in field_map.keys()
    }

    for sample in gt_data:
        img_path = os.path.join(image_dir, sample["file"])
        result = extractor.extract_from_file(img_path)
        extracted_fields = result.get("fields", {})

        for gt_key, cfg in field_map.items():
            pred_key = cfg["pred_key"]
            mode = cfg.get("mode", "strict")

            gt_value = normalize(sample.get(gt_key))
            pred_value = normalize(
                extracted_fields.get(pred_key, {}).get("value")
            )

            if gt_value and pred_value:
                if mode == "strict":
                    match = gt_value == pred_value
                elif mode == "ocr":
                    match = ocr_match(gt_value, pred_value)
                else:
                    match = relaxed_match(gt_value, pred_value)

                if match:
                    stats[gt_key]["TP"] += 1
                else:
                    stats[gt_key]["FP"] += 1
                    stats[gt_key]["FN"] += 1

            elif gt_value and not pred_value:
                stats[gt_key]["FN"] += 1

            elif pred_value and not gt_value:
                stats[gt_key]["FP"] += 1

    return stats


def compute_metrics(stats):
    metrics = {}

    for field, counts in stats.items():
        tp = counts["TP"]
        fp = counts["FP"]
        fn = counts["FN"]

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        accuracy = tp / (tp + fp + fn) if (tp + fp + fn) > 0 else 0.0

        f1 = (
            2 * precision * recall / (precision + recall)
            if (precision + recall) > 0
            else 0.0
        )

        metrics[field] = {
            "accuracy": round(accuracy, 3),
            "precision": round(precision, 3),
            "recall": round(recall, 3),
            "f1_score": round(f1, 3),
            "support": tp + fn
        }

    return metrics


# Main
def main():
    use_gpu = torch.cuda.is_available()
    print(f"\nInitializing Extractor (GPU={use_gpu})")
    extractor = DocumentExtractor(use_gpu=use_gpu)

    # PAN
    print("\nEvaluating PAN Documents...")
    pan_stats = evaluate_doc_type(
        image_dir=PAN_DIR,
        gt_file=os.path.join(GT_DIR, "pan_ground_truth.json"),
        field_map={
            "pan": {"pred_key": "pan_number", "mode": "strict"},
            "name": {"pred_key": "name", "mode": "relaxed"},
            "dob": {"pred_key": "date_of_birth", "mode": "strict"},
        },
        extractor=extractor
    )
    pan_metrics = compute_metrics(pan_stats)

    # GST
    print("\nEvaluating GST Documents...")
    gst_stats = evaluate_doc_type(
        image_dir=GST_DIR,
        gt_file=os.path.join(GT_DIR, "gst_ground_truth.json"),
        field_map={
            "gstin": {"pred_key": "gstin", "mode": "ocr"},
            "legal_name": {"pred_key": "legal_name", "mode": "relaxed"},
            "address": {"pred_key": "address", "mode": "relaxed"},
            "registration_date": {"pred_key": "registration_date", "mode": "strict"},
            "registration_type": {"pred_key": "registration_type", "mode": "relaxed"},
        },
        extractor=extractor
    )
    gst_metrics = compute_metrics(gst_stats)

    # BANK
    print("\nEvaluating Bank Documents...")
    bank_stats = evaluate_doc_type(
        image_dir=BANK_DIR,
        gt_file=os.path.join(GT_DIR, "bank_ground_truth.json"),
        field_map={
            "account_number": {"pred_key": "account_number", "mode": "strict"},
            "ifsc": {"pred_key": "ifsc_code", "mode": "strict"},
            "account_holder": {"pred_key": "payee_name", "mode": "relaxed"},
        },
        extractor=extractor
    )
    bank_metrics = compute_metrics(bank_stats)

    # Print Metrics
    print("\nFINAL METRICS SUMMARY\n")

    print("PAN Metrics:")
    for k, v in pan_metrics.items():
        print(f"  {k}: {v}")

    print("\nGST Metrics:")
    for k, v in gst_metrics.items():
        print(f"  {k}: {v}")

    print("\nBank Metrics:")
    for k, v in bank_metrics.items():
        print(f"  {k}: {v}")


if __name__ == "__main__":
    main()
