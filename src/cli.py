#!/usr/bin/env python
"""
Document OCR Extractor - Command Line Interface

Usage:
    python -m src.cli extract <file_path> [options]
    python -m src.cli evaluate [options]

Examples:
    python -m src.cli extract document.png
    python -m src.cli extract document.pdf --output json --gpu
    python -m src.cli evaluate --doc-type pan --output json
    python -m src.cli evaluate --doc-type all --gpu
"""
import argparse
import json
import sys
import os
from typing import Dict, Any

import torch

from src.core.extractor import DocumentExtractor
from src.evaluation.metrics import (
    evaluate_doc_type,
    compute_metrics,
    DATA_DIR,
    GT_DIR,
    PAN_DIR,
    GST_DIR,
    BANK_DIR,
)

# Version should match the API version
__version__ = "1.0.0"


# Field mappings for evaluation
FIELD_MAPS: Dict[str, Dict[str, Dict]] = {
    "pan": {
        "pan": {"pred_key": "pan_number", "mode": "strict"},
        "name": {"pred_key": "name", "mode": "relaxed"},
        "dob": {"pred_key": "date_of_birth", "mode": "strict"},
    },
    "gst": {
        "gstin": {"pred_key": "gstin", "mode": "ocr"},
        "legal_name": {"pred_key": "legal_name", "mode": "relaxed"},
        "address": {"pred_key": "address", "mode": "relaxed"},
        "registration_date": {"pred_key": "registration_date", "mode": "strict"},
        "registration_type": {"pred_key": "registration_type", "mode": "relaxed"},
    },
    "bank": {
        "account_number": {"pred_key": "account_number", "mode": "strict"},
        "ifsc": {"pred_key": "ifsc_code", "mode": "strict"},
        "account_holder": {"pred_key": "payee_name", "mode": "relaxed"},
    },
}

DOC_TYPE_CONFIG = {
    "pan": {
        "image_dir": PAN_DIR,
        "gt_file": os.path.join(GT_DIR, "pan_ground_truth.json"),
        "field_map": FIELD_MAPS["pan"],
    },
    "gst": {
        "image_dir": GST_DIR,
        "gt_file": os.path.join(GT_DIR, "gst_ground_truth.json"),
        "field_map": FIELD_MAPS["gst"],
    },
    "bank": {
        "image_dir": BANK_DIR,
        "gt_file": os.path.join(GT_DIR, "bank_ground_truth.json"),
        "field_map": FIELD_MAPS["bank"],
    },
}


def create_parser() -> argparse.ArgumentParser:
    """Create the main argument parser with subcommands."""
    parser = argparse.ArgumentParser(
        prog="document-ocr-extractor",
        description="Document OCR Extractor - Extract and evaluate documents",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""\
Examples:
  python -m src.cli extract pan_card.png --output json
  python -m src.cli extract gst_cert.pdf --gpu
  python -m src.cli evaluate --doc-type pan
  python -m src.cli evaluate --doc-type all --output json
"""
    )
    
    parser.add_argument(
        "-v", "--version",
        action="version",
        version=f"%(prog)s {__version__}"
    )
    
    subparsers = parser.add_subparsers(dest="command", help="Available commands")
    
    # Extract subcommand
    extract_parser = subparsers.add_parser(
        "extract",
        help="Extract fields from a single document",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    extract_parser.add_argument(
        "file_path",
        help="Path to the image (PNG, JPG) or PDF file to process"
    )
    extract_parser.add_argument(
        "-o", "--output",
        choices=["text", "json"],
        default="text",
        help="Output format: 'text' for human-readable, 'json' for machine-readable (default: text)"
    )
    extract_parser.add_argument(
        "--gpu",
        action="store_true",
        help="Enable GPU acceleration if CUDA is available"
    )
    
    # Evaluate subcommand
    evaluate_parser = subparsers.add_parser(
        "evaluate",
        help="Evaluate extraction accuracy against ground truth data",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    evaluate_parser.add_argument(
        "-d", "--doc-type",
        choices=["pan", "gst", "bank", "all"],
        default="all",
        help="Document type to evaluate (default: all)"
    )
    evaluate_parser.add_argument(
        "-o", "--output",
        choices=["text", "json"],
        default="text",
        help="Output format: 'text' for human-readable, 'json' for machine-readable (default: text)"
    )
    evaluate_parser.add_argument(
        "--gpu",
        action="store_true",
        help="Enable GPU acceleration if CUDA is available"
    )
    
    return parser


def print_text_output(result: dict) -> None:
    """Format and print human-readable extraction results."""
    status = result.get("status", "unknown")
    doc_type = result.get("document_type", "UNKNOWN")
    
    print(f"\nDocument Type: {doc_type}")
    print(f"Status: {status}")
    
    if status == "success":
        fields = result.get("fields", {})
        if fields:
            print("\nExtracted Fields:")
            for key, field in fields.items():
                if isinstance(field, dict) and "value" in field:
                    value = field["value"]
                    confidence = field.get("confidence", 0.0)
                    print(f"  {key}: {value} (confidence: {confidence:.2f})")
                else:
                    print(f"  {key}: {field}")
        else:
            print("\nNo fields extracted.")
    else:
        message = result.get("message", "Unknown error")
        print(f"\nError: {message}", file=sys.stderr)


def print_metrics_text(metrics: Dict[str, Dict[str, Any]], doc_type: str) -> None:
    """Print metrics in human-readable format."""
    print(f"\n{doc_type.upper()} Metrics:")
    print("-" * 50)
    
    for field, values in metrics.items():
        accuracy = values.get("accuracy", 0.0)
        precision = values.get("precision", 0.0)
        recall = values.get("recall", 0.0)
        f1 = values.get("f1_score", 0.0)
        support = values.get("support", 0)
        
        # Color code accuracy (ANSI escape codes)
        if accuracy >= 0.9:
            acc_indicator = "✓"
        elif accuracy >= 0.7:
            acc_indicator = "~"
        else:
            acc_indicator = "✗"
        
        print(f"  {field}:")
        print(f"    {acc_indicator} Accuracy:  {accuracy:.1%}")
        print(f"      Precision: {precision:.1%}")
        print(f"      Recall:    {recall:.1%}")
        print(f"      F1 Score:  {f1:.1%}")
        print(f"      Support:   {support}")


def run_extract(args: argparse.Namespace) -> int:
    """Run the extract command."""
    # Validate file exists
    if not os.path.exists(args.file_path):
        print(f"Error: File not found: {args.file_path}", file=sys.stderr)
        return 1
    
    # Determine GPU usage
    use_gpu = args.gpu and torch.cuda.is_available()
    if args.gpu and not torch.cuda.is_available():
        print("Warning: --gpu specified but CUDA is not available. Using CPU.", file=sys.stderr)
    
    # Initialize extractor and process
    try:
        extractor = DocumentExtractor(use_gpu=use_gpu)
        result = extractor.extract_from_file(args.file_path)
    except Exception as e:
        print(f"Error: Extraction failed: {e}", file=sys.stderr)
        return 2
    
    # Output results
    if args.output == "json":
        print(json.dumps(result, indent=2, ensure_ascii=False))
    else:
        print_text_output(result)
    
    return 0 if result.get("status") == "success" else 1


def run_evaluate(args: argparse.Namespace) -> int:
    """Run the evaluate command."""
    # Determine GPU usage
    use_gpu = args.gpu and torch.cuda.is_available()
    if args.gpu and not torch.cuda.is_available():
        print("Warning: --gpu specified but CUDA is not available. Using CPU.", file=sys.stderr)
    
    print(f"\nInitializing Extractor (GPU={use_gpu})...")
    extractor = DocumentExtractor(use_gpu=use_gpu)
    
    # Determine which document types to evaluate
    if args.doc_type == "all":
        doc_types = ["pan", "gst", "bank"]
    else:
        doc_types = [args.doc_type]
    
    all_metrics: Dict[str, Dict[str, Any]] = {}
    
    for doc_type in doc_types:
        config = DOC_TYPE_CONFIG[doc_type]
        
        # Check if ground truth file exists
        if not os.path.exists(config["gt_file"]):
            print(f"Warning: Ground truth file not found: {config['gt_file']}", file=sys.stderr)
            continue
        
        # Check if image directory exists
        if not os.path.exists(config["image_dir"]):
            print(f"Warning: Image directory not found: {config['image_dir']}", file=sys.stderr)
            continue
        
        print(f"\nEvaluating {doc_type.upper()} documents...")
        
        try:
            stats = evaluate_doc_type(
                image_dir=config["image_dir"],
                gt_file=config["gt_file"],
                field_map=config["field_map"],
                extractor=extractor,
            )
            metrics = compute_metrics(stats)
            all_metrics[doc_type] = metrics
        except Exception as e:
            print(f"Error evaluating {doc_type}: {e}", file=sys.stderr)
            continue
    
    # Output results
    if args.output == "json":
        print(json.dumps(all_metrics, indent=2, ensure_ascii=False))
    else:
        print("\n" + "=" * 50)
        print("EVALUATION METRICS SUMMARY")
        print("=" * 50)
        
        for doc_type, metrics in all_metrics.items():
            print_metrics_text(metrics, doc_type)
        
        # Print overall summary
        if all_metrics:
            print("\n" + "=" * 50)
            print("OVERALL SUMMARY")
            print("=" * 50)
            
            total_fields = 0
            passing_fields = 0
            
            for doc_type, metrics in all_metrics.items():
                for field, values in metrics.items():
                    total_fields += 1
                    if values.get("accuracy", 0) >= 0.9:
                        passing_fields += 1
            
            print(f"\nFields meeting 90% accuracy target: {passing_fields}/{total_fields}")
            
            if total_fields > 0:
                overall_pass_rate = passing_fields / total_fields
                status = "PASSED" if overall_pass_rate >= 0.8 else "NEEDS IMPROVEMENT"
                print(f"Overall pass rate: {overall_pass_rate:.1%} - {status}")
    
    return 0


def main() -> int:
    """Main entry point. Returns exit code."""
    parser = create_parser()
    args = parser.parse_args()
    
    if args.command is None:
        parser.print_help()
        return 0
    
    if args.command == "extract":
        return run_extract(args)
    elif args.command == "evaluate":
        return run_evaluate(args)
    else:
        parser.print_help()
        return 1


if __name__ == "__main__":
    sys.exit(main())
