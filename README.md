# ZeroTouch KYC OCR Extractor

## Project Overview

ZeroTouch KYC OCR Extractor is an automated document extraction system designed for vendor onboarding workflows. It extracts structured data from Indian KYC documents using OCR (Optical Character Recognition) and rule-based logic.

**Supported Document Types:**
- **PAN Card** - Extracts PAN number, name, date of birth
- **GST Certificate** - Extracts GSTIN, legal name, address, registration type
- **Bank Documents** - Extracts account number, IFSC code, account holder name

**Key Capabilities:**
- ≥90% field-level extraction accuracy
- Confidence scores (0.0-1.0) for each extracted field
- Multiple interfaces: CLI and REST API (FastAPI)
- ERP integration layer for vendor master updates
- Built-in evaluation framework with precision/recall metrics

---

## Environment Setup

### Prerequisites
- Python 3.10 or higher
- pip (Python package manager)
- GPU with CUDA (optional, for faster processing)

### Step 1: Clone the Repository
```bash
git clone <repository-url>
cd ZeroTouch-KYC-OCR-Extractor
```

### Step 2: Create Virtual Environment
```bash
# Windows
python -m venv .venv
.venv\Scripts\activate

# Linux/Mac
python -m venv .venv
source .venv/bin/activate
```

### Step 3: Install Dependencies
```bash
pip install -r requirements.txt
```

---

## Libraries Used

| Library | Version | Purpose |
|---------|---------|---------|
| `easyocr` | 1.7+ | OCR engine for text extraction |
| `opencv-python` | 4.8+ | Image preprocessing and enhancement |
| `Pillow` | 10.0+ | Image loading and manipulation |
| `numpy` | 1.24+ | Numerical operations |
| `torch` | 2.0+ | Deep learning backend for EasyOCR |
| `fastapi` | 0.100+ | REST API framework |
| `uvicorn` | 0.23+ | ASGI server for FastAPI |
| `pydantic` | 2.0+ | Data validation and schemas |
| `spacy` | 3.6+ | Named Entity Recognition (NER) |
| `PyMuPDF` | 1.23+ | PDF processing |
| `python-multipart` | 0.0.6+ | File upload handling |
| `pytest` | 8.0+ | Unit testing framework |
| `faker` | 18.0+ | Synthetic data generation |

---

## Steps to Run the Code

### Option 1: Command Line Interface (CLI)

```bash
# Basic extraction (human-readable output)
python -m src.cli path/to/document.png

# JSON output for automation
python -m src.cli path/to/document.png --output json

# Enable GPU acceleration
python -m src.cli path/to/document.png --gpu

# View all options
python -m src.cli --help
```

### Option 2: REST API

```bash
# Start the API server
uvicorn src.api.main:app --reload

# Server runs at: http://localhost:8000
# API Documentation: http://localhost:8000/docs
```

**API Endpoints:**

| Method | Endpoint | Description |
|--------|----------|-------------|
| `POST` | `/extract` | Upload document and extract fields |
| `GET` | `/` | Health check |

**Example API Call:**
```bash
curl -X POST "http://localhost:8000/extract" \
  -F "file=@pan_card.png"
```

### Option 3: Run Evaluation Metrics

```bash
# Evaluate extraction accuracy against ground truth
python -m src.evaluation.metrics
```

### Option 4: Run Unit Tests

```bash
# Run all tests
python -m pytest tests/ -v
```

---

## Sample Data Explanation

The `data/` folder contains synthetic test documents and ground truth labels for evaluation.

### Folder Structure
```
data/
├── pan/                      # 20 synthetic PAN card images
│   ├── pan_01.png
│   ├── pan_02.png
│   └── ...
├── gst/                      # 20 synthetic GST certificate images
│   ├── gst_01.png
│   ├── gst_02.png
│   └── ...
├── bank/                     # 10 synthetic bank document images
│   ├── bank_01.png
│   ├── bank_02.png
│   └── ...
└── ground_truth/             # Expected extraction results (JSON)
    ├── pan_ground_truth.json
    ├── gst_ground_truth.json
    └── bank_ground_truth.json
```

### Sample Data Generation

Synthetic data is generated using `synthetic_data_generator.py`:

```bash
python scripts/synthetic_data_generator.py
```

This creates:
- Synthetic document images with randomized names, IDs, and dates
- Corresponding ground truth JSON files for accuracy evaluation
- Mild noise and rotation to simulate real-world scanning conditions

### Ground Truth Format

**PAN Ground Truth Example (`pan_ground_truth.json`):**
```json
[
  {
    "file": "pan_01.png",
    "name": "RAJESH KUMAR",
    "pan": "ABCDE1234F",
    "dob": "15/08/1990"
  }
]
```

**GST Ground Truth Example (`gst_ground_truth.json`):**
```json
[
  {
    "file": "gst_01.png",
    "legal_name": "ACME CORPORATION PVT LTD",
    "gstin": "27ABCDE1234F1Z5",
    "address": "123 MAIN STREET, MUMBAI 400001",
    "registration_date": "01/04/2020",
    "registration_type": "Regular"
  }
]
```

**Bank Ground Truth Example (`bank_ground_truth.json`):**
```json
[
  {
    "file": "bank_01.png",
    "account_holder": "PRIYA SHARMA",
    "account_number": "123456789012",
    "ifsc": "HDFC0001234"
  }
]
```

---

## Project Structure

```
ZeroTouch-KYC-OCR-Extractor/
│
├── src/                          # Source code
│   ├── api/                      # REST API (FastAPI)
│   │   └── main.py
│   ├── core/                     # Core extraction logic
│   │   ├── extractor.py          # DocumentExtractor class
│   │   ├── schema.py             # Pydantic response schemas
│   │   └── vendor_model.py       # Vendor data model
│   ├── integration/              # ERP integration layer
│   │   ├── vendor_mapper.py      # OCR → Vendor mapper
│   │   └── erp/                  # ERP client implementations
│   ├── evaluation/               # Metrics framework
│   │   └── metrics.py
│   ├── utils/                    # Utilities
│   │   └── image_utils.py
│   └── cli.py                    # Command-line interface
│
├── tests/                        # Unit tests
│   ├── test_extractor.py
│   ├── test_vendor_mapper.py
│   └── test_mock_erp.py
│
├── data/                         # Sample data & ground truth
│
├── synth.py                      # Synthetic data generator
├── requirements.txt              # Python dependencies
├── .gitignore
└── README.md
```
