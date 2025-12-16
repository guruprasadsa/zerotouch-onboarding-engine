"""
Synthetic Data Generator for Document OCR Extractor.
Generates synthetic PAN cards, GST certificates, and Bank documents
for training and evaluation purposes.

Uses OpenCV for image generation and text rendering.
"""

import os
import json
import random
import string
import cv2
import numpy as np
from faker import Faker

# ---------------- CONFIG ----------------
BASE_DIR = "data"
PAN_DIR = os.path.join(BASE_DIR, "pan")
GST_DIR = os.path.join(BASE_DIR, "gst")
BANK_DIR = os.path.join(BASE_DIR, "bank")
GT_DIR = os.path.join(BASE_DIR, "ground_truth")

PAN_COUNT = 20
GST_COUNT = 20
BANK_COUNT = 10

os.makedirs(PAN_DIR, exist_ok=True)
os.makedirs(GST_DIR, exist_ok=True)
os.makedirs(BANK_DIR, exist_ok=True)
os.makedirs(GT_DIR, exist_ok=True)

# Fixed seed for reproducible synthetic data
random.seed(42)
np.random.seed(42)

fake = Faker("en_IN")
fake.seed_instance(42)

# OpenCV font settings
FONT = cv2.FONT_HERSHEY_SIMPLEX
FONT_BOLD = cv2.FONT_HERSHEY_DUPLEX

# Colors (BGR format for OpenCV)
BLACK = (0, 0, 0)
WHITE = (255, 255, 255)
GRAY = (128, 128, 128)
DARK_BLUE = (139, 0, 0)


# ---------------- HELPERS ----------------
def put_text(img, text, position, font_scale=0.6, color=BLACK, thickness=1, font=FONT):
    """Draw text on image using OpenCV with anti-aliasing."""
    cv2.putText(img, text, position, font, font_scale, color, thickness, cv2.LINE_AA)


def add_mild_noise(img):
    """Add very mild noise that won't corrupt OCR."""
    # Minimal rotation (0-0.5 degree)
    angle = random.uniform(-0.5, 0.5)
    h, w = img.shape[:2]
    center = (w // 2, h // 2)
    rotation_matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
    img = cv2.warpAffine(img, rotation_matrix, (w, h), borderValue=WHITE)
    
    # Very mild Gaussian noise
    noise = np.random.normal(0, 3, img.shape).astype(np.float32)
    noisy = img.astype(np.float32) + noise
    noisy = np.clip(noisy, 0, 255).astype(np.uint8)
    
    return noisy


def random_pan(entity_type='P', name_initial=None):
    """
    Generate a valid PAN format following official structure:
    - Positions 1-3: Random alphabetic series (AAA-ZZZ)
    - Position 4: Entity type (P/C/H/A/T/F/G/J/L/B)
    - Position 5: First letter of surname
    - Positions 6-9: Sequential number (0001-9999)
    - Position 10: Alphabetic check digit
    """
    series = ''.join(random.choices(string.ascii_uppercase, k=3))
    
    valid_types = 'PCHATFGJLB'
    if entity_type not in valid_types:
        entity_type = 'P'
    
    if name_initial is None:
        name_initial = random.choice(string.ascii_uppercase)
    else:
        name_initial = name_initial[0].upper()
    
    seq_num = f"{random.randint(1, 9999):04d}"
    check_digit = random.choice(string.ascii_uppercase)
    
    return f"{series}{entity_type}{name_initial}{seq_num}{check_digit}"


def extract_surname(full_name):
    """Extract surname (last name) from full name."""
    parts = full_name.strip().split()
    if len(parts) >= 2:
        return parts[-1]
    return parts[0] if parts else "UNKNOWN"


def random_gstin():
    """
    Generate a valid GSTIN format with correct checksum:
    - Positions 1-2: State code (01-37)
    - Positions 3-7: PAN letters (5 uppercase)
    - Positions 8-11: PAN digits (4 digits)
    - Position 12: PAN check letter
    - Position 13: Entity number (1-9, A-Z)
    - Position 14: Default 'Z'
    - Position 15: Checksum (calculated)
    """
    state_code = f"{random.randint(1, 37):02d}"
    pan_letters = ''.join(random.choices(string.ascii_uppercase, k=5))
    pan_digits = ''.join(random.choices(string.digits, k=4))
    pan_check = random.choice(string.ascii_uppercase)
    entity_code = random.choice(string.digits[1:] + string.ascii_uppercase)
    
    # Build first 14 chars
    gstin_14 = f"{state_code}{pan_letters}{pan_digits}{pan_check}{entity_code}Z"
    
    # Calculate checksum (position 15)
    checksum = _calculate_gstin_checksum(gstin_14)
    
    return gstin_14 + checksum


def _calculate_gstin_checksum(gstin_14: str) -> str:
    """Calculate GSTIN checksum for first 14 characters."""
    char_values = "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ"
    
    total = 0
    for i, char in enumerate(gstin_14):
        value = char_values.index(char.upper())
        factor = 1 if i % 2 == 0 else 2
        product = value * factor
        total += (product // 36) + (product % 36)
    
    checksum_value = (36 - (total % 36)) % 36
    return char_values[checksum_value]


def random_ifsc(bank_code):
    """Generate valid IFSC: 4 letters + 0 + 6 digits."""
    return bank_code + "0" + ''.join(random.choices(string.digits, k=6))


def random_account():
    """Generate 12-digit account number."""
    return ''.join(random.choices(string.digits, k=12))


# ---------------- BANK CODES ----------------
BANK_CONFIG = {
    "STATE BANK OF INDIA": "SBIN",
    "HDFC BANK": "HDFC",
    "ICICI BANK": "ICIC",
    "AXIS BANK": "UTIB",
    "CANARA BANK": "CNRB",
    "PUNJAB NATIONAL BANK": "PUNB",
    "BANK OF BARODA": "BARB",
    "KOTAK MAHINDRA BANK": "KKBK",
    "INDUSIND BANK": "INDB",
    "YES BANK": "YESB"
}


# ---------------- PAN GENERATION ----------------
print(" Generating PAN cards...")
pan_gt = []

for i in range(1, PAN_COUNT + 1):
    # Create white background - standard PAN card size ratio (85.6mm x 53.98mm)
    img = np.ones((350, 550, 3), dtype=np.uint8) * 255
    
    # Generate name first to extract surname initial
    full_name = fake.name().upper()
    surname = extract_surname(full_name)
    name_initial = surname[0] if surname else 'X'
    
    # Generate PAN with correct structure
    pan = random_pan(entity_type='P', name_initial=name_initial)
    dob = fake.date_of_birth(minimum_age=18, maximum_age=60).strftime("%d/%m/%Y")
    father_name = fake.name_male().upper()

    # === OFFICIAL PAN CARD LAYOUT (2018+) ===
    
    # Top Header
    put_text(img, "INCOME TAX DEPARTMENT", (140, 28), 0.6, BLACK, 1, FONT_BOLD)
    put_text(img, "GOVT. OF INDIA", (210, 50), 0.45, BLACK, 1)
    
    # Photo placeholder (left side) - gray box
    cv2.rectangle(img, (30, 70), (120, 170), GRAY, 1)
    put_text(img, "PHOTO", (52, 125), 0.4, GRAY, 1)
    
    # PAN Number - prominently displayed (right of photo, top area)
    put_text(img, "Permanent Account Number", (140, 80), 0.4, BLACK, 1)
    put_text(img, pan, (140, 110), 0.8, BLACK, 2, FONT_BOLD)
    
    # Name (below photo area)
    put_text(img, "Name", (30, 195), 0.4, GRAY, 1)
    put_text(img, full_name, (30, 215), 0.55, BLACK, 1)
    
    # Father's Name
    put_text(img, "Father's Name", (30, 245), 0.4, GRAY, 1)
    put_text(img, father_name, (30, 265), 0.55, BLACK, 1)
    
    # Date of Birth (bottom left)
    put_text(img, "Date of Birth", (30, 295), 0.4, GRAY, 1)
    put_text(img, dob, (30, 315), 0.55, BLACK, 1)
    
    # Signature placeholder (bottom right)
    cv2.rectangle(img, (350, 270), (510, 320), GRAY, 1)
    put_text(img, "Signature", (395, 300), 0.35, GRAY, 1)

    img = add_mild_noise(img)

    fname = f"pan_{i:02d}.png"
    cv2.imwrite(os.path.join(PAN_DIR, fname), img)

    pan_gt.append({
        "file": fname,
        "name": full_name,
        "pan": pan,
        "dob": dob,
        "father_name": father_name
    })


# ---------------- GST GENERATION ----------------
print(" Generating GST certificates...")
gst_gt = []

GST_REG_TYPES = ["Regular", "Composition", "Casual Taxable Person", 
                 "SEZ Unit", "SEZ Developer", "Input Service Distributor"]

for i in range(1, GST_COUNT + 1):
    # Create white background
    img = np.ones((600, 800, 3), dtype=np.uint8) * 255

    legal_name = fake.company().upper()
    gstin = random_gstin()
    address = fake.address().replace("\n", ", ").upper()
    reg_date = fake.date_between(start_date="-5y", end_date="today").strftime("%d/%m/%Y")
    reg_type = random.choice(GST_REG_TYPES)

    # Header
    put_text(img, "GST REGISTRATION CERTIFICATE", (180, 50), 0.8, BLACK, 2, FONT_BOLD)
    
    # Fields - use thickness=1 for cleaner OCR
    put_text(img, f"Legal Name: {legal_name}", (50, 150), 0.6, BLACK, 1)
    put_text(img, f"GSTIN: {gstin}", (50, 200), 0.6, BLACK,1)
    put_text(img, f"Address: {address[:60]}", (50, 250), 0.5, BLACK, 1)
    if len(address) > 60:
        put_text(img, address[60:120], (50, 285), 0.5, BLACK, 1)
    put_text(img, f"Registration Date: {reg_date}", (50, 340), 0.6, BLACK, 1)
    put_text(img, f"Registration Type: {reg_type}", (50, 390), 0.6, BLACK, 1)

    img = add_mild_noise(img)

    fname = f"gst_{i:02d}.png"
    cv2.imwrite(os.path.join(GST_DIR, fname), img)

    gst_gt.append({
        "file": fname,
        "legal_name": legal_name,
        "gstin": gstin,
        "address": address,
        "registration_date": reg_date,
        "registration_type": reg_type
    })


# ---------------- BANK GENERATION ----------------
print("Generating Bank documents...")
bank_gt = []

bank_list = list(BANK_CONFIG.keys())

for i in range(1, BANK_COUNT + 1):
    # Create white background
    img = np.ones((500, 900, 3), dtype=np.uint8) * 255
    
    bank_name = bank_list[i % len(bank_list)]
    bank_code = BANK_CONFIG[bank_name]
    
    name = fake.name().upper()
    acc_no = random_account()
    ifsc = random_ifsc(bank_code)
    branch = fake.city().upper()
    doc_type = "BANK CERTIFICATE" if i % 2 == 0 else "CANCELLED CHEQUE"

    # Header
    put_text(img, bank_name, (300, 40), 0.8, BLACK, 2, FONT_BOLD)
    put_text(img, doc_type, (350, 80), 0.6, BLACK, 1)
    
    # Fields
    y_start = 160
    line_height = 50
    
    put_text(img, f"Account Holder Name: {name}", (60, y_start), 0.6, BLACK, 1)
    put_text(img, f"Account Number: {acc_no}", (60, y_start + line_height), 0.6, BLACK, 1)
    put_text(img, f"IFSC Code: {ifsc}", (60, y_start + 2*line_height), 0.6, BLACK, 1)
    put_text(img, f"Branch: {branch}", (60, y_start + 3*line_height), 0.6, BLACK, 1)

    img = add_mild_noise(img)

    fname = f"bank_{i:02d}.png"
    cv2.imwrite(os.path.join(BANK_DIR, fname), img)

    bank_gt.append({
        "file": fname,
        "document_type": doc_type,
        "bank_name": bank_name,
        "account_holder": name,
        "account_number": acc_no,
        "ifsc": ifsc,
        "branch": branch
    })


# ---------------- SAVE GROUND TRUTH ----------------
with open(os.path.join(GT_DIR, "pan_ground_truth.json"), "w") as f:
    json.dump(pan_gt, f, indent=2)

with open(os.path.join(GT_DIR, "gst_ground_truth.json"), "w") as f:
    json.dump(gst_gt, f, indent=2)

with open(os.path.join(GT_DIR, "bank_ground_truth.json"), "w") as f:
    json.dump(bank_gt, f, indent=2)

print(" PAN, GST, and Bank images generated with ground truth JSON")
print(f"    PAN: {PAN_COUNT} files in {PAN_DIR}")
print(f"    GST: {GST_COUNT} files in {GST_DIR}")
print(f"    Bank: {BANK_COUNT} files in {BANK_DIR}")
