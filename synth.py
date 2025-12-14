import os
import json
import random
import string
import numpy as np
from PIL import Image, ImageDraw, ImageFont, ImageFilter
from faker import Faker

# ---------------- CONFIG ----------------
BASE_DIR = "data"
PAN_DIR = os.path.join(BASE_DIR, "pan")
GST_DIR = os.path.join(BASE_DIR, "gst")
BANK_DIR = os.path.join(BASE_DIR, "bank")
GT_DIR = os.path.join(BASE_DIR, "ground_truth")

FONT_PATH = "arial.ttf"  # change if missing
PAN_COUNT = 20
GST_COUNT = 20
BANK_COUNT = 10

os.makedirs(PAN_DIR, exist_ok=True)
os.makedirs(GST_DIR, exist_ok=True)
os.makedirs(BANK_DIR, exist_ok=True)
os.makedirs(GT_DIR, exist_ok=True)

fake = Faker("en_IN")

# ---------------- HELPERS ----------------
def add_mild_noise(img):
    """Add very mild noise that won't corrupt OCR."""
    # Minimal rotation (0-0.5 degree)
    angle = random.uniform(-0.5, 0.5)
    img = img.rotate(angle, expand=True, fillcolor="white")
    
    # Very mild noise
    arr = np.array(img)
    noise = np.random.normal(0, 3, arr.shape)  # Reduced from 5 to 3
    noisy = arr + noise
    noisy = np.clip(noisy, 0, 255).astype(np.uint8)
    
    # No blur
    return Image.fromarray(noisy)

def random_pan():
    """Generate a valid PAN format: 5 letters + 4 digits + 1 letter."""
    return (
        ''.join(random.choices(string.ascii_uppercase, k=5)) +
        ''.join(random.choices(string.digits, k=4)) +
        random.choice(string.ascii_uppercase)
    )

def random_gstin():
    """Generate a valid GSTIN format: 2 digits + 5 letters + 4 digits + 1 letter + 1 alphanum + Z + 1 alphanum = 15 chars."""
    state_code = str(random.randint(10, 35))  # 2 digits
    pan_letters = ''.join(random.choices(string.ascii_uppercase, k=5))  # 5 letters
    pan_digits = ''.join(random.choices(string.digits, k=4))  # 4 digits
    pan_check = random.choice(string.ascii_uppercase)  # 1 letter
    # Avoid ambiguous chars (0, O, 1, I, 5, S, 8, B) in entity_code for cleaner OCR
    safe_chars = "234679ACDEFGHJKLMNPQRTUVWXYZ"
    entity_code = random.choice(safe_chars)  # 1 alphanum
    # Z is fixed
    checksum = random.choice(string.ascii_uppercase)  # 1 letter
    return f"{state_code}{pan_letters}{pan_digits}{pan_check}{entity_code}Z{checksum}"

def random_ifsc(bank_code):
    """Generate valid IFSC: 4 letters + 0 + 6 alphanumeric."""
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
print("🔄 Generating PAN cards...")
pan_gt = []

for i in range(1, PAN_COUNT + 1):
    img = Image.new("RGB", (600, 400), "white")
    draw = ImageDraw.Draw(img)

    title_font = ImageFont.truetype(FONT_PATH, 26)
    text_font = ImageFont.truetype(FONT_PATH, 20)

    name = fake.name().upper()
    pan = random_pan()
    dob = fake.date_of_birth(minimum_age=18, maximum_age=60).strftime("%d/%m/%Y")

    # Header
    draw.text((150, 30), "INCOME TAX DEPARTMENT", fill="black", font=title_font)
    draw.text((200, 70), "GOVT OF INDIA", fill="black", font=text_font)
    
    # Fields with clear labels
    draw.text((50, 150), f"Name: {name}", fill="black", font=text_font)
    draw.text((50, 200), f"PAN: {pan}", fill="black", font=text_font)
    draw.text((50, 250), f"Date of Birth: {dob}", fill="black", font=text_font)

    img = add_mild_noise(img)

    fname = f"pan_{i:02d}.png"
    img.save(os.path.join(PAN_DIR, fname))

    pan_gt.append({
        "file": fname,
        "name": name,
        "pan": pan,
        "dob": dob
    })

# ---------------- GST GENERATION ----------------
print("🔄 Generating GST certificates...")
gst_gt = []

# Common GST registration types
GST_REG_TYPES = ["Regular", "Composition", "Casual Taxable Person", "SEZ Unit", "SEZ Developer", "Input Service Distributor"]

for i in range(1, GST_COUNT + 1):
    img = Image.new("RGB", (800, 600), "white")
    draw = ImageDraw.Draw(img)

    title_font = ImageFont.truetype(FONT_PATH, 28)
    text_font = ImageFont.truetype(FONT_PATH, 20)

    legal_name = fake.company().upper()
    gstin = random_gstin()
    address = fake.address().replace("\n", ", ").upper()
    reg_date = fake.date_between(start_date="-5y", end_date="today").strftime("%d/%m/%Y")
    reg_type = random.choice(GST_REG_TYPES)

    # Header
    draw.text((180, 40), "GST REGISTRATION CERTIFICATE", fill="black", font=title_font)
    
    # Fields with clear colon-separated format
    draw.text((50, 140), f"Legal Name: {legal_name}", fill="black", font=text_font)
    draw.text((50, 190), f"GSTIN: {gstin}", fill="black", font=text_font)
    draw.text((50, 240), f"Address: {address}", fill="black", font=text_font)
    draw.text((50, 340), f"Registration Date: {reg_date}", fill="black", font=text_font)
    draw.text((50, 390), f"Registration Type: {reg_type}", fill="black", font=text_font)

    img = add_mild_noise(img)

    fname = f"gst_{i:02d}.png"
    img.save(os.path.join(GST_DIR, fname))

    gst_gt.append({
        "file": fname,
        "legal_name": legal_name,
        "gstin": gstin,
        "address": address,
        "registration_date": reg_date,
        "registration_type": reg_type
    })


# ---------------- BANK GENERATION ----------------
print("🔄 Generating Bank documents...")
bank_gt = []

bank_list = list(BANK_CONFIG.keys())

for i in range(1, BANK_COUNT + 1):
    img = Image.new("RGB", (900, 500), "white")
    draw = ImageDraw.Draw(img)

    title_font = ImageFont.truetype(FONT_PATH, 28)
    text_font = ImageFont.truetype(FONT_PATH, 22)
    
    bank_name = bank_list[i % len(bank_list)]  # Cycle through banks for diversity
    bank_code = BANK_CONFIG[bank_name]
    
    name = fake.name().upper()
    acc_no = random_account()
    ifsc = random_ifsc(bank_code)
    branch = fake.city().upper()
    doc_type = "BANK CERTIFICATE" if i % 2 == 0 else "CANCELLED CHEQUE"

    # Header - Bank name centered
    draw.text((300, 30), bank_name, fill="black", font=title_font)
    draw.text((350, 70), doc_type, fill="black", font=text_font)
    
    # Fields with CLEAR colon-separated format on same line
    y_start = 150
    line_height = 50
    
    draw.text((60, y_start), f"Account Holder Name: {name}", fill="black", font=text_font)
    draw.text((60, y_start + line_height), f"Account Number: {acc_no}", fill="black", font=text_font)
    draw.text((60, y_start + 2*line_height), f"IFSC Code: {ifsc}", fill="black", font=text_font)
    draw.text((60, y_start + 3*line_height), f"Branch: {branch}", fill="black", font=text_font)

    img = add_mild_noise(img)

    fname = f"bank_{i:02d}.png"
    img.save(os.path.join(BANK_DIR, fname))

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

print("✅ PAN, GST, and Bank images generated with ground truth JSON")
print(f"   📁 PAN: {PAN_COUNT} files in {PAN_DIR}")
print(f"   📁 GST: {GST_COUNT} files in {GST_DIR}")
print(f"   📁 Bank: {BANK_COUNT} files in {BANK_DIR}")
