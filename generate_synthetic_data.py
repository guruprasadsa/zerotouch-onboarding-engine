import os
from PIL import Image, ImageDraw

output_dir = "data/raw_images"
os.makedirs(output_dir, exist_ok=True)


def generate_gst_image(index: int) -> None:
    """Create a synthetic GST certificate-like image."""
    img = Image.new("RGB", (800, 600), color="white")
    draw = ImageDraw.Draw(img)

    draw.text((300, 50), "GOVERNMENT OF INDIA", fill="black")
    draw.text((280, 80), "FORM GST REG-06", fill="black")
    draw.text((50, 150), f"Registration Number (GSTIN): 29ABCDE{index + 1000}F1Z5", fill="black")
    draw.text((50, 200), f"Legal Name: BUSINESS_TEST_{index}", fill="black")
    draw.text((50, 250), f"Trade Name: TRADER_{index}", fill="black")
    draw.text((50, 300), "Date of Liability: 01/07/2017", fill="black")

    img.save(f"{output_dir}/gst_{index:02d}.jpg")


def generate_bank_image(index: int) -> None:
    """Create a synthetic bank cheque-like image."""
    img = Image.new("RGB", (800, 400), color="white")
    draw = ImageDraw.Draw(img)

    draw.text((50, 50), "STATE BANK OF INDIA", fill="blue")
    draw.text((600, 50), f"Date: 12/12/202{index}", fill="black")
    draw.text((50, 100), "Pay to _______________ or Bearer", fill="black")
    draw.text((50, 150), "Rupees ________________________________", fill="black")
    draw.text((50, 250), f"A/c No: 3000{index}123456", fill="black")
    draw.text((50, 280), f"IFS Code: SBIN000{index}78", fill="black")
    draw.text((300, 300), "CANCELLED", fill="red")

    img.save(f"{output_dir}/bank_{index:02d}.jpg")


if __name__ == "__main__":
    for i in range(1, 16):
        generate_gst_image(i)
        generate_bank_image(i)

    print(f"Generated 15 GST and 15 Bank images in '{output_dir}'")
