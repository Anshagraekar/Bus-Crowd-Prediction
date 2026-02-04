import cv2
import pytesseract
import re
from PIL import Image
import numpy as np
import os

pytesseract.pytesseract.tesseract_cmd = r"C:\Users\Asus\OneDrive\Documents\bus_crowd_project\Tesseract.exe"

# ======================================================
# 1️⃣ OCR FUNCTION (Image → Raw Text)
# ======================================================
def extract_text_from_image(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    thresh = cv2.threshold(gray, 150, 255, cv2.THRESH_BINARY)[1]
    text = pytesseract.image_to_string(thresh)
    return text


# ======================================================
# 2️⃣ PARSER FUNCTION (Raw Text → Structured Inputs)
# ======================================================
def parse_ticket_text(text):
    text = text.upper()

    # Route ID (e.g., 61 or 35)
    route_match = re.search(r"\n(\d{2})\n", text)
    route_id = route_match.group(1) if route_match else None

    # Boarding & Alighting stops
    route_line = re.search(r"([A-Z\s]+)\s-\s([A-Z\s]+)", text)
    boarding_stop = route_line.group(1).title().strip() if route_line else None
    alighting_stop = route_line.group(2).title().strip() if route_line else None

    # Time (09:55 AM)
    time_match = re.search(r"(\d{2}:\d{2}\s?(AM|PM))", text)
    time = time_match.group(1) if time_match else None

    return route_id, boarding_stop, alighting_stop, time