import cv2
import pytesseract
from pdf2image import convert_from_path
from PIL import Image
import os

pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"

def extract_text_from_image(image_path):
    """ Extract text from an image using OCR """
    image = cv2.imread(image_path)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    text = pytesseract.image_to_string(gray)
    return text

def extract_text_from_pdf(pdf_path):
    """ Convert PDF to images and extract text from each page """
    images = convert_from_path(pdf_path)
    text = "\n".join([pytesseract.image_to_string(img) for img in images])
    return text

def extract_text(file_path):
    """ Identify file type and extract text accordingly """
    if file_path.endswith((".png", ".jpg", ".jpeg")):
        return extract_text_from_image(file_path)
    elif file_path.endswith(".pdf"):
        return extract_text_from_pdf(file_path)
    else:
        raise ValueError("Unsupported file format")
