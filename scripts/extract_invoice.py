import os
import json
import re
from groq import Groq
from scripts.ocr_extraction import extract_text  
from dotenv import load_dotenv

load_dotenv()
client = Groq(api_key=os.environ.get("GROQ_API_KEY"))

def clean_text(text):
    """Cleans and normalizes OCR-extracted text."""
    return ' '.join(text.replace("\n", " ").replace("\r", " ").split())

def extract_json_from_response(response_text):
    """Extracts JSON data from a mixed response using regex."""
    match = re.search(r'\{.*\}', response_text, re.DOTALL)
    if match:
        return match.group(0)  
    return None  

def extract_invoice_data(file_path, category):
    """Extracts relevant details based on the category using OCR and Groq AI."""
    
    raw_text = extract_text(file_path) 
    cleaned_text = clean_text(raw_text)  

    if category == "invoice":
        prompt = """
        Extract the following invoice details from the given text:
        - Invoice Number
        - Date
        - Total Amount
        - Vendor Name

        Return a **valid JSON object ONLY** with the structure:
        {
            "invoice_number": "<Invoice Number or null>",
            "date": "<Invoice Date or null>",
            "amount": "<Total Amount or null>",
            "vendor": "<Vendor Name or null>"
        }
        Ensure **NO extra text or explanations**, only JSON.
        """
    elif category == "email":
        prompt = """
        Extract the email address from the given text.

        Return a **valid JSON object ONLY** with the structure:
        {
            "email": "<Email Address or null>"
        }
        Ensure **NO extra text or explanations**, only JSON.
        """
    elif category == "budget":
        prompt = """
        Extract the budget amount from the given text.

        Return a **valid JSON object ONLY** with the structure:
        {
            "budget": "<Budget Amount or null>"
        }
        Ensure **NO extra text or explanations**, only JSON.
        """
    else:
        return {"error": "Invalid category"}

    prompt += f"\nHere is the extracted text:\n---\n{cleaned_text}\n---"

    try:
        response = client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[
                {"role": "system", "content": "You are an AI that extracts structured document details accurately."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.2
        )

        response_text = response.choices[0].message.content.strip()

        json_data = extract_json_from_response(response_text)

        if not json_data:
            raise ValueError("No valid JSON found in API response")

        extracted_data = json.loads(json_data)

        return extracted_data

    except json.JSONDecodeError:
        print("Error: Groq API returned invalid JSON format")
        return None
    except Exception as e:
        print("Error extracting details:", str(e))
        return None