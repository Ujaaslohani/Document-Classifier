# Document Classification and Extraction Service

## Overview
This project is a Python-based document classification and extraction service that processes document images and classifies them into three categories: **invoice, budget, and email**. If a document is classified as an **invoice**, key details such as invoice number, date, amount, and vendor details are extracted using the **Groq AI API**.

[![Watch the video](https://example.com/thumbnail.jpg)](https://drive.google.com/file/d/1sEnWsWkI8wl4MKU9iJzuMiXGqujt9Gyr/view?usp=sharing)

The project includes:
- **Document classification** using a Convolutional Neural Network (CNN)
- **Text extraction** from images using OCR (Tesseract)
- **Invoice data extraction** powered by the **Groq AI API**
- **Deployment-ready REST API** using FastAPI

## Features
- **Classifies images into**: `invoice`, `budget`, `email`
- **Extracts structured details** from invoices using AI
- **REST API** with endpoints for classification and data extraction
- **Modular codebase** for easy customization
- **Uses sklearn, OpenCV, FastAPI, and Groq AI API**

## Folder Structure
```
.
├── docs-sm
│   ├── invoice
│   ├── budget
│   ├── email
├── models
│   ├── classifier.pkl
├── scripts
│   ├── train_model.py
│   ├── ocr_extraction.py
│   ├── extract_invoice.py
├── api   
|    ├── main.py
├── requirements.txt
├── README.md
```

## Setup and Installation
### 1. Clone the Repository
```sh
$ git clone <https://github.com/Ujaaslohani/Document-Classifier.git>
$ cd Document-Classification
```

### 2. Create a Virtual Environment
```sh
$ python -m venv venv
$ source venv/bin/activate  # On Windows use `venv\Scripts\activate`
```

### 3. Install Dependencies
```sh
$ pip install -r requirements.txt
```

### 4. Set Up API Keys
- Create an account on **Groq AI** and get an API key.
- Export the key as an environment variable:
  ```sh
  $ export GROQ_API_KEY="your_api_key_here"
  ```

### 5. Train the Model (Optional)
If you want to retrain the classification model, run:
```sh
$ python scripts/train_model.py
```

### 6. Run the API Service
```sh
$ uvicorn src.api:app --host 0.0.0.0 --port 8000
```

## API Endpoints
### 1. Classify and Extract Document Data
**Endpoint:**
```http
POST /classify_and_extract
```
**Request:** (Upload an image file)
```json
{
    "file": "invoice_image.jpg"
}
```
**Response:**
```json
{
    "category": "invoice",
    "details": {
        "invoice_number": "12345",
        "date": "2024-01-29",
        "amount": "$1500",
        "vendor": "ABC Corp"
    }
}
```

## Technologies Used
- **Python 3.12**
- **OpenCV & Tesseract OCR** (for text extraction)
- **FastAPI** (for REST API)
- **Groq AI API** (for invoice data extraction)
- **scikit-learn & joblib** (for model serialization)

## Future Improvements
- Extend classification categories
- Improve OCR performance with advanced techniques
- Deploy using Docker and cloud services

## Author
Developed by **Ujaas Lohani** as part of an AI-powered document processing solution.

