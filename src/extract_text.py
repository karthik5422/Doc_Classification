import os
import PyPDF2
import pandas as pd

def extract_text_from_pdf(pdf_path):
    """Extracts text from a PDF file."""
    text = ""
    with open(pdf_path, "rb") as file:
        reader = PyPDF2.PdfReader(file)
        for page in reader.pages:
            text += page.extract_text() or ""   
    return text

def create_labeled_dataset(data_dir="data", output_csv="labeled_data.csv"):
    """Creates a labeled dataset by extracting text from PDFs and associating labels."""
    categories = ["Warranty", "Transactions", "Troubleshooting"]
    labeled_data = []

    for category in categories:
        category_dir = os.path.join(data_dir, category)
        for filename in os.listdir(category_dir):
            if filename.endswith(".pdf"):
                pdf_path = os.path.join(category_dir, filename)
                text = extract_text_from_pdf(pdf_path)
                labeled_data.append({"text": text, "label": category})
 
    df = pd.DataFrame(labeled_data)
    df.to_csv(output_csv, index=False)
    print(f"Labeled dataset saved to {output_csv}")

if __name__ == "__main__":
    create_labeled_dataset()
