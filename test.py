import gradio as gr
import pdfplumber
from io import BytesIO

def process_pdf(file):
    # Check if a file was uploaded
    if file is None:
        return "No file provided."

    # Read the PDF directly from the uploaded file content (file is a BytesIO object)
    text = []
    with pdfplumber.open(BytesIO(file)) as pdf:
        for page in pdf.pages:
            extracted_text = page.extract_text()
            if extracted_text:
                text.append(extracted_text)

    # Joining page texts with separators
    return "\n\n---\n\n".join(text) if text else "No text extracted."

# Setup Gradio interface
demo = gr.Interface(
    fn=process_pdf,
    inputs=gr.File(type="binary", label="Upload a PDF"),
    outputs="text",
    title="PDF Text Extractor",
    description="Upload a PDF file to extract and display its text."
)

demo.launch()
