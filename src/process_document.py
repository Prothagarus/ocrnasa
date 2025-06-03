# process_document.py
from ollama_ocr import OCRProcessor
import os

def process_extracted_page(image_path, model_name='llama3.2-vision:11b', format_type="markdown"):
    """
    Processes an image using Ollama-OCR.

    Args:
        image_path (str): Path to the image file.
        model_name (str): The Ollama vision model to use.
        format_type (str): Desired output format (e.g., "markdown", "text", "json").
    """
    if not os.path.exists(image_path):
        print(f"Error: Image file not found at {image_path}")
        return

    # Initialize OCR processor
    # You might need to adjust base_url if Ollama is not running on default host.docker.internal
    # For local Ollama, "http://localhost:11434/api/generate" is common.
    ocr = OCRProcessor(model_name=model_name, base_url="http://localhost:11434/api/generate")

    print(f"Processing image: {image_path} with model: {model_name}...")
    result = ocr.process_image(
        image_path=image_path,
        format_type=format_type,
        language="English" # Specify the language for better accuracy
    )
    print("\n--- Extracted Text ---")
    print(result)
    print("----------------------")

if __name__ == "__main__":
    extracted_image_path = "extracted_page_18.png" # This should be the image created by extract_page.py
    process_extracted_page(extracted_image_path)