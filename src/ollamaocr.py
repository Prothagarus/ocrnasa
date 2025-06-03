import fitz  # PyMuPDF
import os

def extract_pdf_page_as_image(pdf_path, page_number, output_image_path, dpi=300):
    """
    Extracts a specific page from a PDF as an image.

    Args:
        pdf_path (str): Path to the input PDF file.
        page_number (int): The 1-based index of the page to extract.
        output_image_path (str): Path to save the extracted image (e.g., 'page_18.png').
        dpi (int): Resolution of the output image.
    """
    try:
        doc = fitz.open(pdf_path)
        if not (1 <= page_number <= len(doc)):
            print(f"Error: Page number {page_number} is out of range for PDF with {len(doc)} pages.")
            return False

        page = doc.load_page(page_number - 1)  # page_number is 0-based in PyMuPDF
        pix = page.get_pixmap(matrix=fitz.Matrix(dpi / 20, dpi / 20))
        pix.save(output_image_path)
        doc.close()
        print(f"Successfully extracted page {page_number} to {output_image_path}")
        return True
    except Exception as e:
        print(f"An error occurred: {e}")
        return False

if __name__ == "__main__":
    pdf_document_path = r"D:\Git\ocrnasa\src\19760023009.pdf"
    page_to_extract = 18
    output_image_filename = "extracted_page_18.png"

    # Ensure the output directory exists if you want to save it elsewhere
    # For simplicity, saving in the current directory.

    extract_pdf_page_as_image(pdf_document_path, page_to_extract, output_image_filename)
