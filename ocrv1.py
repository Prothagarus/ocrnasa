import pandas as pd
from pdf2image import convert_from_path
import pytesseract
import os

# --- Configuration ---
# IMPORTANT: You need to install Tesseract OCR engine and Poppler utilities separately.
# Refer to the "Prerequisites" section above for detailed instructions.

# Path to the Tesseract executable (update this if Tesseract is not in your system's PATH)
# Example for Windows: r'C:\Program Files\Tesseract-OCR\tesseract.exe'
pytesseract.pytesseract.tesseract_cmd = 'tesseract' # Default: assumes Tesseract is in PATH

# Name of your PDF file (should be in the same directory as this script)
PDF_FILE_NAME = 'input.pdf'

# Directory to save raw extracted text files for each page (optional)
OUTPUT_TEXT_DIR = 'extracted_text_pages'

# Minimum number of characters extracted from a page to consider it legible.
# Pages with less text than this threshold will be marked as illegible.
MIN_TEXT_LENGTH_FOR_LEGIBILITY = 50

def extract_text_from_pdf(pdf_path: str):
    """
    Extracts text from each page of a PDF using OCR, identifies potentially illegible pages,
    and returns the data in a pandas DataFrame along with a list of illegible page numbers.

    Args:
        pdf_path (str): The path to the input PDF file.

    Returns:
        tuple[pd.DataFrame | None, list[int]]: A tuple containing:
            - A pandas DataFrame with 'page_number', 'extracted_text', and 'is_legible' columns.
              Returns None if the PDF cannot be processed.
            - A list of page numbers identified as illegible.
    """
    if not os.path.exists(pdf_path):
        print(f"Error: PDF file not found at '{pdf_path}'")
        return None, []

    print(f"Converting PDF '{pdf_path}' to images...")
    pages = []
    try:
        # Convert PDF pages to PIL Image objects.
        # '300' DPI is recommended for better OCR accuracy.
        # If using Windows, you might need to specify poppler_path here:
        # pages = convert_from_path(pdf_path, 300, poppler_path=r'C:\path\to\poppler-0.68.0\bin')
        pages = convert_from_path(pdf_path, 300)
    except Exception as e:
        print(f"Error converting PDF to images. Ensure Poppler is installed and accessible.")
        print(f"Details: {e}")
        return None, []

    extracted_data = []
    illegible_pages = []

    print(f"Performing OCR on {len(pages)} pages...")
    for i, page_image in enumerate(pages):
        page_number = i + 1
        print(f"  Processing page {page_number}...")
        try:
            # Perform OCR using Tesseract
            text = pytesseract.image_to_string(page_image)

            # Clean up text (remove excessive whitespace, newlines)
            cleaned_text = ' '.join(text.split()).strip()

            # Check for legibility based on the length of extracted text
            if len(cleaned_text) < MIN_TEXT_LENGTH_FOR_LEGIBILITY:
                illegible_pages.append(page_number)
                print(f"    Page {page_number} marked as illegible (extracted text length: {len(cleaned_text)}).")
                extracted_data.append({
                    'page_number': page_number,
                    'extracted_text': '', # Store empty string for illegible pages
                    'is_legible': False
                })
            else:
                extracted_data.append({
                    'page_number': page_number,
                    'extracted_text': cleaned_text,
                    'is_legible': True
                })
                print(f"    Page {page_number} processed successfully.")

                # Save the raw extracted text to a file
                if not os.path.exists(OUTPUT_TEXT_DIR):
                    os.makedirs(OUTPUT_TEXT_DIR)
                with open(os.path.join(OUTPUT_TEXT_DIR, f'page_{page_number}.txt'), 'w', encoding='utf-8') as f:
                    f.write(cleaned_text)

        except Exception as e:
            print(f"    Error performing OCR on page {page_number}: {e}")
            illegible_pages.append(page_number)
            extracted_data.append({
                'page_number': page_number,
                'extracted_text': '',
                'is_legible': False
            })

    df = pd.DataFrame(extracted_data)
    return df, illegible_pages

if __name__ == "__main__":
    print("--- Starting PDF OCR Application ---")

    # Run the OCR process
    df_result, illegible_pages_list = extract_text_from_pdf(PDF_FILE_NAME)

    if df_result is not None:
        print("\n--- OCR Summary ---")
        total_pages = len(df_result)
        legible_count = df_result['is_legible'].sum()
        illegible_count = len(illegible_pages_list)

        print(f"Total pages processed: {total_pages}")
        print(f"Legible pages: {legible_count}")
        print(f"Illegible pages identified: {illegible_count}")

        if illegible_pages_list:
            print(f"Pages marked for reference (illegible): {illegible_pages_list}")
        else:
            print("No illegible pages identified. All pages appear legible.")

        print("\n--- Extracted Data (DataFrame Head) ---")
        # Display the first few rows of the DataFrame
        print(df_result.head())

        print("\n--- DataFrame Info ---")
        # Display DataFrame structure and data types
        df_result.info()

        # --- Further Actions ---
        # You can now work with 'df_result'.
        # For example, save the entire DataFrame to a CSV file:
        # df_result.to_csv('extracted_pdf_data_all_pages.csv', index=False, encoding='utf-8')
        # print("\nFull DataFrame saved to 'extracted_pdf_data_all_pages.csv'")

        # Or, filter for only the legible pages and save them:
        legible_df = df_result[df_result['is_legible'] == True].copy()
        if not legible_df.empty:
            print("\n--- Legible Pages Data (DataFrame Head) ---")
            print(legible_df.head())
            # legible_df.to_csv('extracted_pdf_data_legible_pages.csv', index=False, encoding='utf-8')
            # print("\nLegible pages DataFrame saved to 'extracted_pdf_data_legible_pages.csv'")
        else:
            print("\nNo legible pages found to create a separate DataFrame.")

        print(f"\nRaw text files for legible pages are saved in the '{OUTPUT_TEXT_DIR}' directory.")

    else:
        print("PDF processing failed. Please check the error messages above.")

    print("\n--- Application Finished ---")