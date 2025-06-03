import pandas as pd
from pdf2image import convert_from_path
import pytesseract
from pytesseract import Output # Import Output for image_to_data
import os
import numpy as np # For numerical operations
 
# --- Configuration ---
# IMPORTANT: You need to install Tesseract OCR engine and Poppler utilities separately.
# Refer to the "Prerequisites" section above for detailed instructions.

# Path to the Tesseract executable (update this if Tesseract is not in your system's PATH)
# Example for Windows: r'C:\Program Files\Tesseract-OCR\tesseract.exe'
pytesseract.pytesseract.tesseract_cmd = 'tesseract' # Default: assumes Tesseract is in PATH

# Name of your PDF file (should be in the same directory as this script)
# Adjusted path to find 'input.pdf' in the parent directory (project root)
PDF_FILE_NAME = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '19760023009.pdf')

# Directory to save raw extracted text files for each page (optional)
OUTPUT_TEXT_DIR = 'extracted_text_pages'

# Minimum number of characters extracted from a page to consider it legible.
# Pages with less text than this threshold will be marked as illegible.
MIN_TEXT_LENGTH_FOR_LEGIBILITY = 50

# --- Table Extraction Configuration (Heuristic) ---
# Minimum confidence for a word to be considered in table extraction
TABLE_CONFIDENCE_THRESHOLD = 70
# Pixels tolerance for aligning words into the same column horizontally

COLUMN_TOLERANCE = 15 # Increased slightly for better flexibility

def extract_table_from_image(image):
    """
    Attempts to extract tabular data from an image using pytesseract's image_to_data.
    This is a heuristic approach and may not work for all table layouts.

    Args:
        image (PIL.Image.Image): The image of a page.

    Returns:
        pd.DataFrame | None: A pandas DataFrame representing the table, or None if no table
                             is confidently detected or parsed.
    """
    try:
        # Get detailed data including bounding boxes and confidence for each word
        data = pytesseract.image_to_data(image, output_type=Output.DATAFRAME)

        # Filter out low confidence words and empty text
        data = data[data.conf > TABLE_CONFIDENCE_THRESHOLD]
        data = data[data.text.notna()]
        data = data[data.text.str.strip() != '']

        if data.empty:
            return None

        # Sort by top, then by left to process words in reading order (row by row, then left to right)
        data = data.sort_values(by=['top', 'left']).reset_index(drop=True)

        # Heuristic to identify columns:
        # Group words by approximate vertical alignment (left coordinate)
        # This creates 'bins' for columns. A more robust method would involve
        # clustering or more sophisticated line/column detection.
        column_centers = []
        if not data.empty:
            # Initial pass to find potential column centers
            for i, row_data in data.iterrows():
                found_bin = False
                for j, center in enumerate(column_centers):
                    if abs(row_data['left'] - center) <= COLUMN_TOLERANCE:
                        # Update center with average to refine
                        column_centers[j] = (center * len(column_centers) + row_data['left']) / (len(column_centers) + 1)
                        found_bin = True
                        break
                if not found_bin:
                    column_centers.append(row_data['left'])

            # Sort and refine column centers
            column_centers = sorted(list(set(int(x / COLUMN_TOLERANCE) * COLUMN_TOLERANCE for x in column_centers)))
            column_centers = sorted(list(set(column_centers))) # Remove duplicates after rounding

            # Assign words to columns based on their 'left' position
            data['column_idx'] = -1
            for i, row_data in data.iterrows():
                # Find the closest column bin
                closest_col_idx = np.argmin(np.abs(np.array(column_centers) - row_data['left']))
                data.loc[i, 'column_idx'] = closest_col_idx

            # Reconstruct table row by row
            table_rows = []
            current_row_words = []
            # Initialize current_row_top with the top of the first word, or 0 if data is empty
            current_row_top = data.iloc[0]['top'] if not data.empty else 0

            for i, row_data in data.iterrows():
                # If the word is significantly below the current row's top, start a new row
                # This threshold (0.8 * height) might need adjustment based on font size/line spacing
                if row_data['top'] > current_row_top + row_data['height'] * 0.8:
                    # Process the completed row
                    if current_row_words:
                        # Create a list for the current row, filling empty cells
                        row_output = [''] * len(column_centers)
                        for word_info in current_row_words:
                            col_idx = word_info['column_idx']
                            if 0 <= col_idx < len(column_centers):
                                if row_output[col_idx]: # If cell already has content, append
                                    row_output[col_idx] += ' ' + word_info['text']
                                else:
                                    row_output[col_idx] = word_info['text']
                        table_rows.append(row_output)
                    current_row_words = []
                    current_row_top = row_data['top']

                current_row_words.append(row_data)

            # Add the last row after the loop finishes
            if current_row_words:
                row_output = [''] * len(column_centers)
                for word_info in current_row_words:
                    col_idx = word_info['column_idx']
                    if 0 <= col_idx < len(column_centers):
                        if row_output[col_idx]:
                            row_output[col_idx] += ' ' + word_info['text']
                        else:
                            row_output[col_idx] = word_info['text']
                table_rows.append(row_output)

            if table_rows:
                # Determine column names (e.g., Col1, Col2, ...)
                num_cols = max(len(row) for row in table_rows) if table_rows else 0
                column_names = [f'Col{j+1}' for j in range(num_cols)]
                # Ensure all rows have the same number of columns by padding with empty strings
                table_rows_padded = [row + [''] * (num_cols - len(row)) for row in table_rows]

                # Basic check to see if it's likely a table (e.g., more than 1 row and 1 column)
                if len(table_rows_padded) > 1 and num_cols > 1:
                    return pd.DataFrame(table_rows_padded, columns=column_names)
                else:
                    return None # Not enough structure to be considered a table
            else:
                return None

        else:
            return None # No data after filtering

    except Exception as e:
        print(f"Error extracting table: {e}")
        return None

def extract_text_from_pdf(pdf_path: str):
    """
    Extracts text from each page of a PDF using OCR, identifies potentially illegible pages,
    and attempts to extract tabular data. Returns the data in a pandas DataFrame along
    with a list of illegible page numbers.

    Args:
        pdf_path (str): The path to the input PDF file.

    Returns:
        tuple[pd.DataFrame | None, list[int]]: A tuple containing:
            - A pandas DataFrame with 'page_number', 'extracted_text', 'is_legible',
              'has_table', and 'table_data' columns. Returns None if the PDF cannot be processed.
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
        page_text = ''
        is_legible = False
        has_table = False
        table_df = None # Will store a pandas DataFrame if a table is found

        try:
            # First, perform general OCR for text and legibility check
            text = pytesseract.image_to_string(page_image)
            cleaned_text = ' '.join(text.split()).strip()
            page_text = cleaned_text

            if len(cleaned_text) < MIN_TEXT_LENGTH_FOR_LEGIBILITY:
                illegible_pages.append(page_number)
                print(f"    Page {page_number} marked as illegible (extracted text length: {len(cleaned_text)}).")
                is_legible = False
            else:
                is_legible = True
                print(f"    Page {page_number} processed successfully.")

                # Attempt to extract table if the page is deemed legible
                print(f"    Attempting table extraction for page {page_number}...")
                table_df = extract_table_from_image(page_image)
                if table_df is not None and not table_df.empty:
                    has_table = True
                    print(f"    Table detected and extracted from page {page_number}.")
                    # Optionally, save the table to a CSV
                    table_output_dir = os.path.join(OUTPUT_TEXT_DIR, 'tables')
                    if not os.path.exists(table_output_dir):
                        os.makedirs(table_output_dir)
                    table_df.to_csv(os.path.join(table_output_dir, f'page_{page_number}_table.csv'), index=False, encoding='utf-8')
                    print(f"    Table for page {page_number} saved to CSV in '{table_output_dir}'.")
                else:
                    print(f"    No significant table detected on page {page_number}.")

                # Save the raw extracted text to a file
                if not os.path.exists(OUTPUT_TEXT_DIR):
                    os.makedirs(OUTPUT_TEXT_DIR)
                with open(os.path.join(OUTPUT_TEXT_DIR, f'page_{page_number}.txt'), 'w', encoding='utf-8') as f:
                    f.write(cleaned_text)

        except Exception as e:
            print(f"    Error during OCR or table extraction on page {page_number}: {e}")
            illegible_pages.append(page_number)
            is_legible = False

        extracted_data.append({
            'page_number': page_number,
            'extracted_text': page_text, # Raw text string of the entire page
            'is_legible': is_legible,
            'has_table': has_table,
            'table_data': table_df # Store the DataFrame directly or None
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

        print("\n--- Table Extraction Summary ---")
        tables_found_count = df_result['has_table'].sum()
        print(f"Pages with detected tables: {tables_found_count}")
        if tables_found_count > 0:
            pages_with_tables = df_result[df_result['has_table'] == True]['page_number'].tolist()
            print(f"Pages with tables: {pages_with_tables}")
            print(f"Extracted tables are saved as CSVs in the '{os.path.join(OUTPUT_TEXT_DIR, 'tables')}' directory.")
            print("The 'table_data' column in the main DataFrame contains the pandas DataFrame for each detected table.")
        else:
            print("No tables were detected in any of the pages.")

        print("\n--- Extracted Data (DataFrame Head) ---")
        # Display the first few rows of the DataFrame, including new columns
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

        # Example of accessing a specific table DataFrame:
        # if tables_found_count > 0:
        #     # Get the first row that has a table
        #     first_table_row = df_result[df_result['has_table'] == True].iloc[0]
        #     print(f"\n--- Example: Table from Page {first_table_row['page_number']} (from 'table_data' column) ---")
        #     print(first_table_row['table_data'])

    else:
        print("PDF processing failed. Please check the error messages above.")

    print("\n--- Application Finished ---")
