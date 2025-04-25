import os
import cv2
import numpy as np
import pandas as pd
import pytesseract
import fitz
from pdf2image import convert_from_path
from PIL import Image
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor
import re
from collections import defaultdict
import multiprocessing

class PDFTableExtractor:
    def __init__(self, pdf_path, output_dir="extracted_tables", dpi=300, 
                 tesseract_path=None, poppler_path=None):
        """
        Initialize the PDF Table Extractor.
        
        Args:
            pdf_path: Path to the PDF file
            output_dir: Directory to save the output Excel files
            dpi: DPI for PDF to image conversion (higher means better quality but slower)
            tesseract_path: Path to tesseract executable (if not in PATH)
            poppler_path: Path to poppler binaries (if not in PATH)
        """
        self.pdf_path = pdf_path
        self.output_dir = output_dir
        self.dpi = dpi
        
        # Configure external dependencies
        if tesseract_path:
            pytesseract.pytesseract.tesseract_cmd = tesseract_path
            
        self.poppler_path = poppler_path
        
        # Create output directory if it doesn't exist
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
            
        # Get PDF info
        # self.doc = PyMuPDF.open(pdf_path)
        self.doc = fitz.open(pdf_path) 
        self.num_pages = len(self.doc)
        print(f"PDF loaded: {pdf_path}")
        print(f"Number of pages: {self.num_pages}")
        
        # Table format storage
        self.table_formats = defaultdict(list)
        
    def process_all_pages(self, start_page=0, end_page=None, batch_size=10):
        """Process all pages in the PDF in batches for memory efficiency"""
        if end_page is None:
            end_page = self.num_pages
            
        total_batches = (end_page - start_page + batch_size - 1) // batch_size
        
        for batch_idx in range(total_batches):
            batch_start = start_page + batch_idx * batch_size
            batch_end = min(batch_start + batch_size, end_page)
            
            print(f"Processing batch {batch_idx+1}/{total_batches} (pages {batch_start+1}-{batch_end})")
            self._process_page_batch(batch_start, batch_end)
            
        # After processing all pages, export tables to Excel
        self.export_tables_to_excel()
        
    def _process_page_batch(self, start_page, end_page):
        """Process a batch of pages"""
        pages_to_process = list(range(start_page, end_page))
        
        # Convert pages to images for this batch
        images = convert_from_path(
            self.pdf_path, 
            dpi=self.dpi, 
            first_page=start_page+1, 
            last_page=end_page,
            poppler_path=self.poppler_path,
            fmt="png"
        )
        
        # Process pages sequentially
        for page_num, img in tqdm(zip(pages_to_process, images), 
                                total=len(pages_to_process),
                                desc="Extracting tables"):
            table_data = self._process_single_page_image((page_num, img))
            if table_data[1]:  # If there are tables detected
                for table_id, table_content in table_data[1].items():
                    self.table_formats[table_id].append({
                        'page': page_num + 1,
                        'data': table_content
                    })
    
    def _process_single_page_image(self, page_data):
        """Process a single page image to detect and extract tables"""
        page_num, img = page_data
        
        # Convert PIL image to OpenCV format
        img_cv = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
        
        # Detect tables in the image
        tables = self._detect_tables(img_cv)

        
        
        if not tables:
            return page_num, {}
            
        # Extract content from each table
        table_data = {}
        for idx, table_coords in enumerate(tables):
            x, y, w, h = table_coords
            table_img = img_cv[y:y+h, x:x+w]
            
            # Extract table structure and content
            detected_table = self._extract_table_content(table_img)
            # print("Original tables format before extracting table content: ", detected_table)
            if detected_table is not None:
                # Determine table format (number of columns)
                num_cols = len(detected_table[0]) if detected_table else 0
                table_id = f"table_format_{num_cols}cols"
                
                table_data[table_id] = detected_table
                print("\nTable Detection Details:")
                print(f"Page Number: {page_num + 1}")
                print(f"Table ID: {table_id}")
                print("Detected Rows:")
                for row_idx, row in enumerate(detected_table):
                    print(f"Row {row_idx + 1}: {row}")
                print("-" * 50)
                
        return page_num, table_data
 

    def _detect_tables(self, img):
        """Detect tables in an image using computer vision techniques"""
        # Convert to grayscale
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        # Apply thresholding with adjusted parameters
        _, binary = cv2.threshold(gray, 180, 255, cv2.THRESH_BINARY_INV)
        
        # Enhance line detection
        kernel_vertical = np.ones((5, 1), np.uint8)
        kernel_horizontal = np.ones((1, 5), np.uint8)
        
        # Dilate to connect components
        dilated_v = cv2.dilate(binary, kernel_vertical, iterations=2)
        dilated_h = cv2.dilate(binary, kernel_horizontal, iterations=2)
        dilated = cv2.bitwise_or(dilated_v, dilated_h)
        
        # Find contours with different parameters
        contours, _ = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # Filter contours to find tables with adjusted parameters
        tables = []
        for contour in contours:
            x, y, w, h = cv2.boundingRect(contour)
            
            # Adjusted size filtering
            if w > img.shape[1] * 0.15 and h > img.shape[0] * 0.03:
                # Add larger margin
                margin = 15
                x = max(0, x - margin)
                y = max(0, y - margin)
                w = min(img.shape[1] - x, w + 2*margin)
                h = min(img.shape[0] - y, h + 2*margin)
                
                tables.append((x, y, w, h))

        return tables

    def _extract_table_content(self, table_img):
        """Extract content from a table image using line detection and OCR"""
        # Convert to grayscale
        gray = cv2.cvtColor(table_img, cv2.COLOR_BGR2GRAY)
        
        # Apply thresholding
        _, binary = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY_INV)
        
        # Detect horizontal and vertical lines
        horizontal_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (40, 1))
        vertical_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 40))
        
        horizontal_lines = cv2.morphologyEx(binary, cv2.MORPH_OPEN, horizontal_kernel, iterations=3)
        vertical_lines = cv2.morphologyEx(binary, cv2.MORPH_OPEN, vertical_kernel, iterations=3)
        
        # Combine horizontal and vertical lines
        table_structure = cv2.add(horizontal_lines, vertical_lines)
        
        
        # Find contours of the cells
        contours, _ = cv2.findContours(table_structure, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        
        if not contours:
            # If no clear table structure is detected, use alternative approach
            return self._extract_table_without_lines(table_img)
            # return None
        
        # Extract cell coordinates
        cells = []
        for contour in contours:
            x, y, w, h = cv2.boundingRect(contour)
            # Filter small noise
            if w > 20 and h > 20:
                cells.append((x, y, x+w, y+h))
        
        if not cells:
            return None
            
        # Sort cells by rows and columns
        cells.sort(key=lambda c: (c[1], c[0]))  # Sort by y then x
        
        # Group cells by rows
        row_groups = []
        current_row = [cells[0]]
        y_tolerance = 10  # Reduced tolerance for better row separation
        
        for cell in cells[1:]:
            # If y-coordinate is close to previous cell, it's in the same row
            if abs(cell[1] - current_row[0][1]) < y_tolerance:
                current_row.append(cell)
            else:
                # Sort cells in the row by x-coordinate
                current_row.sort(key=lambda c: c[0])
                row_groups.append(current_row)
                current_row = [cell]
                
        if current_row:
            current_row.sort(key=lambda c: c[0])
            row_groups.append(current_row)
        
        # Extract text from each cell
        table_data = []
        for row in row_groups:
            row_data = []
            for cell in row:
                x1, y1, x2, y2 = cell
                # cell_img = gray[y1:y2, x1:x2]
                
                # # Apply OCR to the cell
                # text = pytesseract.image_to_string(cell_img, config='--psm 6').strip()
                # row_data.append(text)

                # Add padding to cell image
                padding = 4 #2
                x1 = max(0, x1 - padding)
                y1 = max(0, y1 - padding)
                x2 = min(table_img.shape[1], x2 + padding)
                y2 = min(table_img.shape[0], y2 + padding)
                
                cell_img = table_img[y1:y2, x1:x2]
                

                # # Enhance image before OCR
                # cell_img = cv2.cvtColor(cell_img, cv2.COLOR_BGR2GRAY)
                # _, cell_img = cv2.threshold(cell_img, 180, 255, cv2.THRESH_BINARY)
                
                # # Apply OCR with adjusted configuration
                # text = pytesseract.image_to_string(
                #     cell_img, 
                #     config='--psm 6 --oem 3 -c preserve_interword_spaces=1'
                # ).strip()
                # row_data.append(text)


                # Enhanced image preprocessing
                cell_img = cv2.cvtColor(cell_img, cv2.COLOR_BGR2GRAY)
                # Apply adaptive thresholding instead of simple thresholding
                cell_img = cv2.adaptiveThreshold(
                    cell_img,
                    255,
                    cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                    cv2.THRESH_BINARY,
                    11,  # Block size
                    2    # C constant
                )
                
                # Denoise the image
                cell_img = cv2.fastNlMeansDenoising(cell_img)
                
                # Apply OCR with improved configuration
                text = pytesseract.image_to_string(
                    cell_img, 
                    config='--psm 6 --oem 3 -c preserve_interword_spaces=1 -c tessedit_char_whitelist=0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz-.,:/\\ '
                ).strip()
                row_data.append(text)
                
            # if row_data:
            #     table_data.append(row_data)
            if row_data:
                # Check if this row doesn't start with 'COM'
                first_cell = row_data[0].strip()
                if not first_cell.startswith('COM') and not first_cell.startswith('SPL'):
                    # Add two empty columns at the beginning
                    row_data = ['', ''] + row_data
                table_data.append(row_data)
                
        # # Normalize table structure (ensure all rows have same number of columns)
        # if table_data:
        #     max_cols = max(len(row) for row in table_data)
        #     for row in table_data:
        #         while len(row) < max_cols:
        #             row.append("")
        
        # print("This is table data: ", table_data)


        # # Normalize table structure and clean duplicates
        # if table_data:
        #     # Get all unique identifiers from non-first rows
        #     unique_identifiers = set()
        #     for row in table_data[1:]:
        #         # Get identifiers from columns 2 and 3 (part numbers and codes)
        #         if len(row) > 2:
        #             unique_identifiers.add(row[2].strip() if row[2] else '')
        #             if len(row) > 3:
        #                 unique_identifiers.add(row[3].strip() if row[3] else '')

        #     # # Clean first row if it exists
        #     # if len(table_data) > 0:
        #     #     first_row_text = table_data[0][2] if len(table_data[0]) > 2 else ''
        #     #     cleaned_parts = []
                
        #     #     # Split the text by newlines and process each part
        #     #     for part in first_row_text.split('\n'):
        #     #         # Check if this part contains any unique identifier
        #     #         should_keep = True
        #     #         for identifier in unique_identifiers:
        #     #             if identifier and identifier in part:
        #     #                 should_keep = False
        #     #                 break
        #     #         if should_keep:
        #     #             cleaned_parts.append(part)
                
        #     #     # Update first row with cleaned content
        #     #     if cleaned_parts:
        #     #         table_data[0][2] = '\n'.join(cleaned_parts)
        #     #     else:
        #     #         table_data.pop(0)  # Remove first row if it's empty after cleaning

        #     # Normalize number of columns
        #     max_cols = max(len(row) for row in table_data)
        #     for row in table_data:
        #         while len(row) < max_cols:
        #             row.append("")

   
        # Process aircraft models in the first row
        table_data = self._process_aircraft_models(table_data)
        # Process last row content
        table_data = self._process_last_row_content(table_data)

        # Normalize columns
        if table_data:
            max_cols = 5  # Expected number of columns
            for row in table_data:
                while len(row) < max_cols:
                        row.append("")


        return table_data
    

    def _extract_table_without_lines(self, table_img):
        """Extract table content when lines are not clearly detectable"""
        # Convert PIL Image to text using Tesseract with table detection
        config = r'--oem 3 --psm 6 -c preserve_interword_spaces=1'
        text = pytesseract.image_to_string(table_img, config=config)
        
        # Split text into lines
        lines = [line.strip() for line in text.split('\n') if line.strip()]
        
        # Try to determine the table structure from spacing patterns
        processed_rows = []
        for line in lines:
            # Look for consistent spacing patterns
            spaces = [m.start() for m in re.finditer('\\s{2,}', line)]
            if spaces:
                # Use spaces as column separators
                row = []
                prev_pos = 0
                for pos in spaces:
                    row.append(line[prev_pos:pos].strip())
                    prev_pos = pos
                row.append(line[prev_pos:].strip())
                # processed_rows.append(row) 
                processed_rows.append("rows")
            else:
                # Single column row
                # processed_rows.append("Single Cell")
                processed_rows.append([line])
                
        # Normalize table structure (ensure all rows have same number of columns)
        if processed_rows:
            max_cols = max(len(row) for row in processed_rows)
            for row in processed_rows:
                while len(row) < max_cols:
                    row.append("")
        # print("This is processed_rows: ", processed_rows)
        return processed_rows
              
    def export_tables_to_excel(self):
        """Export all tables to a single Excel file"""
        all_data = []
        current_page = None
        max_cols = 0
        
        # Get all tables from all formats
        all_tables = []
        for tables in self.table_formats.values():
            all_tables.extend(tables)
            # print("All tables: ", all_tables) 
            
        # Sort all tables by page number
        all_tables.sort(key=lambda t: t['page'])
        
        # First pass to determine maximum number of columns
        for table_info in all_tables:
            data = table_info['data']
            for row in data:
                max_cols = max(max_cols, len(row))
        
        # Second pass to build data with consistent columns
        for table_info in all_tables:
            page = table_info['page']
            data = table_info['data']
            
            # Add page separator if not the first table
            if current_page is not None:
                all_data.append([""] * max_cols)  # Empty row as separator
                all_data.append([f"Page {page}"] + [""] * (max_cols - 1))
                all_data.append([""] * max_cols)  # Empty row as separator
                # print("thi is in export_tables_to_excel: ", all_data)
            else:
                all_data.append([f"Page {page}"] + [""] * (max_cols - 1))
            
            # Ensure each row has the same number of columns
            for row in data:
                padded_row = row + [""] * (max_cols - len(row))
                all_data.append(padded_row)
                
            current_page = page
        
        # Create headers based on actual column count
        headers = [f"Column {i+1}" for i in range(max_cols)]
        
        df = pd.DataFrame(all_data, columns=headers)
        
        # Generate file name
        output_file = os.path.join(
            self.output_dir, 
            f"{os.path.splitext(os.path.basename(self.pdf_path))[0]}_all_tables.xlsx"
        )
        
        # Export to Excel
        df.to_excel(output_file, index=False, engine='openpyxl')
        print(f"Exported all tables to {output_file}")
    
    def _process_aircraft_models(self, table_data):
        """Process and extract aircraft models from the first row if it contains model numbers"""
        if not table_data or len(table_data) < 1:
            return table_data
            
        first_row = table_data[0]
        if len(first_row) < 3:
            return table_data
            
        # Check if the row contains aircraft model information
        content = first_row[2] if first_row[2] else ''
        if any(content.startswith(prefix) for prefix in ['-200', '-300', '200', '300', '777']):
            # Extract the first aircraft model section (up to first -F)
            models = []
            lines = content.split('\n')
            current_model = []
            
            for line in lines:
                if line.strip():
                    current_model.append(line.strip())
                    if '-F' in line:
                        break
            
            if current_model:
                # Create new row with the extracted model
                model_text = '\n'.join(current_model)
                new_row = ['', '', '', '', model_text]
                
                # Insert the new row after the first row
                table_data.insert(1, new_row)
                
        return table_data


    # def _process_last_row_content(self, table_data):
    #     """Process the last row content and create a new row if needed"""
    #     if not table_data or len(table_data) < 2:
    #         return table_data
            
    #     first_row = table_data[0]
    #     last_row = table_data[-1]
        
    #     # print("\nDebug: Processing last row content")
    #     # print(f"Last Row: {last_row}")
    #     # print("\nFirst Row Content:")
    #     # print(first_row[2] if len(first_row) > 2 else "No content")
        
    #     # Check if both rows have enough columns
    #     if len(first_row) > 2 and len(last_row) > 4:
    #         try:
    #             # Get the part number and code from last row
    #             part_num = last_row[2].strip()
    #             code = last_row[3].strip()
    #             models = last_row[4].strip()
                
    #             print(f"\nSearching for:")
    #             print(f"Part Number: {part_num}")
    #             print(f"Code: {code}")
    #             print(f"Models: {models}")
                
    #             first_row_content = first_row[2]
    #             lines = first_row_content.split('\n')
                
    #             # Find the line containing our part number and code
    #             for i, line in enumerate(lines):
    #                 if part_num in line and code in line:
    #                     print(f"\nFound match in line: {line}")
                        
    #                     # Look for the next valid content line
    #                     for next_line in lines[i+1:]:
    #                         next_line = next_line.strip()
    #                         if next_line:
    #                             # Split by multiple spaces and filter empty parts
    #                             parts = [p for p in re.split(r'\s{2,}', next_line) if p.strip()]
    #                             if len(parts) >= 2:
    #                                 # Extract part number (first part)
    #                                 next_part_num = parts[0].strip()
    #                                 # Extract code (second part)
    #                                 next_code = parts[1].strip()
    #                                 # Extract models (everything after code)
    #                                 next_models = ' '.join(parts[2:]).strip()
                                    
    #                                 print(f"Found next content: {next_part_num}, {next_code}, {next_models}")
    #                                 new_row = ['', '', next_part_num, next_code, next_models]
    #                                 table_data.append(new_row)
    #                                 break
    #                     break
                
    #         except IndexError:
    #             print(f"Warning: Row does not have enough columns: {last_row}")
    #             return table_data
    #     else:
    #         print("Rows don't have enough columns for processing")
        
    #     return table_data
    
    def _process_last_row_content(self, table_data):
        """Process the last row content and create a new row if needed"""
        if not table_data or len(table_data) < 2:
            return table_data
            
        first_row = table_data[0]
        last_row = table_data[-1]
        
        # Check if both rows have enough columns
        if len(first_row) > 2 and len(last_row) > 4:
            try:
                # Get the part number and code from last row
                part_num = last_row[2].strip()
                code = last_row[3].strip()
                models = last_row[4].strip()
                
                print(f"\nSearching for:")
                print(f"Part Number: {part_num}")
                print(f"Code: {code}")
                print(f"Models: {models}")
                
                first_row_content = first_row[2]
                lines = first_row_content.split('\n')
                
                # Find the line containing our part number and code
                for i, line in enumerate(lines):
                    if part_num in line and code in line:
                        print(f"\nFound match in line: {line}")
                        
                        # Look for the next valid content line
                        for next_line in lines[i+1:]:
                            next_line = next_line.strip()
                            if next_line:
                                # Split by multiple spaces and filter empty parts
                                parts = [p for p in re.split(r'\s{2,}', next_line) if p.strip()]
                                if len(parts) >= 2:
                                    # Check if the line starts with COM or SPL
                                    if parts[0].startswith(('COM-', 'SPL-')):
                                        # Create new row with COM/SPL in first column
                                        next_part_num = parts[0].strip()
                                        next_code = parts[1].strip()
                                        # Extract models (last part)
                                        next_models = parts[-1].strip() if len(parts) > 2 else ''
                                        new_row = [next_part_num, next_code, '', '', next_models]
                                    else:
                                        # Handle normal case
                                        next_part_num = parts[0].strip()
                                        next_code = parts[1].strip()
                                        next_models = ' '.join(parts[2:]).strip()
                                        new_row = ['', '', next_part_num, next_code, next_models]
                                    
                                    print(f"Found next content: {next_part_num}, {next_code}, {next_models}")
                                    table_data.append(new_row)
                                    break
                        break
                
            except IndexError:
                print(f"Warning: Row does not have enough columns: {last_row}")
                return table_data
        else:
            print("Rows don't have enough columns for processing")
        
        # Delete row 1 after all processing is complete
        if len(table_data) > 0:
            table_data.pop(0)

        return table_data

    def cleanup(self):
        """Clean up resources"""
        self.doc.close()
        print("PDF processing completed")


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description="Extract tables from PDF and convert to Excel")
    parser.add_argument("pdf_path", help="Path to the PDF file")
    parser.add_argument("--output-dir", default="extracted_tables", help="Directory to save output Excel files")
    parser.add_argument("--dpi", type=int, default=300, help="DPI for PDF to image conversion")
    parser.add_argument("--tesseract-path", help="Path to Tesseract executable (if not in PATH)")
    parser.add_argument("--poppler-path", help="Path to Poppler binaries (if not in PATH)")
    parser.add_argument("--start-page", type=int, default=0, help="First page to process (0-indexed)")
    parser.add_argument("--end-page", type=int, default=None, help="Last page to process (0-indexed)")
    
    args = parser.parse_args()
    
    extractor = PDFTableExtractor(
        pdf_path=args.pdf_path,
        output_dir=args.output_dir,
        dpi=args.dpi,
        tesseract_path=args.tesseract_path,
        poppler_path=args.poppler_path
    )
    
    try:
        extractor.process_all_pages(args.start_page, args.end_page)
    finally:
        extractor.cleanup()


if __name__ == "__main__":
    main()