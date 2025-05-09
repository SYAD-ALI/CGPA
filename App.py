from flask import Flask, request, jsonify, current_app
from flask_cors import CORS
import pdfplumber
import re
import PyPDF2
import logging
import io
import traceback
import datetime

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

app = Flask(__name__)
# Configure CORS with more options
CORS(app, resources={
    r"/*": {
        "origins": "*",  # In production, specify actual origins
        "methods": ["GET", "POST", "OPTIONS"],
        "allow_headers": ["Content-Type", "Authorization"]
    }
})

# Grade point mapping
grade_points = {
    "O": 10,
    "A+": 9,
    "A": 8,
    "B+": 7,
    "B": 6,
    "C": 5,
    "P": 4,
    "F": 0,
}


def extract_text_from_pdf(file_storage):
    """
    Extract tables from PDF using pdfplumber with better error handling
    """
    try:
        # Create a copy of the file in memory to avoid file pointer issues
        file_copy = io.BytesIO(file_storage.read())
        file_storage.seek(0)  # Reset file pointer
        
        with pdfplumber.open(file_copy) as pdf:
            tables = []
            for page in pdf.pages:
                try:
                    table = page.extract_table()
                    if table:
                        tables.append(table)
                except Exception as e:
                    logger.warning(f"Failed to extract table from page: {e}")
            return tables
    except Exception as e:
        logger.error(f"Failed to extract tables from PDF: {e}")
        raise


def extract_result_text(file_storage):
    """
    Extract text from PDF using PyPDF2 with better error handling
    """
    try:
        # Create a copy of the file in memory
        file_copy = io.BytesIO(file_storage.read())
        file_storage.seek(0)  # Reset file pointer
        
        reader = PyPDF2.PdfReader(file_copy)
        text_pages = []
        
        for page in reader.pages:
            text = page.extract_text()
            if text:
                text_pages.append(text)
        
        return "\n".join(text_pages) if text_pages else ""
    except Exception as e:
        logger.error(f"Failed to extract text from PDF: {e}")
        raise


def extract_course_details(tables):
    """
    Extract course details with advanced pattern matching and row analysis
    """
    logger.info("Extracting course details from tables...")
    subjects = {}
    known_titles = set()  # To avoid duplicate titles
    
    # Find header row to understand table structure
    header_indices = {}
    table_headers = ['S.No', 'Course Title', 'Course Code', 'L', 'T', 'P', 'C', 'Credits']
    
    def find_header_indices(row):
        indices = {}
        if not row:
            return {}
        
        for i, cell in enumerate(row):
            if not cell:
                continue
            cell_str = str(cell).strip().lower()
            
            # Match headers approximately
            for header in table_headers:
                header_lower = header.lower()
                if header_lower in cell_str or cell_str in header_lower:
                    indices[header] = i
                    
        return indices
    
    # Process each table
    for page_idx, page in enumerate(tables):
        if not page:
            continue
        
        # Try to find header row
        for row_idx, row in enumerate(page):
            header_indices = find_header_indices(row)
            if len(header_indices) >= 3:  # Found enough headers
                logger.info(f"Found header row on page {page_idx}, row {row_idx}: {header_indices}")
                break
        
        # Process data rows
        for row_idx, row in enumerate(page):
            if not row or len(row) < 5:
                continue
                
            try:
                # Clean and normalize the row data
                cleaned_row = [str(cell).strip() if cell else "" for cell in row]
                
                # Try multiple approaches to identify course codes
                
                # Approach 1: Direct pattern match
                code_candidates = [col for col in cleaned_row if re.match(r'23[A-Z0-9]{5,}', str(col))]
                
                # Approach 2: Use header indices if available
                code_from_header = None
                title_from_header = None
                credits_from_header = None
                
                if 'Course Code' in header_indices and len(cleaned_row) > header_indices['Course Code']:
                    code_cell = cleaned_row[header_indices['Course Code']]
                    if re.match(r'23[A-Z0-9]{4,}', code_cell):
                        code_from_header = code_cell
                        
                if 'Course Title' in header_indices and len(cleaned_row) > header_indices['Course Title']:
                    title_from_header = cleaned_row[header_indices['Course Title']]
                    
                if 'Credits' in header_indices and len(cleaned_row) > header_indices['Credits']:
                    credits_cell = cleaned_row[header_indices['Credits']]
                    if re.match(r'^[0-9]+(\.[0-9]+)?$', credits_cell):
                        credits_from_header = float(credits_cell)
                
                # Determine code, title and credits based on available information
                code = code_from_header if code_from_header else (code_candidates[0] if code_candidates else None)
                
                # Skip if no valid code found
                if not code:
                    continue
                
                # Find title
                if title_from_header:
                    title = title_from_header
                else:
                    # Title is typically a longer string with alphabets
                    title_candidates = [
                        col for col in cleaned_row 
                        if len(str(col)) > 5 and re.search(r'[a-zA-Z]{5,}', str(col)) and col != code
                    ]
                    
                    # Sort by length to likely get the full title
                    title_candidates.sort(key=len, reverse=True)
                    title = title_candidates[0] if title_candidates else "Unknown"
                
                # Find credits
                if credits_from_header is not None:
                    credits = credits_from_header
                else:
                    # Try to find credits at the end of the row
                    credit_candidates = []
                    for col in cleaned_row:
                        if re.match(r'^[0-9]+(\.[0-9]+)?$', str(col)):
                            try:
                                credit_val = float(col)
                                if 0 < credit_val <= 10:  # Valid credit range
                                    credit_candidates.append(credit_val)
                            except ValueError:
                                pass
                    
                    credits = credit_candidates[-1] if credit_candidates else 0
                
                # Add subject if we have valid data
                if code and title and credits > 0:
                    # Avoid duplicates by checking if we already have this code
                    if code not in subjects:
                        subjects[code] = {
                            'title': title,
                            'credits': credits,
                            'grade': None,
                            'original_code': code,  # Store original code for reference
                            'normalized_title': re.sub(r'[^a-zA-Z0-9]', '', title.lower())  # Store normalized title for fuzzy matching
                        }
                        known_titles.add(title)
                        logger.info(f"Extracted subject: {code} - {title} ({credits} credits)")
            except Exception as e:
                logger.warning(f"Failed to process row {row_idx} on page {page_idx}: {e}")
                logger.debug(f"Problematic row: {row}")
    
    logger.info(f"Extracted {len(subjects)} subjects in total")
    
    if len(subjects) < 5:
        logger.warning("Very few subjects extracted. This might indicate an issue with the course file format.")
    
    return subjects


def clean_result_text(text):
    """
    Advanced text cleaning with intelligent line aggregation and pattern recognition
    """
    lines = text.split('\n')
    subject_lines = []
    
    # Pre-processing: clean and normalize lines
    cleaned_lines = []
    for line in lines:
        line = line.strip()
        if not line:  # Skip empty lines
            continue
            
        # Remove excessive whitespace
        line = re.sub(r'\s+', ' ', line)
        
        # Fix common OCR issues
        line = line.replace('0', 'O')  # Common OCR mistake with grades
        
        cleaned_lines.append(line)
    
    logger.info(f"Pre-processed {len(cleaned_lines)} non-empty lines")
    
    # First pass: identify potential subject code patterns
    subject_code_pattern = r'23[A-Z0-9]{5,}'
    subject_codes = set()
    
    for line in cleaned_lines:
        matches = re.findall(subject_code_pattern, line)
        subject_codes.update(matches)
    
    logger.info(f"Found {len(subject_codes)} potential subject codes in text")
    
    # Second pass: identify subject lines and aggregate related content
    current_subject = None
    subject_buffer = ""
    grade_indicators = r'(GRADE|RESULT|CGPA|GPA|MARKS|SCORE)'
    
    for i, line in enumerate(cleaned_lines):
        # Check for subject code in this line
        code_matches = re.findall(subject_code_pattern, line)
        
        if code_matches:
            # New subject found
            if subject_buffer:
                subject_lines.append(subject_buffer)
                
            subject_buffer = line
            current_subject = code_matches[0]
            
            # Check if grade is in the same line
            if re.search(r'[OABCPF][\+]?|RA|AB|W', line):
                # If grade is already in the line, we can end the buffer
                subject_lines.append(subject_buffer)
                subject_buffer = ""
                current_subject = None
        
        elif current_subject:
            # Continue current subject if line might contain grade info
            if re.search(r'[OABCPF][\+]?|RA|AB|W', line) or re.search(grade_indicators, line, re.IGNORECASE):
                subject_buffer += " " + line
                
                # If we found grade info, we can end this subject
                if re.search(r'[OABCPF][\+]?|RA|AB|W', line):
                    subject_lines.append(subject_buffer)
                    subject_buffer = ""
                    current_subject = None
            
            # Check if next line has a different subject code
            elif i < len(cleaned_lines) - 1 and re.search(subject_code_pattern, cleaned_lines[i+1]):
                # End current subject before starting new one
                subject_lines.append(subject_buffer)
                subject_buffer = ""
                current_subject = None
            
            else:
                # Continue appending data for current subject
                subject_buffer += " " + line
    
    # Don't forget the last subject
    if subject_buffer:
        subject_lines.append(subject_buffer)
    
    logger.info(f"Cleaned and identified {len(subject_lines)} subject result lines")
    
    # Third pass: Additional parsing for lines without clear subject codes
    # Look for lines with grade information that weren't captured
    grade_lines = []
    for line in cleaned_lines:
        # If line contains grade pattern but wasn't included in any subject
        if re.search(r'[OABCPF][\+]?|RA|AB|W', line) and not any(line in subj for subj in subject_lines):
            # Check if it's not just a header or general text
            if not re.search(r'(TABLE|HEADER|GRADE.*POINT|RESULT.*SUMMARY)', line, re.IGNORECASE):
                grade_lines.append(line)
    
    # Combine the results
    all_lines = subject_lines + grade_lines
    
    # If very few lines were identified, fall back to simpler parsing
    if len(all_lines) < 5:
        logger.warning("Few subject lines identified. Using fallback parsing method.")
        # Simple parsing: just look for lines with subject codes or grades
        all_lines = [
            line for line in cleaned_lines 
            if re.search(subject_code_pattern, line) or re.search(r'[OABCPF][\+]?|RA|AB|W', line)
        ]
    
    logger.info(f"Final count: {len(all_lines)} processed result lines")
    return all_lines, cleaned_lines  # Return both processed lines and all cleaned lines


def normalize_code(code):
    """
    Normalize course code to handle variations
    """
    # Remove any non-alphanumeric characters
    code = re.sub(r'[^A-Z0-9]', '', code.upper())
    # Handle common OCR mistakes
    code = code.replace('O', '0')
    code = code.replace('I', '1')
    code = code.replace('Z', '2')
    return code


def extract_grades(cleaned_lines, all_cleaned_lines, subjects):
    """
    Enhanced grade extraction with better handling of mismatched course codes
    """
    logger.info("Extracting grades from result lines...")
    
    # Prepare grade pattern - all possible grades for robust matching
    all_grades = '|'.join(['O', 'A\\+', 'A', 'B\\+', 'B', 'C', 'P', 'F', 'AB', 'RA', 'W'])
    grade_pattern = f"({all_grades})"
    
    # Create a map of normalized codes to original codes
    norm_to_orig = {}
    code_to_title = {}
    title_to_code = {}
    
    for code, subject in subjects.items():
        norm_code = normalize_code(code)
        norm_to_orig[norm_code] = code
        code_to_title[code] = subject['title']
        title_to_code[subject['normalized_title']] = code
    
    # First, try direct mapping of codes to lines
    subject_lines_map = {}
    result_codes = set()
    
    for line in cleaned_lines:
        # Extract codes from the line
        line_upper = line.upper()
        code_matches = re.findall(r'23[A-Z0-9]{5,}', line_upper)
        
        for code_match in code_matches:
            result_codes.add(code_match)
            norm_match = normalize_code(code_match)
            
            # Check if we have a matching subject
            if norm_match in norm_to_orig:
                orig_code = norm_to_orig[norm_match]
                if orig_code not in subject_lines_map or re.search(grade_pattern, line_upper):
                    subject_lines_map[orig_code] = line
    
    # Helper function to extract grade from a line using various patterns
    def extract_grade_from_line(line, code):
        # Normalize line for consistent matching
        line = line.upper().strip()
        
        # List of patterns to try in order of specificity
        patterns = [
            # Pattern 1: Very specific - code followed closely by grade
            rf"{code}\s*[-:]\s*({all_grades})",
            
            # Pattern 2: Code followed by text and ending with grade
            rf"{code}.*?({all_grades})\s*$",
            
            # Pattern 3: Find grade at the end of the line
            rf".*?({all_grades})\s*$",
            
            # Pattern 4: Grade with clear separation (spaces, brackets, etc)
            rf"[^A-Z0-9]({all_grades})[^A-Z0-9]",
            
            # Pattern 5: Look for grade within specific sections
            rf"GRADE\s*[:-]?\s*({all_grades})",
            
            # Pattern 6: Most general - any valid grade in the line
            rf"({all_grades})"
        ]
        
        for pattern in patterns:
            matches = re.findall(pattern, line)
            if matches:
                # Use the last match as it's typically the one after the code
                grade = matches[-1]
                # Handle tuple results from regex groups
                if isinstance(grade, tuple):
                    grade = grade[0]
                return grade.strip()
        
        return None
    
    # Track how grades were found for logging
    found_methods = {
        "direct_mapping": 0,
        "fuzzy_code_mapping": 0,
        "title_mapping": 0,
        "multiple_line_search": 0,
        "not_found": 0
    }
    
    # Log detected mismatches between course registration and result codes
    if result_codes:
        for code in subjects.keys():
            norm_code = normalize_code(code)
            found_exact = code in result_codes
            found_norm = any(normalize_code(rc) == norm_code for rc in result_codes)
            
            if not found_exact and not found_norm:
                logger.warning(f"Potential code mismatch: {code} (from course registration) not found in result sheet")
    
    # Process each subject
    for code, subject in subjects.items():
        # Initialize the grade field if not present
        if 'grade' not in subject:
            subject['grade'] = None
            
        # Strategy 1: Direct mapping from subject code to result line
        if code in subject_lines_map:
            line = subject_lines_map[code]
            grade = extract_grade_from_line(line, code)
            
            if grade:
                subject['grade'] = grade
                found_methods["direct_mapping"] += 1
                logger.info(f"Grade found for {code}: {grade} (direct mapping)")
                continue
        
        # Strategy 2: Try fuzzy matching for subject codes
        norm_code = normalize_code(code)
        for line in cleaned_lines:
            line_upper = line.upper()
            if norm_code in normalize_code(line_upper):
                grade = extract_grade_from_line(line, "")  # Don't filter by code here
                
                if grade:
                    subject['grade'] = grade
                    found_methods["fuzzy_code_mapping"] += 1
                    logger.info(f"Grade found for {code}: {grade} (fuzzy code mapping)")
                    break
        
        # Strategy 3: Match by subject title
        if not subject['grade']:
            subject_title = subject['title'].upper()
            for line in all_cleaned_lines:  # Use all cleaned lines for title matching
                line_upper = line.upper()
                
                # Check if substantial part of the title appears in the line
                title_words = re.findall(r'\b\w{4,}\b', subject_title)  # Important words in title (4+ chars)
                
                if title_words:
                    # Count how many significant title words appear in the line
                    matches = sum(1 for word in title_words if word in line_upper)
                    if matches >= max(1, len(title_words) // 2):  # At least half of important words match
                        grade = extract_grade_from_line(line, "")
                        
                        if grade:
                            subject['grade'] = grade
                            found_methods["title_mapping"] += 1
                            logger.info(f"Grade found for {code}: {grade} (title mapping)")
                            break
        
        # Strategy 4: Look for similar codes in result that might be variants
        if not subject['grade']:
            code_prefix = code[:6]  # First 6 chars (23DEPT)
            for result_code in result_codes:
                if result_code.startswith(code_prefix):
                    # Potentially matching code variant
                    for line in cleaned_lines:
                        if result_code in line.upper():
                            grade = extract_grade_from_line(line, "")
                            
                            if grade:
                                subject['grade'] = grade
                                found_methods["multiple_line_search"] += 1
                                logger.info(f"Grade found for {code}: {grade} (variant code search)")
                                # Add mapping info for debugging
                                subject['possible_variant_code'] = result_code
                                break
                if subject['grade']:
                    break
        
        # Log if no grade found
        if not subject['grade']:
            found_methods["not_found"] += 1
            logger.warning(f"Could not extract grade for {code} - {subject['title']}")
    
    # Count subjects with grades
    graded_subjects = sum(1 for s in subjects.values() if s['grade'])
    logger.info(f"Extracted grades for {graded_subjects}/{len(subjects)} subjects")
    logger.info(f"Grade extraction methods: {found_methods}")
    
    # If too many grades not found, provide diagnostic info
    if found_methods["not_found"] > len(subjects) * 0.3:
        logger.warning("High grade extraction failure rate. Check result PDF format.")
        
        # Log some sample lines for diagnosis
        logger.debug("Sample lines from result text:")
        for i, line in enumerate(cleaned_lines[:5]):
            logger.debug(f"Line {i}: {line}")
    
    return subjects


def calculate_sgpa(subjects):
    """
    Calculate SGPA with improved error handling
    """
    total_credits = 0
    total_points = 0
    
    for code, sub in subjects.items():
        credits = sub['credits']
        grade = sub['grade']
    
        if grade and grade in grade_points:
            gp = grade_points[grade]
            total_credits += credits
            total_points += gp * credits
            logger.info(f"Subject {code}: {credits} credits × {gp} points = {credits * gp}")
    
    if total_credits == 0:
        logger.warning("No valid credits found, SGPA calculation impossible")
        return 0
        
    sgpa = total_points / total_credits
    logger.info(f"SGPA calculation: {total_points} points / {total_credits} credits = {sgpa}")
    
    return round(sgpa, 2)


def validate_files(request):
    """
    Validate uploaded files
    """
    if 'courses' not in request.files or 'results' not in request.files:
        raise ValueError("Missing required files: 'courses' and 'results'")
        
    course_file = request.files['courses']
    result_file = request.files['results']
    
    if not course_file.filename.lower().endswith('.pdf'):
        raise ValueError("Course file must be a PDF")
        
    if not result_file.filename.lower().endswith('.pdf'):
        raise ValueError("Result file must be a PDF")
        
    return course_file, result_file


@app.route("/upload", methods=["POST"])
def upload():
    """
    Improved upload endpoint with better error handling and debugging info
    """
    try:
        logger.info("Processing new upload request")
        
        # Validate files
        course_file, result_file = validate_files(request)
        
        # Extract data from PDFs
        logger.info("Extracting data from course file")
        course_tables = extract_text_from_pdf(course_file)
        
        logger.info("Extracting data from result file")
        result_text = extract_result_text(result_file)
        
        # Process extracted data
        subjects = extract_course_details(course_tables)
        
        if not subjects:
            return jsonify({
                'error': 'No subjects found in the course file',
                'debug_info': {'course_tables_count': len(course_tables)}
            }), 400
        
        cleaned_lines, all_cleaned_lines = clean_result_text(result_text)
        subjects = extract_grades(cleaned_lines, all_cleaned_lines, subjects)
        
        # Record subjects without grades for debugging
        subjects_without_grades = [code for code, subject in subjects.items() if not subject['grade']]
        if subjects_without_grades:
            logger.warning(f"Subjects without grades: {subjects_without_grades}")
        
        # Calculate SGPA
        sgpa = calculate_sgpa(subjects)
        
        # Prepare response data - clean up internal fields before sending
        response_subjects = {}
        for code, subject in subjects.items():
            response_subjects[code] = {
                'title': subject['title'],
                'credits': subject['credits'],
                'grade': subject['grade']
            }
            # Include variant code info if available
            if 'possible_variant_code' in subject:
                response_subjects[code]['possible_variant_code'] = subject['possible_variant_code']
        
        # Prepare summary
        summary = {
            'total_subjects': len(subjects),
            'subjects_with_grades': sum(1 for s in subjects.values() if s['grade']),
            'total_credits': sum(s['credits'] for s in subjects.values() if s['grade']),
            'total_points': sum(s['credits'] * grade_points.get(s['grade'], 0) for s in subjects.values() if s['grade']),
        }
        
        # Check if we have enough grades
        if summary['subjects_with_grades'] < summary['total_subjects'] * 0.7:
            logger.warning(f"Low grade extraction rate: {summary['subjects_with_grades']}/{summary['total_subjects']}")
        
        logger.info(f"Successfully processed request: SGPA = {sgpa}")
        
        return jsonify({
            'sgpa': sgpa,
            'subjects': response_subjects,
            'summary': summary
        })

    except ValueError as e:
        logger.warning(f"Validation error: {str(e)}")
        return jsonify({'error': str(e)}), 400
        
    except Exception as e:
        logger.error(f"Internal error: {str(e)}")
        logger.error(traceback.format_exc())
        return jsonify({
            'error': 'An internal error occurred',
            'message': str(e),
            'debug_info': {
                'type': type(e).__name__,
                'trace': traceback.format_exc()
            }
        }), 500


@app.route("/validate", methods=["POST"])
def validate_pdf():
    """
    Endpoint to validate PDF files before full processing
    """
    try:
        if 'file' not in request.files:
            return jsonify({'valid': False, 'error': 'No file provided'}), 400
            
        file = request.files['file']
        file_type = request.args.get('type', 'unknown')
        
        if not file.filename.lower().endswith('.pdf'):
            return jsonify({'valid': False, 'error': 'File must be a PDF'}), 400
            
        # Create a copy of the file in memory
        file_copy = io.BytesIO(file.read())
        file.seek(0)  # Reset file pointer
        
        # Basic validation - check if it's a readable PDF
        try:
            reader = PyPDF2.PdfReader(file_copy)
            page_count = len(reader.pages)
            
            # Basic content validation
            if page_count == 0:
                return jsonify({'valid': False, 'error': 'PDF has no pages'}), 400
                
            # Try to extract first page text
            first_page_text = reader.pages[0].extract_text() or ""
            
            # Specific validation based on file type
            if file_type == 'courses':
                # Check for course-related keywords
                course_keywords = ['course', 'credit', 'subject', 'curriculum', 'syllabus']
                if not any(keyword in first_page_text.lower() for keyword in course_keywords):
                    return jsonify({
                        'valid': True, 
                        'warning': 'This may not be a course PDF. Please verify content.'
                    })
            
            elif file_type == 'results':
                # Check for result-related keywords
                result_keywords = ['result', 'grade', 'cgpa', 'sgpa', 'semester', 'examination']
                if not any(keyword in first_page_text.lower() for keyword in result_keywords):
                    return jsonify({
                        'valid': True, 
                        'warning': 'This may not be a results PDF. Please verify content.'
                    })
            
            return jsonify({
                'valid': True,
                'page_count': page_count,
                'file_size': len(file_copy.getvalue()),
                'message': 'PDF validation successful'
            })
            
        except Exception as e:
            logger.warning(f"PDF validation failed: {str(e)}")
            return jsonify({
                'valid': False, 
                'error': 'Invalid or corrupted PDF file'
            }), 400
            
    except Exception as e:
        logger.error(f"Validation endpoint error: {str(e)}")
        return jsonify({
            'valid': False,
            'error': f'An error occurred during validation: {str(e)}'
        }), 500


@app.route("/health", methods=["GET"])
def health_check():
    """
    Health check endpoint for monitoring
    """
    return jsonify({
        'status': 'ok',
        'version': '1.0.0',
        'timestamp': datetime.datetime.now().isoformat()
    })


@app.errorhandler(404)
def not_found(error):
    return jsonify({'error': 'Endpoint not found'}), 404


@app.errorhandler(405)
def method_not_allowed(error):
    return jsonify({'error': 'Method not allowed'}), 405


@app.errorhandler(500)
def server_error(error):
    return jsonify({'error': 'Internal server error'}), 500


if __name__ == "__main__":
    logger.info("Starting SGPA Calculator backend server")
    app.run(debug=True, host='0.0.0.0', port=5000)

