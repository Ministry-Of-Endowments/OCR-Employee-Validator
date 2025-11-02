import os
import pandas as pd
import cv2
import numpy as np
import easyocr
import re
import difflib
import fitz
from PIL import Image
import io
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from datetime import datetime

ARABIC_DIACRITICS_RE = re.compile(r'[\u064B-\u065F\u0670]')
TATWEEL_RE = re.compile(r'[\u0640\u200c\u200d\u200e\u200f\u202a-\u202e]')
PUNCTUATION_RE = re.compile(r'[^\w\s\u0600-\u06FF]')
WHITESPACE_RE = re.compile(r'\s+')

class EasyOCRProcessor:
    
    def __init__(self):
        print("Initializing EasyOCR...")
        self.reader = easyocr.Reader(['ar', 'en'], gpu=True, verbose=False)
        self.executor = ThreadPoolExecutor(max_workers=4)
        
        self.clahe = cv2.createCLAHE(clipLimit=1.5, tileGridSize=(8, 8))
        self.kernel_1x1 = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 1))
        
        self.ocr_params = {
            'detail': 1,
            'paragraph': False,
            'min_size': 5,
            'text_threshold': 0.8,
            'low_text': 0.5,
            'width_ths': 0.7,
            'height_ths': 0.7,
            'mag_ratio': 2.0,
            'slope_ths': 0.1,
            'ycenter_ths': 0.5,
            'add_margin': 0.1,
            'rotation_info': None,
        }
        
        self.header_ocr_params = {
            'detail': 1,
            'paragraph': False,
            'min_size': 4,
            'text_threshold': 0.7,
            'low_text': 0.4,
            'width_ths': 0.8,
            'height_ths': 0.8,
            'mag_ratio': 2.0,
            'slope_ths': 0.2,
            'ycenter_ths': 0.7,
            'add_margin': 0.05,
            'rotation_info': None,
        }
        
        print("EasyOCR ready!")
    
    def deskew_image(self, img: np.ndarray) -> np.ndarray:
        coords = np.column_stack(np.where(img > 0))
        if len(coords) == 0:
            return img
        
        angle = cv2.minAreaRect(coords)[-1]
        if angle < -45:
            angle = -(90 + angle)
        else:
            angle = -angle
        
        if abs(angle) > 0.5:
            h, w = img.shape[:2]
            center = (w // 2, h // 2)
            M = cv2.getRotationMatrix2D(center, angle, 1.0)
            img = cv2.warpAffine(img, M, (w, h), 
                               flags=cv2.INTER_CUBIC, 
                               borderMode=cv2.BORDER_REPLICATE)
        return img
    
    def preprocess_image(self, img: np.ndarray) -> np.ndarray:
        if len(img.shape) == 3:
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        else:
            gray = img
        
        height, width = gray.shape
        max_dimension = 2500
        scale = None
        
        if height > max_dimension or width > max_dimension:
            scale = max_dimension / max(height, width)
        elif height < 1500 or width < 1500:
            scale = 1500 / min(height, width)
        
        if scale and scale != 1.0:
            interpolation = cv2.INTER_AREA if scale < 1.0 else cv2.INTER_CUBIC
            gray = cv2.resize(gray, None, fx=scale, fy=scale, interpolation=interpolation)
        
        denoised = cv2.fastNlMeansDenoising(gray, None, h=3, templateWindowSize=7, searchWindowSize=21)
        enhanced = self.clahe.apply(denoised)
        cleaned = cv2.morphologyEx(enhanced, cv2.MORPH_CLOSE, self.kernel_1x1)
        deskewed = self.deskew_image(cleaned)
        
        return deskewed
    
    def extract_text(self, img: np.ndarray) -> str:
        processed = self.preprocess_image(img)
        
        results = self.reader.readtext(processed, **self.ocr_params)
        
        try:
            header_h = max(120, int(processed.shape[0] * 0.22))
            header_region = processed[0:header_h, :]
            header_results = self.reader.readtext(header_region, **self.header_ocr_params)
            if header_results:
                results.extend(header_results)
        except:
            pass
        
        if len(img.shape) == 2:
            img_rgb = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
        elif img.shape[2] == 4:
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGRA2RGB)
        else:
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        results_original = self.reader.readtext(img_rgb, **self.ocr_params)
        
        if len(results_original) > len(results) or (len(results_original) == len(results) and sum(conf for _, _, conf in results_original) > sum(conf for _, _, conf in results)):
            results = results_original
        
        lines = {}
        for bbox, text, conf in results:
            y_center = (bbox[0][1] + bbox[2][1]) / 2
            line_key = None
            for existing_y in lines.keys():
                if abs(y_center - existing_y) < 15:
                    line_key = existing_y
                    break
            
            if line_key is None:
                line_key = y_center
                lines[line_key] = []
            
            x_center = (bbox[0][0] + bbox[2][0]) / 2
            lines[line_key].append((x_center, text))
        
        sorted_lines = sorted(lines.items())
        full_text_parts = []
        
        for y, words in sorted_lines:
            sorted_words = sorted(words, key=lambda x: x[0], reverse=True)
            line_text = ' '.join([word[1] for word in sorted_words])
            full_text_parts.append(line_text)
        
        full_text = '\n'.join(full_text_parts)
        full_text = post_process_arabic_text(full_text)
        full_text = normalize_arabic_text(full_text)
        
        return full_text.strip()


ARABIC_REPLACEMENTS = {
    'دزاره': 'وزارة', 'رزارذ': 'وزارة', 'ززاره': 'وزارة',
    'الاوفاف': 'الأوقاف', 'الارذف': 'الأوقاف', 'الذرلاف': 'الأوقاف', 'العديربه': 'الأوقاف',
    'ماه': 'حالة', 'مالسه': 'حالة',
    'وظبنبه': 'وظيفية', 'وظينيه': 'وظيفية',
    'تاربخ': 'تاريخ',
    'البلال': 'الميلاد', 'البلاد': 'الميلاد',
    'الؤهلات': 'المؤهلات',
    'طيها': 'عليها', 'عنيها': 'عليها', 'ملبه': 'عليها',
    'انحصو': 'الحصول',
    'الخدمه': 'الخدمة',
    'الالنداق': 'الانضمام', 'الالنداف': 'الانضمام',
    'بالمديريه': 'بالمديرية', 'المديربه': 'المديرية', 'المديريه': 'المديرية', 'المدبربه': 'المديرية',
    'الرفم': 'الرقم', 'رفم': 'رقم',
    'الفومي': 'الوظيفي',
    'المسمي': 'المسمى',
    'اللسنوي': 'السنوي',
    'الجزا': 'الجزاءات', 'ءات': 'ءات',
    'لا بوحد': 'لا يوجد',
    'سربر': 'سرب', 'سربف': 'سرب',
    'مسنول': 'مسؤول', 'مسرل': 'مسؤول',
    'وثايق': 'وثائق',
    'بان': 'بأن',
    'صعبحه': 'صحيحة', 'صحيحه': 'صحيحة',
    'علي': 'على',
    'درجه': 'درجة',
    'دالمه': 'دائمة', 'دائمه': 'دائمة',
    'لجنه': 'لجنة',
    'ثمنون': 'تشكيل', 'شنون': 'تشكيل',
    'التعبن': 'التعيين', 'التعبين': 'التعيين',
    'لسنه': 'لسنة',
    'البنربه': 'البشرية', 'البشريه': 'البشرية',
    'مدبر': 'مدير', 'مدبد': 'مدير',
    'بيان مالسه وظينيه': 'بيان حالة وظيفية',
    'بيان مله وظبنبه': 'بيان حالة وظيفية',
    'بيان حاله وظيفيه': 'بيان حالة وظيفية',
    'بيان حاله وظيفه': 'بيان حالة وظيفية',
}

def post_process_arabic_text(text: str) -> str:
    for wrong, correct in ARABIC_REPLACEMENTS.items():
        text = text.replace(wrong, correct)
    return text

def normalize_arabic_text(text: str) -> str:
    text = ARABIC_DIACRITICS_RE.sub('', text)
    text = TATWEEL_RE.sub('', text)
    text = text.replace('أ', 'ا').replace('إ', 'ا').replace('آ', 'ا')
    text = text.replace('ة', 'هـ')
    text = text.replace('ى', 'ي')
    text = PUNCTUATION_RE.sub(' ', text)
    text = WHITESPACE_RE.sub(' ', text.lower()).strip()
    return text


ERROR_PATTERNS_LOWER = [
    'بيان مالسه وظينيه',
    'بيان مله وظبنبه', 
    'بيان حاله وظيفيه',
    'بيان حاله وظيفه',
    'بيان حالة وظيفيه',
    'بيان حالة وظيفه',
    'وزاره الاوقاف',
    'وزارة الاوقاف',
    'وزارة الأوقاف',
    'وزاره الأوقاف'
]

def fuzzy_search_arabic(text: str, keywords: list, verbose: bool = False) -> bool:
    arabic_chars = sum(1 for c in text if '\u0600' <= c <= '\u06FF')
    total_chars = len(text.replace(' ', '').replace('\n', ''))
    
    if total_chars < 10:
        if verbose:
            print(f" TEXT TOO SHORT: {total_chars} characters")
        return False
    
    if total_chars > 0 and arabic_chars / total_chars < 0.2:
        if verbose:
            print(f"LOW ARABIC CONTENT: {arabic_chars}/{total_chars} = {arabic_chars/total_chars:.2f}")
        return False
    
    normalized_text = normalize_arabic_text(text)
    text_words = normalized_text.split()
    text_no_space = normalized_text.replace(' ', '')
    
    if verbose:
        print(f" SEARCHING IN: {normalized_text[:200]}...")
    
    for keyword in keywords:
        normalized_keyword = normalize_arabic_text(keyword)
        if verbose:
            print(f" Looking for: '{normalized_keyword}'")
        
        if normalized_keyword in normalized_text:
            if verbose:
                print(f" DIRECT MATCH: '{normalized_keyword}'")
            return True
        
        if normalized_keyword.replace(' ', '') in text_no_space:
            if verbose:
                print(f"NO-SPACE MATCH: '{normalized_keyword}'")
            return True
        
        keyword_words = normalized_keyword.split()
        if len(keyword_words) > 1 and keyword_words and all(word in text_words for word in keyword_words):
            if verbose:
                print(f"ALL WORDS MATCH: '{normalized_keyword}'")
            return True
        
        window_size = max(1, len(keyword_words))
        for i in range(0, max(1, len(text_words) - window_size + 1)):
            window = ' '.join(text_words[i:i + window_size])
            ratio = difflib.SequenceMatcher(None, normalized_keyword, window).ratio()
            if ratio >= 0.85:
                if verbose:
                    print(f"FUZZY MATCH: '{normalized_keyword}' vs '{window}' (ratio: {ratio:.2f})")
                return True
        
        kw_chars = normalized_keyword.replace(' ', '')
        if not kw_chars:
            continue
        n = len(kw_chars)
        max_window = min(len(text_no_space), n + 2)
        for w in range(n, max_window + 1):
            for i in range(0, max(0, len(text_no_space) - w + 1)):
                window = text_no_space[i:i + w]
                ratio = difflib.SequenceMatcher(None, kw_chars, window).ratio()
                if ratio >= 0.80:
                    if verbose:
                        print(f" CHAR MATCH: '{normalized_keyword}' vs '{window}' (ratio: {ratio:.2f})")
                    return True
    
    original_text_lower = text.lower()
    for pattern in ERROR_PATTERNS_LOWER:
        if pattern in original_text_lower:
            if verbose:
                print(f" ERROR PATTERN MATCH: '{pattern}'")
            return True
    
    if verbose:
        print("NO MATCHES FOUND")
    return False

def validate_document(file_path: str, ocr_processor: EasyOCRProcessor, verbose: bool = False) -> dict:
    try:
        # Support both image files and PDFs. For PDFs render the first page to an image.
        ext = os.path.splitext(file_path)[1].lower()
        if ext == '.pdf':
            pdf = fitz.open(file_path)
            if pdf.page_count == 0:
                return {"success": False, "error": "PDF has no pages"}
            page = pdf.load_page(0)
            zoom = 300 / 72  # render at 300 DPI
            pix = page.get_pixmap(matrix=fitz.Matrix(zoom, zoom))
            img_bytes = pix.tobytes("png")
            pdf.close()

            pil_img = Image.open(io.BytesIO(img_bytes)).convert("RGB")
            img = np.array(pil_img)
            if len(img.shape) == 3:
                img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        else:
            img = cv2.imread(file_path)
            if img is None:
                return {"success": False, "error": "Failed to load image"}

        text = ocr_processor.extract_text(img)
        
        keywords = [
            "بيان حالة وظيفية",
            "بيان الحالة الوظيفية",
            "بيان حاله وظيفه",
            "بيان حاله وظيفيه",
            "بيان حالة وظيفيه",
            "بيان مالسه وظينيه",
            "وزارة الأوقاف",
            "الاسم",
            "تاريخ الميلاد"
        ]
        
        is_valid = fuzzy_search_arabic(text, keywords, verbose=verbose)
        
        return {
            "success": is_valid,
            "text": text,
            "error": None if is_valid else "Keywords not found"
        }
        
    except Exception as e:
        return {"success": False, "error": str(e), "text": ""}


def process_employees(excel_file: str, test_folder: str, output_file: str = "validation_results.xlsx"):
    print("=" * 80)
    print("EMPLOYEE DOCUMENT VALIDATION SCRIPT")
    print("=" * 80)
    
    print(f"\n Reading Excel file: {excel_file}")
    df = pd.read_excel(excel_file)
    
    df = df.loc[:, ~df.columns.str.contains('^Unnamed')]
    
    print(f"Total employees: {len(df)}")
    print(f"Test folder: {test_folder}")
    
    document_column = 'بيان حالة وظيفية'
    validation_column = 'صحة بيان الحالة'
    
    if document_column not in df.columns:
        print(f"Error: Column '{document_column}' not found in Excel file")
        print(f"Available columns: {df.columns.tolist()}")
        return
    
    print(f"\nInitializing OCR processor...")
    ocr_processor = EasyOCRProcessor()
    
    validation_results = []
    valid_count = 0
    invalid_count = 0
    error_count = 0

    print(f"\nStarting validation...")
    print("-" * 80)
    
    for idx, row in df.iterrows():
        employee_name = row.get('الاسم رباعي', f'Employee_{idx}')
        document_filename = row.get(document_column)
        
        print(f"\n[{idx + 1}/{len(df)}] Processing: {employee_name}")
        
        if pd.isna(document_filename) or not document_filename:
            print(f"No document specified")
            validation_results.append(0)
            error_count += 1
            continue
        
        file_path = os.path.join(test_folder, document_filename)
        
        if not os.path.exists(file_path):
            print(f"File not found: {document_filename}")
            validation_results.append(0)
            error_count += 1
            continue
        
        print(f"Validating: {document_filename}")
        validation_result = validate_document(file_path, ocr_processor, verbose=False)
        
        if validation_result['success']:
            print(f"VALID")
            validation_results.append(1)
            valid_count += 1
        else:
            print(f"INVALID: {validation_result['error']}")
            validation_results.append(0)
            invalid_count += 1
    
    df[validation_column] = validation_results
    
    print("\n" + "=" * 80)
    print("VALIDATION COMPLETE")
    print("=" * 80)
    print(f"Valid documents: {valid_count}")
    print(f"Invalid documents: {invalid_count}")
    print(f"Errors/Missing: {error_count}")
    print(f"Total processed: {len(df)}")
    
    df.to_excel(output_file, index=False, engine='openpyxl')
    print(f"\nUpdated Excel saved to: {output_file}")
    print("=" * 80)


if __name__ == "__main__":
    import sys
    
    script_dir = os.path.dirname(os.path.abspath(__file__))
    parent_dir = os.path.dirname(script_dir)
    
    if len(sys.argv) > 1:
        excel_filename = sys.argv[1]
    else:
        excel_filename = "employees_updated.xlsx"
    
    excel_file = os.path.join(parent_dir, excel_filename)
    test_folder = os.path.join(parent_dir, "TESTFOLDER")
    output_file = os.path.join(parent_dir, f"employees_validated_{datetime.now().strftime('%Y%m%d_%H%M%S')}.xlsx")
    
    process_employees(excel_file, test_folder, output_file)
