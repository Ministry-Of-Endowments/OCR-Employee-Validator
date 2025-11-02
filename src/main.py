from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import cv2
import numpy as np
import easyocr
from PIL import Image
import io
import fitz
import zipfile
import re
from typing import List, Dict, Tuple
from concurrent.futures import ThreadPoolExecutor
import asyncio
import difflib
from functools import lru_cache

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://localhost:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

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

# Global processor instance (initialized once)
ocr_processor = None

@app.on_event("startup")
async def startup_event():
    """Initialize EasyOCR on startup"""
    global ocr_processor
    ocr_processor = EasyOCRProcessor()

@app.post("/api/process")
async def processFile(file: UploadFile = File(...), expectedValue: str = "test"):
    try:
        fileContent = await file.read()
        fileType = file.content_type

        if fileType == "application/zip" or file.filename.endswith(".zip"):
            result = await processZipAsync(fileContent)
            return JSONResponse(content={
                "success": True,  # ZIP processing itself succeeded
                "isZip": True,
                "totalFiles": result["totalFiles"],
                "validFiles": result["validFiles"],
                "invalidFiles": result["invalidFiles"],
                "errors": result["errors"]
            })
        elif fileType.startswith("image/"):
            result = await processImageAsync(fileContent)
        elif fileType == "application/pdf":
            result = await processPdfAsync(fileContent)
        else:
            raise HTTPException(status_code=400, detail="Unsupported file type")

        if result and result.get("success", False):
            return JSONResponse(content={
                "success": True, 
                "data": result["data"]
            })
        else:
            return JSONResponse(content={"success": False, "data": result.get("data", "") if result else ""})

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

async def processImageAsync(imageBytes):
    """Async wrapper for image processing"""
    loop = asyncio.get_event_loop()
    return await loop.run_in_executor(ocr_processor.executor, processImage, imageBytes)


@lru_cache(maxsize=128)
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

def fuzzy_search_arabic(text: str, keywords: list) -> bool:
    arabic_chars = sum(1 for c in text if '\u0600' <= c <= '\u06FF')
    total_chars = len(text.replace(' ', '').replace('\n', ''))
    
    if total_chars < 10:
        print(f"TEXT TOO SHORT: {total_chars} characters")
        return False
    
    if total_chars > 0 and arabic_chars / total_chars < 0.2:
        print(f"LOW ARABIC CONTENT: {arabic_chars}/{total_chars} = {arabic_chars/total_chars:.2f}")
        return False
    
    normalized_text = normalize_arabic_text(text)
    text_words = normalized_text.split()
    text_no_space = normalized_text.replace(' ', '')
    
    print(f"SEARCHING IN: {normalized_text[:200]}...")
    
    for keyword in keywords:
        normalized_keyword = normalize_arabic_text(keyword)
        print(f"Looking for: '{normalized_keyword}'")
        
        if normalized_keyword in normalized_text:
            print(f"DIRECT MATCH: '{normalized_keyword}'")
            return True
        
        if normalized_keyword.replace(' ', '') in text_no_space:
            print(f"NO-SPACE MATCH: '{normalized_keyword}'")
            return True
        
        keyword_words = normalized_keyword.split()
        if len(keyword_words) > 1 and keyword_words and all(word in text_words for word in keyword_words):
            print(f"ALL WORDS MATCH: '{normalized_keyword}'")
            return True
        
        window_size = max(1, len(keyword_words))
        for i in range(0, max(1, len(text_words) - window_size + 1)):
            window = ' '.join(text_words[i:i + window_size])
            ratio = difflib.SequenceMatcher(None, normalized_keyword, window).ratio()
            if ratio >= 0.85:
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
                    print(f"CHAR MATCH: '{normalized_keyword}' vs '{window}' (ratio: {ratio:.2f})")
                    return True
    
    original_text_lower = text.lower()
    for pattern in ERROR_PATTERNS_LOWER:
        if pattern in original_text_lower:
            print(f"ERROR PATTERN MATCH: '{pattern}'")
            return True
    
    print("NO MATCHES FOUND")
    return False

def processImage(imageBytes):
    npArr = np.frombuffer(imageBytes, np.uint8)
    img = cv2.imdecode(npArr, cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError("Failed to decode image")
    
    # Extract text using EasyOCR
    text = ocr_processor.extract_text(img)
    
    # DEBUG: Print extracted text to console
    print("=== EXTRACTED TEXT ===")
    print(text)
    print("=== END EXTRACTED TEXT ===")
    
    # Check for key phrases in the document with fuzzy matching
    keywords = [
        "بيان حالة وظيفية",
        "بيان الحالة الوظيفية",
        # common OCR corruptions/variants
        "بيان حاله وظيفه",
        "بيان حاله وظيفيه",
        "بيان حالة وظيفيه",
        "بيان مالسه وظينيه",
        "وزارة الأوقاف",
        "الاسم",
        "تاريخ الميلاد"
    ]
    
    if fuzzy_search_arabic(text, keywords):
        return {"success": True, "data": text}
    
    print("⚠️ NO KEYWORDS MATCHED")
    return {"success": False, "data": text}

async def processZipAsync(zipBytes):
    validFiles = []
    invalidFiles = []
    errors = []
    totalFiles = 0
    
    try:
        with zipfile.ZipFile(io.BytesIO(zipBytes)) as zipFile:
            fileList = [f for f in zipFile.namelist() 
                       if not f.startswith('__MACOSX') 
                       and not f.startswith('.') 
                       and not f.endswith('/')]
            
            validFileList = []
            fileDataList = []
            for fileName in fileList:
                fileExt = fileName.lower().split('.')[-1]
                if fileExt in ['jpg', 'jpeg', 'png', 'gif', 'bmp', 'pdf']:
                    validFileList.append(fileName)
                    fileDataList.append(zipFile.read(fileName))
                    totalFiles += 1
            
            results = await asyncio.gather(*[
                processPdfAsync(fileData) if fileName.lower().endswith('.pdf') else processImageAsync(fileData)
                for fileName, fileData in zip(validFileList, fileDataList)
            ], return_exceptions=True)
            
            for fileName, result in zip(validFileList, results):
                if isinstance(result, Exception):
                    errors.append({
                        "filename": fileName,
                        "error": str(result)
                    })
                    continue
                
                print(f"=== ZIP FILE: {fileName} ===")
                print(f"Result: {result}")
                print(f"Success: {result.get('success', False) if result else False}")
                print(f"Data preview: {result.get('data', '')[:100] if result and result.get('data') else 'No data'}")
                print("=== END ZIP FILE ===")
                
                if result and result.get("success", False):
                    validFiles.append({
                        "filename": fileName,
                        "text": result["data"]
                    })
                else:
                    invalidFiles.append({
                        "filename": fileName,
                        "reason": "Phrase not found"
                    })
                    
    except zipfile.BadZipFile:
        raise ValueError("Invalid ZIP file")
    
    return {
        "totalFiles": totalFiles,
        "validFiles": validFiles,
        "invalidFiles": invalidFiles,
        "errors": errors
    }

async def processPdfAsync(pdfBytes):
    """Async wrapper for PDF processing"""
    loop = asyncio.get_event_loop()
    return await loop.run_in_executor(ocr_processor.executor, processPdf, pdfBytes)

def processPdf(pdfBytes):
    pdfDocument = fitz.open(stream=pdfBytes, filetype="pdf")
    if pdfDocument.page_count == 0:
        raise ValueError("PDF has no pages")
    
    page = pdfDocument.load_page(0)
    zoom = 300 / 72
    pix = page.get_pixmap(matrix=fitz.Matrix(zoom, zoom))
    imgData = pix.tobytes("png")
    pdfDocument.close()
    
    img = Image.open(io.BytesIO(imgData))
    imgArray = np.array(img)
    
    if len(imgArray.shape) == 3:
        imgBgr = cv2.cvtColor(imgArray, cv2.COLOR_RGB2BGR)
    else:
        imgBgr = imgArray
    
    text = ocr_processor.extract_text(imgBgr)
    
    print("=== EXTRACTED TEXT (PDF) ===")
    print(text)
    print("=== END EXTRACTED TEXT ===")
    
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
    
    if fuzzy_search_arabic(text, keywords):
        return {"success": True, "data": text}
    
    print("NO KEYWORDS MATCHED")
    return {"success": False, "data": text}

@app.get("/")
async def root():
    return {"message": "QR Code Processor API with EasyOCR"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
