import os
import docx

try:
    import easyocr
    reader = easyocr.Reader(['en'], gpu=False)
except Exception as e:
    print(f"OCR Init Error: {e}")
    reader = None


def extract_text_from_docx(file_path):
    abs_path = os.path.abspath(file_path)
    try:
        doc = docx.Document(abs_path)
    except Exception as e:
        print(f"DOCX Error on {file_path}: {e}")
        return ""

    full_text = []
    for para in doc.paragraphs:
        if para.text.strip():
            full_text.append(para.text.strip())
    for table in doc.tables:
        for row in table.rows:
            row_text = [cell.text.strip() for cell in row.cells if cell.text.strip()]
            if row_text:
                full_text.append(" | ".join(row_text))
    return "\n".join(full_text)


def extract_text_from_image(file_path):
    abs_path = os.path.abspath(file_path)
    if reader is None:
        raise RuntimeError("EasyOCR is unavailable. Cannot extract text from image.")
    try:
        results = reader.readtext(abs_path, detail=0)
    except Exception as e:
        raise RuntimeError(f"OCR failed to read the document: {file_path} ({e})")
    return "\n".join(results)


def process_document(file_path):
    ext = os.path.splitext(str(file_path))[1].lower()
    if ext == ".docx":
        return extract_text_from_docx(file_path)
    if ext in [".png", ".jpg", ".jpeg"]:
        return extract_text_from_image(file_path)
    return ""