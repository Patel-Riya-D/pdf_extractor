"""
PDF Extraction Benchmark
========================
Change LIBRARY below to test each one:

    "pymupdf"      - native text layer (fastest, needs text-based PDF)
    "pdfplumber"   - native text layer with layout
    "paddleocr"    - deep learning OCR
    "rapidocr"     - lightweight OCR (no PaddlePaddle needed)
    "unstructured" - structured element extraction
    "docling"      - IBM structured parser -> Markdown/JSON output
    "florence2"    - Microsoft Florence-2 vision model OCR (GPU recommended)

Install all:
    pip install pymupdf pdfplumber paddlepaddle paddleocr rapidocr-onnxruntime unstructured[pdf]
    pip install docling
    pip install transformers timm einops pdf2image    # for florence2

OCR-FROM-IMAGE TEST:
    Set TEST_OCR_ON_IMAGE = True to also extract text from images
    embedded inside the PDF (true OCR challenge).
"""

import time, os
import fitz   # pymupdf - used for page count + image extraction

PDF_PATH         = "4th_sample-report_english_final.pdf"  # <-- your PDF
LIBRARY          = "florence2"    # <-- CHANGE THIS to switch library
TEST_OCR_ON_IMAGE = True          # <-- True = also OCR the embedded images

os.environ["FLAGS_use_mkldnn"]      = "0"
os.environ["PADDLE_DISABLE_MKLDNN"] = "1"


# ── helpers ───────────────────────────────────────────────────────────────────

def get_embedded_images(path):
    """Return list of PIL Images extracted from the PDF via pymupdf."""
    from PIL import Image
    import io
    doc  = fitz.open(path)
    imgs = []
    for page in doc:
        for img in page.get_images(full=True):
            xref = img[0]
            raw  = doc.extract_image(xref)
            imgs.append(Image.open(io.BytesIO(raw["image"])).convert("RGB"))
    return imgs


# ── runners ───────────────────────────────────────────────────────────────────

def run_pymupdf(path):
    doc  = fitz.open(path)
    text = ""
    imgs = []
    for page in doc:
        text += page.get_text()
        for img in page.get_images(full=True):
            imgs.append(doc.extract_image(img[0])["image"])
    return text, len(imgs)


def run_pdfplumber(path):
    import pdfplumber
    text, imgs = "", []
    with pdfplumber.open(path) as pdf:
        for page in pdf.pages:
            text += page.extract_text() or ""
            imgs += page.images
    return text, len(imgs)


def run_paddleocr(path):
    import numpy as np
    from pdf2image import convert_from_path
    from paddleocr import PaddleOCR
    ocr   = PaddleOCR(use_textline_orientation=True, lang="en", device="cpu")
    pages = convert_from_path(path, dpi=200)
    text  = ""
    for page in pages:
        result = list(ocr.predict(np.array(page)))
        if result:
            r = result[0]
            if isinstance(r, dict):
                texts = r.get("rec_texts") or r.get("txts") or []
            elif hasattr(r, "rec_texts"):
                texts = r.rec_texts or []
            elif hasattr(r, "txts"):
                texts = r.txts or []
            else:
                texts = [line[1][0] for line in (r or []) if line]
            text += " ".join(texts) + "\n"
    return text, 0


def run_rapidocr(path):
    from rapidocr_onnxruntime import RapidOCR
    from pdf2image import convert_from_path
    import numpy as np
    ocr   = RapidOCR()
    pages = convert_from_path(path, dpi=200)
    text  = ""
    for page in pages:
        result, _ = ocr(np.array(page))
        if result:
            text += " ".join([line[1] for line in result]) + "\n"
    return text, 0


def run_unstructured(path):
    from unstructured.partition.pdf import partition_pdf
    elements = partition_pdf(path)
    text     = "\n".join([str(e) for e in elements])
    imgs     = [e for e in elements if "Image" in type(e).__name__]
    return text, len(imgs)


def run_docling(path):
    from docling.document_converter import DocumentConverter
    converter = DocumentConverter()
    result    = converter.convert(path)
    doc       = result.document
    text      = doc.export_to_markdown()
    # count figures/images in the docling document
    n_imgs    = len([item for item, _ in doc.iterate_items()
                     if "Picture" in type(item).__name__])
    return text, n_imgs


def run_florence2(path):
    import torch
    from transformers import AutoProcessor, AutoModelForCausalLM
    from pdf2image import convert_from_path

    MODEL_ID  = "microsoft/Florence-2-large"
    device    = "cuda" if torch.cuda.is_available() else "cpu"
    dtype     = torch.float16 if device == "cuda" else torch.float32

    print(f"  Loading Florence-2 on {device} ...")
    processor = AutoProcessor.from_pretrained(MODEL_ID, trust_remote_code=True)
    model     = AutoModelForCausalLM.from_pretrained(
                    MODEL_ID, torch_dtype=dtype, trust_remote_code=True).to(device)

    pages  = convert_from_path(path, dpi=150)
    text   = ""
    prompt = "<OCR>"

    for i, page in enumerate(pages, 1):
        print(f"  Florence-2 processing page {i}/{len(pages)} ...")
        inputs = processor(text=prompt, images=page, return_tensors="pt").to(device, dtype)
        ids    = model.generate(
                    input_ids=inputs["input_ids"],
                    pixel_values=inputs["pixel_values"],
                    max_new_tokens=1024,
                    num_beams=3,
                )
        out   = processor.batch_decode(ids, skip_special_tokens=False)[0]
        text += processor.post_process_generation(
                    out, task=prompt, image_size=page.size)[prompt] + "\n"

    return text, 0


# ── OCR on embedded images ────────────────────────────────────────────────────

def ocr_images_with(library, images):
    """Run OCR on a list of PIL images using the chosen library."""
    if not images:
        return ""

    text = ""

    if library == "paddleocr":
        import numpy as np
        from paddleocr import PaddleOCR
        ocr = PaddleOCR(use_textline_orientation=True, lang="en", device="cpu")
        for img in images:
            result = list(ocr.predict(np.array(img)))
            if result:
                r = result[0]
                if isinstance(r, dict):
                    texts = r.get("rec_texts") or []
                elif hasattr(r, "rec_texts"):
                    texts = r.rec_texts or []
                else:
                    texts = []
                text += " ".join(texts) + "\n"

    elif library == "rapidocr":
        from rapidocr_onnxruntime import RapidOCR
        import numpy as np
        ocr = RapidOCR()
        for img in images:
            result, _ = ocr(np.array(img))
            if result:
                text += " ".join([l[1] for l in result]) + "\n"

    elif library == "florence2":
        import torch
        from transformers import AutoProcessor, AutoModelForCausalLM
        MODEL_ID  = "microsoft/Florence-2-large"
        device    = "cuda" if torch.cuda.is_available() else "cpu"
        dtype     = torch.float16 if device == "cuda" else torch.float32
        processor = AutoProcessor.from_pretrained(MODEL_ID, trust_remote_code=True)
        model     = AutoModelForCausalLM.from_pretrained(
                        MODEL_ID, torch_dtype=dtype, trust_remote_code=True).to(device)
        prompt = "<OCR>"
        for img in images:
            inputs = processor(text=prompt, images=img, return_tensors="pt").to(device, dtype)
            ids    = model.generate(input_ids=inputs["input_ids"],
                                    pixel_values=inputs["pixel_values"],
                                    max_new_tokens=512, num_beams=3)
            out    = processor.batch_decode(ids, skip_special_tokens=False)[0]
            text  += processor.post_process_generation(
                        out, task=prompt, image_size=img.size)[prompt] + "\n"

    elif library in ("pymupdf", "pdfplumber", "docling"):
        # these don't have an OCR engine; use pytesseract as fallback
        try:
            import pytesseract
            for img in images:
                text += pytesseract.image_to_string(img) + "\n"
        except ImportError:
            text = "[pytesseract not installed — pip install pytesseract]"

    elif library == "unstructured":
        # unstructured handles image OCR internally; nothing extra needed
        text = "[unstructured runs OCR on images automatically during partition_pdf]"

    return text


# ── main ──────────────────────────────────────────────────────────────────────

runners = {
    "pymupdf":      run_pymupdf,
    "pdfplumber":   run_pdfplumber,
    "paddleocr":    run_paddleocr,
    "rapidocr":     run_rapidocr,
    "unstructured": run_unstructured,
    "docling":      run_docling,
    "florence2":    run_florence2,
}

if LIBRARY not in runners:
    print(f"Unknown library '{LIBRARY}'. Choose from: {list(runners)}")
else:
    page_count = len(fitz.open(PDF_PATH))
    print(f"\nLibrary : {LIBRARY}")
    print(f"PDF     : {PDF_PATH}  ({page_count} pages)")
    print("-" * 45)

    # --- Part 1: extract text from PDF pages ---
    t0           = time.perf_counter()
    text, n_imgs = runners[LIBRARY](PDF_PATH)
    elapsed      = time.perf_counter() - t0

    print(f"[PDF]  Time         : {elapsed:.2f}s")
    print(f"[PDF]  Chars extract: {len(text)}")
    print(f"[PDF]  Images found : {n_imgs}")
    print(f"\n--- Text preview (first 500 chars) ---")
    print(text[:500])

    # --- Part 2: OCR on embedded images ---
    if TEST_OCR_ON_IMAGE:
        print(f"\n--- OCR on embedded images ({LIBRARY}) ---")
        embedded = get_embedded_images(PDF_PATH)
        print(f"Embedded images in PDF: {len(embedded)}")
        if embedded:
            t1       = time.perf_counter()
            img_text = ocr_images_with(LIBRARY, embedded)
            img_t    = time.perf_counter() - t1
            print(f"[IMG]  Time         : {img_t:.2f}s")
            print(f"[IMG]  Chars extract: {len(img_text)}")
            print(f"\n--- Image OCR preview (first 300 chars) ---")
            print(img_text[:300])
        else:
            print("No embedded images found in this PDF.")
