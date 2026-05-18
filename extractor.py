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

import time, os, io, logging
from importlib.util import find_spec
from contextlib import redirect_stdout, redirect_stderr
from shutil import which
import fitz   # pymupdf - used for page count + image extraction

PDF_PATH         = "/home/riyap/pdf_extractor/uploads/4th_sample-report_english_final.pdf"  # <-- your PDF
LIBRARY          = "paddleocr"    # <-- CHANGE THIS to switch library
TEST_OCR_ON_IMAGE = True          # <-- True = also OCR the embedded images

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")
os.environ["FLAGS_use_mkldnn"]      = "0"
os.environ["PADDLE_DISABLE_MKLDNN"] = "1"


class MissingDependencyError(RuntimeError):
    pass


LIBRARY_REQUIREMENTS = {
    "pymupdf": {
        "python": [("fitz", "pymupdf")],
    },
    "pdfplumber": {
        "python": [("pdfplumber", "pdfplumber")],
    },
    "paddleocr": {
        "python": [
            ("numpy", "numpy"),
            ("pdf2image", "pdf2image"),
            ("paddleocr", "paddleocr"),
            ("paddle", "paddlepaddle"),
        ],
        "system": [("pdftoppm", "poppler-utils")],
    },
    "rapidocr": {
        "python": [
            ("numpy", "numpy"),
            ("pdf2image", "pdf2image"),
            ("rapidocr_onnxruntime", "rapidocr-onnxruntime"),
        ],
        "system": [("pdftoppm", "poppler-utils")],
    },
    "unstructured": {
        "python": [("unstructured.partition.pdf", "unstructured[pdf]")],
    },
    "docling": {
        "python": [("docling.document_converter", "docling")],
    },
    "florence2": {
        "python": [
            ("torch", "torch"),
            ("transformers", "transformers"),
            ("timm", "timm"),
            ("einops", "einops"),
            ("pdf2image", "pdf2image"),
        ],
        "system": [("pdftoppm", "poppler-utils")],
    },
}


def check_library_ready(library, image_ocr=False):
    reqs = LIBRARY_REQUIREMENTS.get(library, {})
    missing_py = [
        package for module, package in reqs.get("python", [])
        if find_spec(module) is None
    ]
    missing_sys = [
        package for command, package in reqs.get("system", [])
        if which(command) is None
    ]

    if image_ocr and library in ("pymupdf", "pdfplumber", "docling"):
        if find_spec("pytesseract") is None:
            missing_py.append("pytesseract")
        if which("tesseract") is None:
            missing_sys.append("tesseract-ocr")

    if missing_py or missing_sys:
        parts = []
        if missing_py:
            parts.append("Python: pip install " + " ".join(dict.fromkeys(missing_py)))
        if missing_sys:
            parts.append("System: sudo apt install " + " ".join(dict.fromkeys(missing_sys)))
        raise MissingDependencyError("Missing dependencies for " + library + ". " + " | ".join(parts))


def florence_model_kwargs(dtype):
    kwargs = {"torch_dtype": dtype, "trust_remote_code": True}
    if find_spec("flash_attn") is None:
        kwargs["attn_implementation"] = "eager"
    return kwargs


def format_florence_error(exc):
    msg = str(exc)
    if "flash_attn" in msg or "flash-attn" in msg or "flash attention" in msg.lower():
        return (
            msg
            + " | Tip: flash_attn is optional but may be required by some Florence-2/"
            + "transformers combinations. Try upgrading transformers or install with: "
            + "pip install flash-attn --no-build-isolation. For CPU OCR, paddleocr or "
            + "rapidocr are usually easier."
        )
    return msg


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


def load_paddleocr_model():
    from paddleocr import PaddleOCR
    for logger_name in ("paddle", "paddleocr", "paddlex", "ppocr"):
        logging.getLogger(logger_name).setLevel(logging.ERROR)
    kwargs = {
        "use_doc_orientation_classify": False,
        "use_doc_unwarping": False,
        "use_textline_orientation": False,
        "enable_mkldnn": False,
        "lang": "en",
        "device": "cpu",
    }
    with redirect_stdout(io.StringIO()), redirect_stderr(io.StringIO()):
        return PaddleOCR(**kwargs)


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
    ocr   = load_paddleocr_model()
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
    import transformers
    from transformers import AutoModelForCausalLM, AutoModelForVision2Seq, AutoProcessor
    from pdf2image import convert_from_path

    if not hasattr(transformers, "AutoModelForImageTextToText"):
        transformers.AutoModelForImageTextToText = AutoModelForVision2Seq

    MODEL_ID  = "microsoft/Florence-2-large"
    device    = "cuda" if torch.cuda.is_available() else "cpu"
    dtype     = torch.float16 if device == "cuda" else torch.float32

    print(f"  Loading Florence-2 on {device} ...")
    try:
        processor = AutoProcessor.from_pretrained(MODEL_ID, trust_remote_code=True)
        model     = AutoModelForCausalLM.from_pretrained(
                        MODEL_ID, **florence_model_kwargs(dtype)).to(device)
    except Exception as e:
        raise RuntimeError(format_florence_error(e)) from e

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
        ocr = load_paddleocr_model()
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
        import transformers
        from transformers import AutoModelForCausalLM, AutoModelForVision2Seq, AutoProcessor

        if not hasattr(transformers, "AutoModelForImageTextToText"):
            transformers.AutoModelForImageTextToText = AutoModelForVision2Seq

        MODEL_ID  = "microsoft/Florence-2-large"
        device    = "cuda" if torch.cuda.is_available() else "cpu"
        dtype     = torch.float16 if device == "cuda" else torch.float32
        try:
            processor = AutoProcessor.from_pretrained(MODEL_ID, trust_remote_code=True)
            model     = AutoModelForCausalLM.from_pretrained(
                            MODEL_ID, **florence_model_kwargs(dtype)).to(device)
        except Exception as e:
            raise RuntimeError(format_florence_error(e)) from e
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
            from pytesseract import TesseractNotFoundError
            for img in images:
                try:
                    text += pytesseract.image_to_string(img) + "\n"
                except TesseractNotFoundError:
                    text = (
                        "[Tesseract OCR is not installed or not in PATH. "
                        "On Ubuntu/Linux run: sudo apt install tesseract-ocr]"
                    )
                    break
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
    try:
        check_library_ready(LIBRARY)
        t0           = time.perf_counter()
        text, n_imgs = runners[LIBRARY](PDF_PATH)
        elapsed      = time.perf_counter() - t0

        print(f"[PDF]  Time         : {elapsed:.2f}s")
        print(f"[PDF]  Chars extract: {len(text)}")
        print(f"[PDF]  Images found : {n_imgs}")
        print(f"\n--- Text preview (first 500 chars) ---")
        print(text[:500])
    except MissingDependencyError as e:
        print(f"[PDF]  Error        : {e}")
        raise SystemExit(1)

    # --- Part 2: OCR on embedded images ---
    if TEST_OCR_ON_IMAGE:
        print(f"\n--- OCR on embedded images ({LIBRARY}) ---")
        embedded = get_embedded_images(PDF_PATH)
        print(f"Embedded images in PDF: {len(embedded)}")
        if embedded:
            try:
                check_library_ready(LIBRARY, image_ocr=True)
                t1       = time.perf_counter()
                img_text = ocr_images_with(LIBRARY, embedded)
                img_t    = time.perf_counter() - t1
                print(f"[IMG]  Time         : {img_t:.2f}s")
                print(f"[IMG]  Chars extract: {len(img_text)}")
                print(f"\n--- Image OCR preview (first 300 chars) ---")
                print(img_text[:300])
            except MissingDependencyError as e:
                print(f"[IMG]  Error        : {e}")
        else:
            print("No embedded images found in this PDF.")
