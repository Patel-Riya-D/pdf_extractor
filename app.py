"""
PDF Extraction Benchmark — Streamlit UI
Run: streamlit run app.py
"""

import streamlit as st
import time, os, io, logging
from pathlib import Path

# ── Logger setup ──────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
    handlers=[logging.FileHandler("benchmark.log"), logging.StreamHandler()],
)
log = logging.getLogger("pdf_bench")

os.environ["FLAGS_use_mkldnn"]      = "0"
os.environ["PADDLE_DISABLE_MKLDNN"] = "1"

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(page_title="PDF Extraction Benchmark", layout="wide", page_icon="📄")
st.title("📄 PDF Extraction Benchmark")
st.caption("Compare libraries for text extraction + image OCR from PDFs")

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.header("⚙️ Settings")
    uploaded = st.file_uploader("Upload PDF", type=["pdf"])
    library  = st.selectbox("Library", [
        "pymupdf", "pdfplumber", "paddleocr",
        "rapidocr", "unstructured", "docling", "florence2"
    ])
    run_img_ocr = st.toggle("OCR on embedded images", value=True)
    st.divider()
    st.markdown("""
**Install libs:**
```
pip install pymupdf pdfplumber
pip install paddlepaddle paddleocr
pip install rapidocr-onnxruntime
pip install unstructured[pdf]
pip install docling
pip install transformers timm einops
pip install streamlit pdf2image
```
""")

# ── Helpers ───────────────────────────────────────────────────────────────────

def get_page_count(path):
    import fitz
    return len(fitz.open(path))

def get_embedded_images(path):
    import fitz
    from PIL import Image
    doc, imgs = fitz.open(path), []
    for page in doc:
        for img in page.get_images(full=True):
            raw = doc.extract_image(img[0])
            imgs.append(Image.open(io.BytesIO(raw["image"])).convert("RGB"))
    log.info(f"Extracted {len(imgs)} embedded images from PDF")
    return imgs

# ── Runners ───────────────────────────────────────────────────────────────────

def run_pymupdf(path):
    import fitz
    log.info("pymupdf: starting")
    doc, text, n = fitz.open(path), "", 0
    for page in doc:
        text += page.get_text()
        n    += len(page.get_images(full=True))
    log.info(f"pymupdf: done — {len(text)} chars, {n} images")
    return text, n

def run_pdfplumber(path):
    import pdfplumber
    log.info("pdfplumber: starting")
    text, imgs = "", []
    with pdfplumber.open(path) as pdf:
        for page in pdf.pages:
            text += page.extract_text() or ""
            imgs += page.images
    log.info(f"pdfplumber: done — {len(text)} chars, {len(imgs)} images")
    return text, len(imgs)

def run_paddleocr(path):
    import numpy as np
    from pdf2image import convert_from_path
    from paddleocr import PaddleOCR
    log.info("paddleocr: loading model")
    ocr   = PaddleOCR(use_textline_orientation=True, lang="en", device="cpu")
    pages = convert_from_path(path, dpi=200)
    log.info(f"paddleocr: running on {len(pages)} pages")
    text  = ""
    for i, page in enumerate(pages, 1):
        result = list(ocr.predict(np.array(page)))
        if result:
            r = result[0]
            if isinstance(r, dict):   texts = r.get("rec_texts") or r.get("txts") or []
            elif hasattr(r,"rec_texts"): texts = r.rec_texts or []
            elif hasattr(r,"txts"):   texts = r.txts or []
            else: texts = [l[1][0] for l in (r or []) if l]
            text += " ".join(texts) + "\n"
        log.info(f"paddleocr: page {i}/{len(pages)} done")
    return text, 0

def run_rapidocr(path):
    from rapidocr_onnxruntime import RapidOCR
    from pdf2image import convert_from_path
    import numpy as np
    log.info("rapidocr: starting")
    ocr   = RapidOCR()
    pages = convert_from_path(path, dpi=200)
    text  = ""
    for i, page in enumerate(pages, 1):
        result, _ = ocr(np.array(page))
        if result:
            text += " ".join([l[1] for l in result]) + "\n"
        log.info(f"rapidocr: page {i}/{len(pages)} done")
    return text, 0

def run_unstructured(path):
    from unstructured.partition.pdf import partition_pdf
    log.info("unstructured: partitioning")
    elements = partition_pdf(path)
    text     = "\n".join([str(e) for e in elements])
    n_imgs   = len([e for e in elements if "Image" in type(e).__name__])
    log.info(f"unstructured: {len(elements)} elements, {n_imgs} images")
    return text, n_imgs

def run_docling(path):
    from docling.document_converter import DocumentConverter
    log.info("docling: converting")
    result = DocumentConverter().convert(path)
    doc    = result.document
    text   = doc.export_to_markdown()
    n_imgs = len([i for i, _ in doc.iterate_items() if "Picture" in type(i).__name__])
    log.info(f"docling: done — {len(text)} chars, {n_imgs} pictures")
    return text, n_imgs

def run_florence2(path):
    import torch
    from transformers import AutoProcessor, AutoModelForCausalLM
    from pdf2image import convert_from_path
    MODEL_ID = "microsoft/Florence-2-large"
    device   = "cuda" if torch.cuda.is_available() else "cpu"
    dtype    = torch.float16 if device == "cuda" else torch.float32
    log.info(f"florence2: loading model on {device}")
    processor = AutoProcessor.from_pretrained(MODEL_ID, trust_remote_code=True)
    model     = AutoModelForCausalLM.from_pretrained(
                    MODEL_ID, torch_dtype=dtype, trust_remote_code=True).to(device)
    pages, text, prompt = convert_from_path(path, dpi=150), "", "<OCR>"
    for i, page in enumerate(pages, 1):
        inputs = processor(text=prompt, images=page, return_tensors="pt").to(device, dtype)
        ids    = model.generate(input_ids=inputs["input_ids"],
                                pixel_values=inputs["pixel_values"],
                                max_new_tokens=1024, num_beams=3)
        out    = processor.batch_decode(ids, skip_special_tokens=False)[0]
        text  += processor.post_process_generation(
                    out, task=prompt, image_size=page.size)[prompt] + "\n"
        log.info(f"florence2: page {i}/{len(pages)} done")
    return text, 0

def ocr_images_with(library, images):
    if not images:
        return ""
    text = ""
    log.info(f"Image OCR: running {library} on {len(images)} image(s)")

    if library == "paddleocr":
        import numpy as np
        from paddleocr import PaddleOCR
        ocr = PaddleOCR(use_textline_orientation=True, lang="en", device="cpu")
        for i, img in enumerate(images, 1):
            result = list(ocr.predict(np.array(img)))
            if result:
                r = result[0]
                if isinstance(r, dict):      ts = r.get("rec_texts") or []
                elif hasattr(r,"rec_texts"): ts = r.rec_texts or []
                else:                        ts = []
                text += " ".join(ts) + "\n"
            log.info(f"Image OCR paddleocr: image {i}/{len(images)} done")

    elif library == "rapidocr":
        from rapidocr_onnxruntime import RapidOCR
        import numpy as np
        ocr = RapidOCR()
        for i, img in enumerate(images, 1):
            result, _ = ocr(np.array(img))
            if result:
                text += " ".join([l[1] for l in result]) + "\n"
            log.info(f"Image OCR rapidocr: image {i}/{len(images)} done")

    elif library == "florence2":
        import torch
        from transformers import AutoProcessor, AutoModelForCausalLM
        MODEL_ID = "microsoft/Florence-2-large"
        device   = "cuda" if torch.cuda.is_available() else "cpu"
        dtype    = torch.float16 if device == "cuda" else torch.float32
        processor = AutoProcessor.from_pretrained(MODEL_ID, trust_remote_code=True)
        model     = AutoModelForCausalLM.from_pretrained(
                        MODEL_ID, torch_dtype=dtype, trust_remote_code=True).to(device)
        prompt = "<OCR>"
        for i, img in enumerate(images, 1):
            inputs = processor(text=prompt, images=img, return_tensors="pt").to(device, dtype)
            ids    = model.generate(input_ids=inputs["input_ids"],
                                    pixel_values=inputs["pixel_values"],
                                    max_new_tokens=512, num_beams=3)
            out    = processor.batch_decode(ids, skip_special_tokens=False)[0]
            text  += processor.post_process_generation(
                        out, task=prompt, image_size=img.size)[prompt] + "\n"
            log.info(f"Image OCR florence2: image {i}/{len(images)} done")

    elif library == "unstructured":
        text = "unstructured handles image OCR automatically during partition_pdf."

    else:
        try:
            import pytesseract
            for i, img in enumerate(images, 1):
                text += pytesseract.image_to_string(img) + "\n"
                log.info(f"Image OCR pytesseract: image {i}/{len(images)} done")
        except ImportError:
            text = "pytesseract not installed. Run: pip install pytesseract"

    log.info(f"Image OCR done — {len(text)} chars extracted")
    return text

runners = {
    "pymupdf": run_pymupdf, "pdfplumber": run_pdfplumber,
    "paddleocr": run_paddleocr, "rapidocr": run_rapidocr,
    "unstructured": run_unstructured, "docling": run_docling,
    "florence2": run_florence2,
}

# ── Main UI ───────────────────────────────────────────────────────────────────

if uploaded is None:
    st.info("Upload a PDF from the sidebar to get started.")
    st.stop()

# Save upload to temp file
tmp_path = Path("_uploaded.pdf")
tmp_path.write_bytes(uploaded.read())
page_count = get_page_count(str(tmp_path))

st.markdown(f"**File:** `{uploaded.name}`  |  **Pages:** `{page_count}`  |  **Library:** `{library}`")
st.divider()

if st.button("▶ Run Benchmark", type="primary", use_container_width=True):

    log.info(f"=== Benchmark start | library={library} | pdf={uploaded.name} | pages={page_count} ===")

    # ── Part 1: PDF text extraction ───────────────────────────────────────────
    st.subheader("1️⃣ PDF Text Extraction")
    status = st.status(f"Running {library}...", expanded=True)

    with status:
        t0 = time.perf_counter()
        try:
            pdf_text, n_imgs = runners[library](str(tmp_path))
            elapsed = time.perf_counter() - t0
            status.update(label=f"✅ Done in {elapsed:.2f}s", state="complete")
            log.info(f"PDF extraction complete: {elapsed:.2f}s | {len(pdf_text)} chars | {n_imgs} images")
        except Exception as e:
            elapsed = time.perf_counter() - t0
            status.update(label=f"❌ Error: {e}", state="error")
            log.error(f"PDF extraction failed: {e}")
            st.stop()

    # Metrics row
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("⏱ Time",          f"{elapsed:.2f}s")
    c2.metric("📄 Pages",         page_count)
    c3.metric("🔤 Chars extracted", f"{len(pdf_text):,}")
    c4.metric("🖼 Images found",  n_imgs)

    # Full extracted text
    with st.expander("📝 Full Extracted Text", expanded=True):
        st.text_area("", pdf_text, height=400, label_visibility="collapsed")
        st.download_button("⬇ Download extracted text",
                           pdf_text, file_name=f"{library}_extracted.txt")

    st.divider()

    # ── Part 2: OCR on embedded images ───────────────────────────────────────
    st.subheader("2️⃣ OCR on Embedded Images")

    if not run_img_ocr:
        st.info("Image OCR is disabled. Toggle it on in the sidebar.")
    else:
        embedded_imgs = get_embedded_images(str(tmp_path))

        if not embedded_imgs:
            st.warning("No embedded images found in this PDF.")
            log.info("No embedded images found")
        else:
            # Show the embedded images
            with st.expander(f"🖼 Embedded Images ({len(embedded_imgs)} found)", expanded=True):
                cols = st.columns(min(len(embedded_imgs), 4))
                for i, img in enumerate(embedded_imgs):
                    cols[i % 4].image(img, caption=f"Image {i+1}",
                                      use_container_width=True)

            # Run OCR on them
            img_status = st.status(f"Running image OCR with {library}...", expanded=True)
            with img_status:
                t1 = time.perf_counter()
                try:
                    img_text  = ocr_images_with(library, embedded_imgs)
                    img_elapsed = time.perf_counter() - t1
                    img_status.update(label=f"✅ Done in {img_elapsed:.2f}s", state="complete")
                    log.info(f"Image OCR complete: {img_elapsed:.2f}s | {len(img_text)} chars")
                except Exception as e:
                    img_elapsed = time.perf_counter() - t1
                    img_status.update(label=f"❌ Error: {e}", state="error")
                    log.error(f"Image OCR failed: {e}")
                    img_text = ""

            ic1, ic2 = st.columns(2)
            ic1.metric("⏱ Image OCR Time",    f"{img_elapsed:.2f}s")
            ic2.metric("🔤 Chars from Images", f"{len(img_text):,}")

            with st.expander("📝 Full Image OCR Text", expanded=True):
                st.text_area("", img_text, height=300, label_visibility="collapsed")
                st.download_button("⬇ Download image OCR text",
                                   img_text, file_name=f"{library}_image_ocr.txt")

    st.divider()

    # ── Part 3: Log viewer ────────────────────────────────────────────────────
    st.subheader("3️⃣ Logs")
    with st.expander("📋 View full log", expanded=False):
        try:
            log_text = Path("benchmark.log").read_text()
            st.code(log_text, language="log")
        except FileNotFoundError:
            st.info("No log file yet.")
    log.info("=== Benchmark end ===")