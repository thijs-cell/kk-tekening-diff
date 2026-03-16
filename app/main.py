"""FastAPI applicatie voor het vergelijken van PDF bouwtekeningen."""

import logging
import tempfile
from pathlib import Path
from typing import Any

from fastapi import FastAPI, File, Query, UploadFile
from fastapi.responses import JSONResponse
from pdf2image import convert_from_path
from PIL import Image

from .compare import compare_page
from .config import DPI, MAX_FILE_SIZE_MB, SENSITIVITY

# Logging configureren
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="K&K Tekening Diff",
    description="Vergelijk PDF bouwtekeningen en detecteer wijzigingen per pagina.",
    version="1.0.0",
)

# PDF magic bytes: %PDF
PDF_MAGIC = b"%PDF"


def _validate_pdf(content: bytes, filename: str) -> str | None:
    """
    Valideer of het bestand een PDF is.

    Returns:
        Foutmelding string bij fout, None als alles ok is.
    """
    if not content.startswith(PDF_MAGIC):
        return f"Bestand '{filename}' is geen geldig PDF bestand."

    max_bytes = MAX_FILE_SIZE_MB * 1024 * 1024
    if len(content) > max_bytes:
        return (
            f"Bestand '{filename}' is te groot "
            f"({len(content) / 1024 / 1024:.1f} MB, maximum is {MAX_FILE_SIZE_MB} MB)."
        )

    return None


def _pdf_to_images(pdf_path: Path, dpi: int) -> list[Image.Image]:
    """
    Converteer alle pagina's van een PDF naar PIL Images.

    Verwerkt pagina voor pagina om geheugen te besparen.
    """
    images: list[Image.Image] = []

    # Eerst het totaal aantal pagina's bepalen via de eerste pagina
    first = convert_from_path(str(pdf_path), dpi=dpi, first_page=1, last_page=1)
    if not first:
        return images

    # Totaal aantal pagina's ophalen door te tellen
    # pdf2image kan info geven via pdfinfo, maar we itereren pagina voor pagina
    page_num = 1
    while True:
        try:
            page_images = convert_from_path(
                str(pdf_path),
                dpi=dpi,
                first_page=page_num,
                last_page=page_num,
            )
            if not page_images:
                break
            images.append(page_images[0])
            page_num += 1
        except Exception:
            # Geen pagina meer beschikbaar
            break

    return images


@app.get("/")
async def health_check() -> dict[str, str]:
    """Health check endpoint."""
    return {"status": "online", "version": "1.0.0"}


@app.post("/compare")
async def compare_pdfs(
    old_pdf: UploadFile = File(..., description="Oude versie van de PDF tekening"),
    new_pdf: UploadFile = File(..., description="Nieuwe versie van de PDF tekening"),
    dpi: int = Query(default=DPI, ge=72, le=600, description="Render DPI"),
    sensitivity: int = Query(
        default=SENSITIVITY,
        ge=1,
        le=255,
        description="Verschildrempel (lager = gevoeliger)",
    ),
) -> JSONResponse:
    """
    Vergelijk twee PDF bouwtekeningen en detecteer wijzigingen per pagina.

    Ontvangt twee PDF bestanden via multipart/form-data en retourneert
    per pagina de gedetecteerde wijzigingen als diff en overlay afbeeldingen.
    """
    # Bestanden inlezen
    old_content = await old_pdf.read()
    new_content = await new_pdf.read()

    # Validatie
    old_filename = old_pdf.filename or "old_pdf"
    new_filename = new_pdf.filename or "new_pdf"

    error = _validate_pdf(old_content, old_filename)
    if error:
        return JSONResponse(status_code=400, content={"status": "error", "detail": error})

    error = _validate_pdf(new_content, new_filename)
    if error:
        return JSONResponse(status_code=400, content={"status": "error", "detail": error})

    logger.info(
        "Vergelijking gestart: '%s' (%d KB) vs '%s' (%d KB), DPI=%d, sensitivity=%d",
        old_filename,
        len(old_content) // 1024,
        new_filename,
        len(new_content) // 1024,
        dpi,
        sensitivity,
    )

    # PDF's opslaan als tijdelijke bestanden en converteren naar afbeeldingen
    old_images: list[Image.Image] = []
    new_images: list[Image.Image] = []

    try:
        with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as tmp_old:
            tmp_old.write(old_content)
            tmp_old_path = Path(tmp_old.name)

        with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as tmp_new:
            tmp_new.write(new_content)
            tmp_new_path = Path(tmp_new.name)

        # Geheugen vrijgeven na wegschrijven
        del old_content, new_content

        logger.info("PDF's omzetten naar afbeeldingen...")
        old_images = _pdf_to_images(tmp_old_path, dpi)
        new_images = _pdf_to_images(tmp_new_path, dpi)

        logger.info(
            "Pagina's geladen: oud=%d, nieuw=%d", len(old_images), len(new_images)
        )

    except Exception as e:
        logger.error("Fout bij het laden van PDF's: %s", str(e))
        return JSONResponse(
            status_code=500,
            content={
                "status": "error",
                "detail": f"Fout bij het verwerken van de PDF bestanden: {str(e)}",
            },
        )
    finally:
        # Tijdelijke bestanden opruimen
        try:
            tmp_old_path.unlink(missing_ok=True)
        except Exception:
            pass
        try:
            tmp_new_path.unlink(missing_ok=True)
        except Exception:
            pass

    old_count = len(old_images)
    new_count = len(new_images)
    max_pages = max(old_count, new_count)

    comparisons: list[dict[str, Any]] = []

    # Pagina's sequentieel vergelijken
    for i in range(max_pages):
        page_num = i + 1
        old_img = old_images[i] if i < old_count else None
        new_img = new_images[i] if i < new_count else None

        logger.info("Pagina %d/%d verwerken...", page_num, max_pages)

        try:
            result = compare_page(old_img, new_img, page_num, sensitivity)
            comparisons.append(result)
        except Exception as e:
            logger.error("Fout bij pagina %d: %s", page_num, str(e))
            comparisons.append(
                {
                    "page": page_num,
                    "status": "error",
                    "changes_detected": False,
                    "change_percentage": 0.0,
                    "diff_image": None,
                    "overlay_image": None,
                    "error": f"Verwerkingsfout op pagina {page_num}: {str(e)}",
                }
            )

        # Verwerkte afbeeldingen vrijgeven
        if old_img is not None:
            old_img.close()
        if new_img is not None and (i >= old_count or old_img is not new_img):
            new_img.close()

    logger.info("Vergelijking voltooid: %d pagina's verwerkt", max_pages)

    return JSONResponse(
        content={
            "status": "success",
            "old_pages": old_count,
            "new_pages": new_count,
            "comparisons": comparisons,
        }
    )
