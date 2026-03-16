"""FastAPI applicatie voor het vergelijken van PDF bouwtekeningen."""

import gc
import logging
from typing import Any

from fastapi import FastAPI, File, Query, UploadFile
from fastapi.responses import JSONResponse
from pdf2image import convert_from_bytes, pdfinfo_from_bytes

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


def _get_page_count(pdf_bytes: bytes) -> int:
    """Tel het aantal pagina's in een PDF zonder ze te renderen."""
    info = pdfinfo_from_bytes(pdf_bytes)
    return info.get("Pages", 0)


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

    Verwerkt per pagina: converteer, vergelijk, sla op, ruim geheugen op.
    """
    # Bestanden inlezen
    old_bytes = await old_pdf.read()
    new_bytes = await new_pdf.read()

    # Validatie
    old_filename = old_pdf.filename or "old_pdf"
    new_filename = new_pdf.filename or "new_pdf"

    error = _validate_pdf(old_bytes, old_filename)
    if error:
        return JSONResponse(status_code=400, content={"status": "error", "detail": error})

    error = _validate_pdf(new_bytes, new_filename)
    if error:
        return JSONResponse(status_code=400, content={"status": "error", "detail": error})

    logger.info(
        "Vergelijking gestart: '%s' (%d KB) vs '%s' (%d KB), DPI=%d, sensitivity=%d",
        old_filename,
        len(old_bytes) // 1024,
        new_filename,
        len(new_bytes) // 1024,
        dpi,
        sensitivity,
    )

    # Aantal pagina's tellen ZONDER te renderen
    try:
        old_count = _get_page_count(old_bytes)
        new_count = _get_page_count(new_bytes)
    except Exception as e:
        logger.error("Fout bij het lezen van PDF info: %s", str(e))
        return JSONResponse(
            status_code=500,
            content={
                "status": "error",
                "detail": f"Kan PDF info niet lezen: {str(e)}",
            },
        )

    max_pages = max(old_count, new_count)
    logger.info("Pagina's geteld: oud=%d, nieuw=%d", old_count, new_count)

    comparisons: list[dict[str, Any]] = []

    # Per pagina: converteer, vergelijk, ruim op
    for i in range(1, max_pages + 1):
        logger.info("[pagina %d/%d] Start verwerking...", i, max_pages)

        old_img = None
        new_img = None

        try:
            # Alleen converteren als de pagina bestaat in die PDF
            if i <= old_count:
                old_img = convert_from_bytes(
                    old_bytes, dpi=dpi, first_page=i, last_page=i
                )[0]

            if i <= new_count:
                new_img = convert_from_bytes(
                    new_bytes, dpi=dpi, first_page=i, last_page=i
                )[0]

            logger.info("[pagina %d/%d] Geconverteerd, start vergelijking...", i, max_pages)

            result = compare_page(old_img, new_img, i, sensitivity)
            comparisons.append(result)

            logger.info(
                "[pagina %d/%d] Klaar — status=%s, wijziging=%.2f%%",
                i, max_pages, result["status"], result["change_percentage"],
            )

        except Exception as e:
            logger.error("[pagina %d/%d] FOUT: %s", i, max_pages, str(e))
            comparisons.append(
                {
                    "page": i,
                    "status": "error",
                    "changes_detected": False,
                    "change_percentage": 0.0,
                    "diff_image": None,
                    "overlay_image": None,
                    "error": f"Verwerkingsfout op pagina {i}: {str(e)}",
                }
            )

        finally:
            # Afbeeldingen direct vrijgeven
            if old_img is not None:
                old_img.close()
                del old_img
            if new_img is not None:
                new_img.close()
                del new_img
            gc.collect()

    # PDF bytes vrijgeven
    del old_bytes, new_bytes
    gc.collect()

    logger.info("Vergelijking voltooid: %d pagina's verwerkt", max_pages)

    return JSONResponse(
        content={
            "status": "success",
            "old_pages": old_count,
            "new_pages": new_count,
            "comparisons": comparisons,
        }
    )
