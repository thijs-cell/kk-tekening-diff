"""FastAPI applicatie voor het vergelijken van PDF bouwtekeningen."""

import gc
import logging
import os
from typing import Any

from fastapi import FastAPI, File, Query, UploadFile
from fastapi.responses import HTMLResponse, JSONResponse
from pdf2image import convert_from_bytes, pdfinfo_from_bytes

from .compare import compare_page
from .config import ANTHROPIC_API_KEY, DPI, ENABLE_AI_INTERPRETATION, MAX_FILE_SIZE_MB, SENSITIVITY

# Logging configureren
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

# Startup check: toon of de API key geladen is
_raw_key = os.environ.get("ANTHROPIC_API_KEY")
if _raw_key:
    logger.info("ANTHROPIC_API_KEY geladen: %s...", _raw_key[:10])
else:
    logger.warning("ANTHROPIC_API_KEY NIET gevonden in environment variables")
logger.info("ENABLE_AI_INTERPRETATION=%s, ANTHROPIC_API_KEY config=%s...",
            ENABLE_AI_INTERPRETATION, ANTHROPIC_API_KEY[:10] if ANTHROPIC_API_KEY else "(leeg)")

app = FastAPI(
    title="K&K Tekening Diff",
    description="Vergelijk PDF bouwtekeningen en detecteer wijzigingen per pagina.",
    version="1.0.0",
)

# PDF magic bytes: %PDF
PDF_MAGIC = b"%PDF"

# Laatste vergelijkingsresultaat opslaan voor /results endpoint
_last_result: dict[str, Any] | None = None


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

    # Resultaat opslaan voor /results endpoint
    global _last_result
    _last_result = {
        "status": "success",
        "old_pages": old_count,
        "new_pages": new_count,
        "old_filename": old_filename,
        "new_filename": new_filename,
        "comparisons": comparisons,
    }

    return JSONResponse(content=_last_result)


@app.get("/results", response_class=HTMLResponse)
async def results_page() -> HTMLResponse:
    """Toon de laatste vergelijking als HTML pagina met overlay afbeeldingen."""
    if _last_result is None:
        return HTMLResponse(
            content="<html><body><h1>Geen resultaten</h1>"
            "<p>Er is nog geen vergelijking uitgevoerd. "
            "Gebruik POST /compare om twee PDF's te vergelijken.</p></body></html>",
            status_code=200,
        )

    old_name = _last_result.get("old_filename", "onbekend")
    new_name = _last_result.get("new_filename", "onbekend")
    old_pages = _last_result.get("old_pages", 0)
    new_pages = _last_result.get("new_pages", 0)
    comparisons = _last_result.get("comparisons", [])

    # HTML opbouwen
    cards_html = ""
    for comp in comparisons:
        page = comp["page"]
        status = comp["status"]
        pct = comp["change_percentage"]
        overlay_b64 = comp.get("overlay_image")

        # Kleur op basis van status
        if status == "no_changes":
            badge_color = "#27ae60"
            badge_text = "Geen wijzigingen"
        elif status == "new_page":
            badge_color = "#2980b9"
            badge_text = "Nieuwe pagina"
        elif status == "removed_page":
            badge_color = "#8e44ad"
            badge_text = "Verwijderde pagina"
        elif status == "alignment_failed":
            badge_color = "#e67e22"
            badge_text = "Uitlijning mislukt"
        elif status == "error":
            badge_color = "#c0392b"
            badge_text = "Fout"
        else:
            badge_color = "#e74c3c" if pct > 1 else "#f39c12"
            badge_text = f"{pct:.2f}% gewijzigd"

        img_html = ""
        if overlay_b64:
            img_html = (
                f'<img src="data:image/png;base64,{overlay_b64}" '
                f'style="width:100%;border:1px solid #ddd;border-radius:4px;" '
                f'alt="Pagina {page} overlay"/>'
            )
        elif status == "no_changes":
            img_html = (
                '<div style="padding:40px;text-align:center;color:#888;'
                'background:#f9f9f9;border-radius:4px;">'
                "Geen overlay — pagina is ongewijzigd</div>"
            )
        elif status == "removed_page":
            img_html = (
                '<div style="padding:40px;text-align:center;color:#8e44ad;'
                'background:#f5eef8;border-radius:4px;">'
                "Pagina bestaat niet meer in de nieuwe versie</div>"
            )
        else:
            img_html = (
                '<div style="padding:40px;text-align:center;color:#888;'
                'background:#f9f9f9;border-radius:4px;">'
                "Geen overlay beschikbaar</div>"
            )

        # Interpretaties HTML
        interp_html = ""
        interpretations = comp.get("interpretations", [])
        if interpretations:
            interp_items = ""
            for interp in interpretations:
                cls = interp.get("classification", "INFORMATIEF")
                desc = interp.get("description", "")
                crop_b64 = interp.get("crop_image", "")

                if cls == "KRITIEK":
                    badge_cls = (
                        "background:#c0392b;color:#fff;padding:2px 8px;"
                        "border-radius:8px;font-size:12px;font-weight:600;"
                    )
                else:
                    badge_cls = (
                        "background:#95a5a6;color:#fff;padding:2px 8px;"
                        "border-radius:8px;font-size:12px;font-weight:600;"
                    )

                crop_html = ""
                if crop_b64:
                    crop_html = (
                        f'<img src="data:image/png;base64,{crop_b64}" '
                        f'style="height:80px;border:1px solid #ddd;border-radius:4px;'
                        f'cursor:pointer;flex-shrink:0;" '
                        f'onclick="window.open(this.src)" '
                        f'alt="Crop"/>'
                    )

                interp_items += f"""
                <div style="display:flex;align-items:flex-start;gap:12px;
                    padding:8px 0;border-bottom:1px solid #eee;">
                    {crop_html}
                    <div style="flex:1;">
                        <span style="{badge_cls}">{cls}</span>
                        <span style="margin-left:8px;">{desc}</span>
                    </div>
                </div>"""

            interp_html = f"""
            <div style="margin-top:12px;padding:12px;background:#fff;
                border:1px solid #ddd;border-radius:6px;">
                <h3 style="margin:0 0 8px 0;font-size:16px;">AI Interpretatie</h3>
                {interp_items}
            </div>"""

        cards_html += f"""
        <div style="margin-bottom:32px;">
            <div style="display:flex;align-items:center;gap:12px;margin-bottom:8px;">
                <h2 style="margin:0;font-size:20px;">Pagina {page}</h2>
                <span style="background:{badge_color};color:#fff;padding:4px 12px;
                    border-radius:12px;font-size:14px;font-weight:500;">{badge_text}</span>
            </div>
            {img_html}
            {interp_html}
        </div>"""

    html = f"""<!DOCTYPE html>
<html lang="nl">
<head>
    <meta charset="utf-8"/>
    <meta name="viewport" content="width=device-width, initial-scale=1"/>
    <title>K&amp;K Tekening Diff — Resultaten</title>
    <style>
        body {{ font-family: -apple-system, system-ui, sans-serif; margin: 0;
               padding: 24px; background: #f4f4f4; color: #333; }}
        .container {{ max-width: 1200px; margin: 0 auto; }}
        h1 {{ font-size: 24px; margin-bottom: 4px; }}
        .meta {{ color: #666; font-size: 14px; margin-bottom: 24px; }}
    </style>
</head>
<body>
    <div class="container">
        <h1>Vergelijkingsresultaat</h1>
        <p class="meta">
            Oud: <strong>{old_name}</strong> ({old_pages} pagina's) &rarr;
            Nieuw: <strong>{new_name}</strong> ({new_pages} pagina's)
        </p>
        {cards_html}
    </div>
</body>
</html>"""

    return HTMLResponse(content=html)
