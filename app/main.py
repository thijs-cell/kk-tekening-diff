"""FastAPI applicatie voor het vergelijken van PDF bouwtekeningen."""

import gc
import io
import logging
import os
from typing import Any

from fastapi import FastAPI, File, Query, UploadFile
from fastapi.responses import HTMLResponse, JSONResponse, Response
from pdf2image import convert_from_bytes, pdfinfo_from_bytes

from .compare import compare_page, compare_page_raw
from .config import (
    ANTHROPIC_API_KEY,
    DPI,
    ENABLE_AI_INTERPRETATION,
    GRID_COLS,
    GRID_ROWS,
    MAX_FILE_SIZE_MB,
    SENSITIVITY,
)
from .interpreter import (
    _block_has_changes,
    _create_block_comparison,
    _get_displacements_in_block,
)
from .scale_reader import SCALE_DPI, calculate_pixels_per_mm, read_scale

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
logger.info(
    "ENABLE_AI_INTERPRETATION=%s, ANTHROPIC_API_KEY config=%s...",
    ENABLE_AI_INTERPRETATION,
    ANTHROPIC_API_KEY[:10] if ANTHROPIC_API_KEY else "(leeg)",
)

app = FastAPI(
    title="K&K Tekening Diff",
    description="Vergelijk PDF bouwtekeningen en detecteer wijzigingen per pagina.",
    version="2.0.0",
)

# PDF magic bytes: %PDF
PDF_MAGIC = b"%PDF"

# Laatste vergelijkingsresultaat opslaan voor /results endpoint
_last_result: dict[str, Any] | None = None


def _validate_pdf(content: bytes, filename: str) -> str | None:
    """Valideer of het bestand een PDF is."""
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


def _read_scale_from_pdf(pdf_bytes: bytes) -> int:
    """Lees de schaal uit de eerste pagina van een PDF via Claude Vision."""
    try:
        pages = convert_from_bytes(pdf_bytes, dpi=SCALE_DPI, first_page=1, last_page=1)
        if pages:
            scale = read_scale(pages[0])
            pages[0].close()
            del pages
            gc.collect()
            return scale
    except Exception as e:
        logger.error("Schaal lezen mislukt: %s", str(e))
    return 50


@app.get("/")
async def health_check() -> dict[str, str]:
    """Health check endpoint."""
    return {"status": "online", "version": "2.0.0"}


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
    max_pages: int | None = Query(
        default=None,
        ge=1,
        description="Maximaal aantal pagina's om te verwerken (optioneel)",
    ),
    lite: bool = Query(
        default=False,
        description="Lite modus: geen afbeeldingen in response (voor n8n)",
    ),
) -> JSONResponse:
    """Vergelijk twee PDF bouwtekeningen en detecteer wijzigingen per pagina."""
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
        old_filename, len(old_bytes) // 1024,
        new_filename, len(new_bytes) // 1024,
        dpi, sensitivity,
    )

    # Aantal pagina's tellen ZONDER te renderen
    try:
        old_count = _get_page_count(old_bytes)
        new_count = _get_page_count(new_bytes)
    except Exception as e:
        logger.error("Fout bij het lezen van PDF info: %s", str(e))
        return JSONResponse(
            status_code=500,
            content={"status": "error", "detail": f"Kan PDF info niet lezen: {str(e)}"},
        )

    total_pages = max(old_count, new_count)
    if max_pages is not None:
        total_pages = min(total_pages, max_pages)
    logger.info("Pagina's geteld: oud=%d, nieuw=%d, verwerk=%d", old_count, new_count, total_pages)

    # Schaal lezen (1x per vergelijking, niet per pagina)
    scale = _read_scale_from_pdf(new_bytes)
    pixels_per_mm = calculate_pixels_per_mm(dpi, scale)
    logger.info("Schaal: 1:%d, pixels_per_mm: %.4f (bij DPI %d)", scale, pixels_per_mm, dpi)

    comparisons: list[dict[str, Any]] = []

    # Per pagina: converteer, vergelijk, ruim op
    for i in range(1, total_pages + 1):
        logger.info("[pagina %d/%d] Start verwerking...", i, total_pages)

        old_img = None
        new_img = None

        try:
            if i <= old_count:
                old_img = convert_from_bytes(
                    old_bytes, dpi=dpi, first_page=i, last_page=i
                )[0]

            if i <= new_count:
                new_img = convert_from_bytes(
                    new_bytes, dpi=dpi, first_page=i, last_page=i
                )[0]

            logger.info("[pagina %d/%d] Geconverteerd, start vergelijking...", i, total_pages)

            result = compare_page(
                old_img, new_img, i, sensitivity,
                scale=scale, pixels_per_mm=pixels_per_mm,
            )
            comparisons.append(result)

            logger.info(
                "[pagina %d/%d] Klaar — status=%s, wijziging=%.2f%%",
                i, total_pages, result["status"], result["change_percentage"],
            )

        except Exception as e:
            logger.error("[pagina %d/%d] FOUT: %s", i, total_pages, str(e))
            comparisons.append({
                "page": i,
                "status": "error",
                "changes_detected": False,
                "change_percentage": 0.0,
                "diff_image": None,
                "overlay_image": None,
                "error": f"Verwerkingsfout op pagina {i}: {str(e)}",
            })

        finally:
            if old_img is not None:
                old_img.close()
                del old_img
            if new_img is not None:
                new_img.close()
                del new_img
            gc.collect()

    del old_bytes, new_bytes
    gc.collect()

    logger.info("Vergelijking voltooid: %d pagina's verwerkt", total_pages)

    # Resultaat opslaan voor /results endpoint (altijd volledig)
    global _last_result
    _last_result = {
        "status": "success",
        "old_pages": old_count,
        "new_pages": new_count,
        "old_filename": old_filename,
        "new_filename": new_filename,
        "scale": scale,
        "comparisons": comparisons,
    }

    if lite:
        lite_comparisons = []
        for comp in comparisons:
            lite_comp: dict[str, Any] = {
                "page": comp["page"],
                "status": comp["status"],
                "changes_detected": comp["changes_detected"],
                "change_percentage": comp["change_percentage"],
            }
            if "interpretations" in comp:
                lite_comp["interpretations"] = [
                    {
                        "type": interp.get("type", ""),
                        "description": interp.get("description", ""),
                        "location": interp.get("location", ""),
                    }
                    for interp in comp["interpretations"]
                ]
            lite_comparisons.append(lite_comp)

        return JSONResponse(content={
            "status": "success",
            "old_pages": old_count,
            "new_pages": new_count,
            "scale": scale,
            "comparisons": lite_comparisons,
        })

    return JSONResponse(content=_last_result)


# ── Blok-gebaseerde vergelijking (zonder AI) ──────────────────────────

# Opslag voor blokafbeeldingen: key = "pagina_rij_kolom" -> JPEG bytes
_block_store: dict[str, bytes] = {}

BLOCK_JPEG_QUALITY = 85


def _png_to_jpeg(png_bytes: bytes) -> bytes:
    """Converteer PNG bytes naar JPEG bytes met kwaliteit 85."""
    from PIL import Image
    img = Image.open(io.BytesIO(png_bytes))
    if img.mode == "RGBA":
        img = img.convert("RGB")
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=BLOCK_JPEG_QUALITY)
    return buf.getvalue()


@app.get("/strip/{page}/{row}/{col}")
async def get_block(page: int, row: int, col: int) -> Response:
    """Haal een blokafbeelding op (OUD boven, NIEUW onder, JPEG)."""
    key = f"{page}_{row}_{col}"
    if key not in _block_store:
        return JSONResponse(
            status_code=404,
            content={
                "status": "error",
                "detail": (
                    f"Blok rij {row}, kolom {col} van pagina {page} "
                    f"niet gevonden."
                ),
            },
        )
    return Response(content=_block_store[key], media_type="image/jpeg")


@app.post("/compare-strips")
async def compare_strips(
    old_pdf: UploadFile = File(..., description="Oude versie van de PDF tekening"),
    new_pdf: UploadFile = File(..., description="Nieuwe versie van de PDF tekening"),
    dpi: int = Query(default=300, ge=72, le=600, description="Render DPI"),
    sensitivity: int = Query(
        default=SENSITIVITY,
        ge=1,
        le=255,
        description="Verschildrempel (lager = gevoeliger)",
    ),
    max_pages: int | None = Query(
        default=None,
        ge=1,
        description="Maximaal aantal pagina's om te verwerken (optioneel)",
    ),
) -> JSONResponse:
    """
    Vergelijk twee PDF's en geef blokmetadata terug zonder AI-interpretatie.

    Elke pagina wordt opgesplitst in een raster van GRID_COLS x GRID_ROWS blokken.
    Blokafbeeldingen zijn opvraagbaar via GET /strip/{page}/{row}/{col}.
    """
    old_bytes = await old_pdf.read()
    new_bytes = await new_pdf.read()

    old_filename = old_pdf.filename or "old_pdf"
    new_filename = new_pdf.filename or "new_pdf"

    error = _validate_pdf(old_bytes, old_filename)
    if error:
        return JSONResponse(status_code=400, content={"status": "error", "detail": error})

    error = _validate_pdf(new_bytes, new_filename)
    if error:
        return JSONResponse(status_code=400, content={"status": "error", "detail": error})

    logger.info(
        "Blok-vergelijking gestart: '%s' vs '%s', DPI=%d, sensitivity=%d",
        old_filename, new_filename, dpi, sensitivity,
    )

    try:
        old_count = _get_page_count(old_bytes)
        new_count = _get_page_count(new_bytes)
    except Exception as e:
        logger.error("Fout bij het lezen van PDF info: %s", str(e))
        return JSONResponse(
            status_code=500,
            content={"status": "error", "detail": f"Kan PDF info niet lezen: {str(e)}"},
        )

    total_pages = max(old_count, new_count)
    if max_pages is not None:
        total_pages = min(total_pages, max_pages)

    scale = _read_scale_from_pdf(new_bytes)
    pixels_per_mm = calculate_pixels_per_mm(dpi, scale)
    logger.info("Schaal: 1:%d, pixels_per_mm: %.4f (bij DPI %d)", scale, pixels_per_mm, dpi)

    # Oude blokken wissen
    global _block_store
    _block_store = {}

    pages_result: list[dict[str, Any]] = []

    for i in range(1, total_pages + 1):
        logger.info("[pagina %d/%d] Start blok-verwerking...", i, total_pages)
        old_img = None
        new_img = None

        try:
            if i <= old_count:
                old_img = convert_from_bytes(
                    old_bytes, dpi=dpi, first_page=i, last_page=i
                )[0]
            if i <= new_count:
                new_img = convert_from_bytes(
                    new_bytes, dpi=dpi, first_page=i, last_page=i
                )[0]

            raw = compare_page_raw(
                old_img, new_img, i, sensitivity,
                scale=scale, pixels_per_mm=pixels_per_mm,
            )

            blocks_meta: list[dict[str, Any]] = []

            if raw["diff_mask"] is not None and raw["aligned_old_bgr"] is not None:
                diff_mask = raw["diff_mask"]
                aligned_old_bgr = raw["aligned_old_bgr"]
                new_bgr = raw["new_bgr"]
                displacements = raw["displacements"]
                h, w = diff_mask.shape[:2]
                block_h = h // GRID_ROWS
                block_w = w // GRID_COLS

                for row in range(GRID_ROWS):
                    for col in range(GRID_COLS):
                        y_start = row * block_h
                        y_end = h if row == GRID_ROWS - 1 else (row + 1) * block_h
                        x_start = col * block_w
                        x_end = w if col == GRID_COLS - 1 else (col + 1) * block_w

                        has_changes = _block_has_changes(
                            diff_mask, y_start, y_end, x_start, x_end,
                        )
                        block_disps = _get_displacements_in_block(
                            displacements, y_start, y_end, x_start, x_end,
                        )

                        block_meta: dict[str, Any] = {
                            "row": row + 1,
                            "col": col + 1,
                            "has_changes": has_changes,
                            "url": None,
                            "displacements": [
                                {
                                    "x": d["x"],
                                    "y": d["y"],
                                    "verschuiving_mm": d["verschuiving_mm"],
                                    "verschuiving_px": d["verschuiving_px"],
                                }
                                for d in block_disps
                            ],
                        }

                        if has_changes:
                            png_bytes = _create_block_comparison(
                                aligned_old_bgr, new_bgr,
                                y_start, y_end, x_start, x_end,
                            )
                            jpeg_bytes = _png_to_jpeg(png_bytes)
                            key = f"{i}_{row + 1}_{col + 1}"
                            _block_store[key] = jpeg_bytes
                            block_meta["url"] = f"/strip/{i}/{row + 1}/{col + 1}"

                        blocks_meta.append(block_meta)

                # Ruwe data vrijgeven
                del diff_mask, aligned_old_bgr, new_bgr
            else:
                for row in range(GRID_ROWS):
                    for col in range(GRID_COLS):
                        blocks_meta.append({
                            "row": row + 1,
                            "col": col + 1,
                            "has_changes": False,
                            "url": None,
                            "displacements": [],
                        })

            pages_result.append({
                "page": raw["page"],
                "status": raw["status"],
                "changes_detected": raw["changes_detected"],
                "change_percentage": raw["change_percentage"],
                "blocks": blocks_meta,
            })

        except Exception as e:
            logger.error("[pagina %d] FOUT in compare-strips: %s", i, str(e))
            pages_result.append({
                "page": i,
                "status": "error",
                "changes_detected": False,
                "change_percentage": 0.0,
                "blocks": [],
                "error": str(e),
            })

        finally:
            if old_img is not None:
                old_img.close()
                del old_img
            if new_img is not None:
                new_img.close()
                del new_img
            gc.collect()

    del old_bytes, new_bytes
    gc.collect()

    logger.info(
        "Blok-vergelijking voltooid: %d pagina's, %d blokken opgeslagen",
        total_pages, len(_block_store),
    )

    return JSONResponse(content={
        "status": "success",
        "scale": scale,
        "old_pages": old_count,
        "new_pages": new_count,
        "old_filename": old_filename,
        "new_filename": new_filename,
        "grid": {"rows": GRID_ROWS, "cols": GRID_COLS},
        "pages": pages_result,
    })


# Badge kleuren per wijzigingstype
_BADGE_COLORS: dict[str, str] = {
    "WANDDIKTE": "#c0392b",       # rood
    "MAATVOERING": "#e67e22",     # oranje
    "KOZIJNHOOGTE": "#8e44ad",    # paars
    "WANDVERSCHUIVING": "#2980b9", # blauw
    "INDELING": "#27ae60",        # groen
    "SPARING": "#17a2b8",         # cyaan
    "MATERIAALCODE": "#f1c40f",   # geel
}


def _build_interpretations_html(interpretations: list[dict[str, Any]]) -> str:
    """Bouw HTML voor interpretaties, gegroepeerd per type."""
    if not interpretations:
        return ""

    # Groepeer per type
    by_type: dict[str, list[dict[str, Any]]] = {}
    for interp in interpretations:
        t = interp.get("type", "OVERIG")
        by_type.setdefault(t, []).append(interp)

    # Volgorde: WANDDIKTE, MAATVOERING, KOZIJNHOOGTE, WANDVERSCHUIVING, INDELING, SPARING, MATERIAALCODE
    type_order = [
        "WANDDIKTE", "MAATVOERING", "KOZIJNHOOGTE",
        "WANDVERSCHUIVING", "INDELING", "SPARING", "MATERIAALCODE",
    ]

    items_html = ""
    for change_type in type_order:
        changes = by_type.get(change_type, [])
        if not changes:
            continue

        color = _BADGE_COLORS.get(change_type, "#999")
        # Contrastkleur voor tekst
        text_color = "#000" if change_type == "MATERIAALCODE" else "#fff"

        for change in changes:
            desc = change.get("description", "")
            location = change.get("location", "")
            loc_html = (
                f'<span style="color:#888;font-size:13px;margin-left:8px;">'
                f'{location}</span>'
                if location else ""
            )

            items_html += f"""
                <div style="display:flex;align-items:flex-start;gap:12px;
                    padding:8px 0;border-bottom:1px solid #eee;">
                    <div style="flex:1;">
                        <span style="background:{color};color:{text_color};
                            padding:2px 10px;border-radius:8px;font-size:12px;
                            font-weight:600;white-space:nowrap;">{change_type}</span>
                        <span style="margin-left:8px;">{desc}</span>
                        {loc_html}
                    </div>
                </div>"""

    if not items_html:
        return ""

    return f"""
        <div style="margin-top:12px;padding:12px;background:#fff;
            border:1px solid #ddd;border-radius:6px;">
            <h3 style="margin:0 0 8px 0;font-size:16px;">
                Wijzigingen ({len(interpretations)})
            </h3>
            {items_html}
        </div>"""


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
    scale = _last_result.get("scale", "?")
    comparisons = _last_result.get("comparisons", [])

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

        # Interpretaties HTML met type badges
        interp_html = _build_interpretations_html(
            comp.get("interpretations", [])
        )

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
            &nbsp;|&nbsp; Schaal: 1:{scale}
        </p>
        {cards_html}
    </div>
</body>
</html>"""

    return HTMLResponse(content=html)
