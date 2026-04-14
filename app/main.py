"""FastAPI app voor K&K tekeningvergelijking."""

import base64
import logging
import os
import tempfile

from fastapi import FastAPI, File, Form, UploadFile
from fastapi.responses import JSONResponse, Response

import fitz

from .config import DiffConfig
from .diff_engine import run_diff
from .interpreter import interpreteer_diff
from .overlay import generate_overlay_pdf, generate_multi_page_overlay

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="K&K Tekening Diff",
    description="Vergelijk PDF demarcatietekeningen en detecteer wijzigingen.",
    version="2.0.0",
)

PDF_MAGIC = b"%PDF"
MAX_FILE_SIZE_MB = 50


@app.get("/")
def health():
    return {"status": "ok", "version": "2.0"}


@app.post("/diff")
async def diff(
    oud_pdf: UploadFile = File(..., description="Oude versie van de PDF"),
    nieuw_pdf: UploadFile = File(..., description="Nieuwe versie van de PDF"),
    pagina: int = Form(1, description="Paginanummer (1-based)"),
):
    """Vergelijk twee PDF's en retourneer alle wijzigingen als JSON."""
    oud_bytes = await oud_pdf.read()
    nieuw_bytes = await nieuw_pdf.read()

    # Validatie
    for label, content, filename in [
        ("oud_pdf", oud_bytes, oud_pdf.filename),
        ("nieuw_pdf", nieuw_bytes, nieuw_pdf.filename),
    ]:
        if not content.startswith(PDF_MAGIC):
            return JSONResponse(
                status_code=400,
                content={"error": f"{filename} is geen geldig PDF bestand."},
            )
        if len(content) > MAX_FILE_SIZE_MB * 1024 * 1024:
            return JSONResponse(
                status_code=400,
                content={"error": f"{filename} is te groot (max {MAX_FILE_SIZE_MB} MB)."},
            )

    oud_name = oud_pdf.filename or "oud.pdf"
    nieuw_name = nieuw_pdf.filename or "nieuw.pdf"

    logger.info(
        "Diff gestart: '%s' (%d KB) vs '%s' (%d KB), pagina %d",
        oud_name, len(oud_bytes) // 1024,
        nieuw_name, len(nieuw_bytes) // 1024,
        pagina,
    )

    # Schrijf naar temp bestanden (pymupdf werkt met file paths)
    try:
        with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as f_oud:
            f_oud.write(oud_bytes)
            oud_path = f_oud.name

        with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as f_nieuw:
            f_nieuw.write(nieuw_bytes)
            nieuw_path = f_nieuw.name

        config = DiffConfig()
        result = run_diff(oud_path, nieuw_path, pagina=pagina - 1,
                          config=config)

        # Overschrijf bestandsnamen in meta met originele namen
        if "meta" in result:
            result["meta"]["oud_bestand"] = oud_name
            result["meta"]["nieuw_bestand"] = nieuw_name

        if "error" in result:
            logger.error("Diff fout: %s", result["error"])
            return JSONResponse(status_code=400, content=result)

        # Interpreteer de wijzigingen naar leesbaar rapport
        alle_tekst = result.get("nieuw_tekst_items", [])
        pw = result.get("meta", {}).get("pagina_breedte", 0)
        result["interpretatie"] = interpreteer_diff(result, alle_tekst, pw)

        totalen = result.get("totalen", {})
        logger.info("Diff voltooid: %s", totalen)

        # Verwijder intern layout object (niet JSON-serialiseerbaar)
        result.pop("_layout_obj", None)

        return JSONResponse(content=result)

    except Exception as e:
        logger.exception("Onverwachte fout bij diff")
        return JSONResponse(
            status_code=500,
            content={"error": f"Vergelijkingsfout: {str(e)}"},
        )
    finally:
        for path in (oud_path, nieuw_path):
            try:
                os.unlink(path)
            except OSError:
                pass


def _validate_and_save_pdfs(
    oud_bytes: bytes, nieuw_bytes: bytes,
    oud_filename: str, nieuw_filename: str,
) -> tuple[str, str] | JSONResponse:
    """Valideer PDF bytes en schrijf naar temp bestanden. Returns paths of error response."""
    for label, content, filename in [
        ("oud_pdf", oud_bytes, oud_filename),
        ("nieuw_pdf", nieuw_bytes, nieuw_filename),
    ]:
        if not content.startswith(PDF_MAGIC):
            return JSONResponse(
                status_code=400,
                content={"error": f"{filename} is geen geldig PDF bestand."},
            )
        if len(content) > MAX_FILE_SIZE_MB * 1024 * 1024:
            return JSONResponse(
                status_code=400,
                content={"error": f"{filename} is te groot (max {MAX_FILE_SIZE_MB} MB)."},
            )

    with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as f_oud:
        f_oud.write(oud_bytes)
        oud_path = f_oud.name

    with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as f_nieuw:
        f_nieuw.write(nieuw_bytes)
        nieuw_path = f_nieuw.name

    return oud_path, nieuw_path


@app.post("/overlay")
async def overlay(
    oud_pdf: UploadFile = File(..., description="Oude versie van de PDF"),
    nieuw_pdf: UploadFile = File(..., description="Nieuwe versie van de PDF"),
    pagina: int = Form(1, description="Paginanummer (1-based)"),
):
    """Genereer een gemarkeerde PDF met alle wijzigingen visueel aangeduid."""
    oud_bytes = await oud_pdf.read()
    nieuw_bytes = await nieuw_pdf.read()

    result = _validate_and_save_pdfs(
        oud_bytes, nieuw_bytes,
        oud_pdf.filename or "oud.pdf", nieuw_pdf.filename or "nieuw.pdf",
    )
    if isinstance(result, JSONResponse):
        return result
    oud_path, nieuw_path = result

    try:
        logger.info("Overlay gestart: pagina %d", pagina)
        diff_result = run_diff(oud_path, nieuw_path, pagina=pagina - 1,
                               config=DiffConfig())

        if "error" in diff_result:
            return JSONResponse(status_code=400, content=diff_result)

        if "meta" in diff_result:
            diff_result["meta"]["oud_bestand"] = oud_pdf.filename or "oud.pdf"
            diff_result["meta"]["nieuw_bestand"] = nieuw_pdf.filename or "nieuw.pdf"

        pdf_bytes = generate_overlay_pdf(
            oud_path, nieuw_path, diff_result, pagina=pagina - 1,
        )
        logger.info("Overlay voltooid: %d bytes", len(pdf_bytes))

        return Response(
            content=pdf_bytes,
            media_type="application/pdf",
            headers={"Content-Disposition": "attachment; filename=wijzigingsrapport.pdf"},
        )
    except Exception as e:
        logger.exception("Onverwachte fout bij overlay")
        return JSONResponse(
            status_code=500,
            content={"error": f"Overlay fout: {str(e)}"},
        )
    finally:
        for path in (oud_path, nieuw_path):
            try:
                os.unlink(path)
            except OSError:
                pass


@app.post("/rapport")
async def rapport(
    oud_pdf: UploadFile = File(..., description="Oude versie van de PDF"),
    nieuw_pdf: UploadFile = File(..., description="Nieuwe versie van de PDF"),
    pagina: int = Form(1, description="Paginanummer (1-based)"),
):
    """Gecombineerd endpoint: diff JSON + overlay PDF als base64."""
    oud_bytes = await oud_pdf.read()
    nieuw_bytes = await nieuw_pdf.read()

    result = _validate_and_save_pdfs(
        oud_bytes, nieuw_bytes,
        oud_pdf.filename or "oud.pdf", nieuw_pdf.filename or "nieuw.pdf",
    )
    if isinstance(result, JSONResponse):
        return result
    oud_path, nieuw_path = result

    try:
        logger.info("Rapport gestart: pagina %d", pagina)
        diff_result = run_diff(oud_path, nieuw_path, pagina=pagina - 1,
                               config=DiffConfig())

        if "error" in diff_result:
            return JSONResponse(status_code=400, content=diff_result)

        if "meta" in diff_result:
            diff_result["meta"]["oud_bestand"] = oud_pdf.filename or "oud.pdf"
            diff_result["meta"]["nieuw_bestand"] = nieuw_pdf.filename or "nieuw.pdf"

        pdf_bytes = generate_overlay_pdf(
            oud_path, nieuw_path, diff_result, pagina=pagina - 1,
        )
        logger.info("Rapport voltooid: diff + %d bytes PDF", len(pdf_bytes))

        return JSONResponse(content={
            "diff": diff_result,
            "overlay_pdf_base64": base64.b64encode(pdf_bytes).decode(),
        })
    except Exception as e:
        logger.exception("Onverwachte fout bij rapport")
        return JSONResponse(
            status_code=500,
            content={"error": f"Rapport fout: {str(e)}"},
        )
    finally:
        for path in (oud_path, nieuw_path):
            try:
                os.unlink(path)
            except OSError:
                pass


@app.post("/rapport-volledig")
async def rapport_volledig(
    oud_pdf: UploadFile = File(..., description="Oude versie van de PDF"),
    nieuw_pdf: UploadFile = File(..., description="Nieuwe versie van de PDF"),
):
    """Vergelijk ALLE pagina's en genereer een compleet rapport PDF."""
    oud_bytes = await oud_pdf.read()
    nieuw_bytes = await nieuw_pdf.read()

    result = _validate_and_save_pdfs(
        oud_bytes, nieuw_bytes,
        oud_pdf.filename or "oud.pdf", nieuw_pdf.filename or "nieuw.pdf",
    )
    if isinstance(result, JSONResponse):
        return result
    oud_path, nieuw_path = result

    try:
        oud_doc = fitz.open(oud_path)
        nieuw_doc = fitz.open(nieuw_path)
        gemeenschappelijk = min(len(oud_doc), len(nieuw_doc))
        oud_doc.close()
        nieuw_doc.close()

        logger.info(
            "Volledig rapport: %d gemeenschappelijke pagina's",
            gemeenschappelijk,
        )

        pdf_bytes = generate_multi_page_overlay(
            oud_path, nieuw_path, gemeenschappelijk,
            oud_pdf.filename or "oud.pdf",
            nieuw_pdf.filename or "nieuw.pdf",
        )
        logger.info("Volledig rapport: %d bytes", len(pdf_bytes))

        return Response(
            content=pdf_bytes,
            media_type="application/pdf",
            headers={
                "Content-Disposition": "attachment; filename=wijzigingsrapport_volledig.pdf",
            },
        )
    except Exception as e:
        logger.exception("Fout bij volledig rapport")
        return JSONResponse(
            status_code=500,
            content={"error": f"Rapport fout: {str(e)}"},
        )
    finally:
        for path in (oud_path, nieuw_path):
            try:
                os.unlink(path)
            except OSError:
                pass
