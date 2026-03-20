"""FastAPI app voor K&K tekeningvergelijking."""

import logging
import os
import tempfile

from fastapi import FastAPI, File, Form, UploadFile
from fastapi.responses import JSONResponse

from .diff_engine import run_diff

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

        result = run_diff(oud_path, nieuw_path, pagina=pagina - 1)

        # Overschrijf bestandsnamen in meta met originele namen
        if "meta" in result:
            result["meta"]["oud_bestand"] = oud_name
            result["meta"]["nieuw_bestand"] = nieuw_name

        if "error" in result:
            logger.error("Diff fout: %s", result["error"])
            return JSONResponse(status_code=400, content=result)

        totalen = result.get("totalen", {})
        logger.info("Diff voltooid: %s", totalen)

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
