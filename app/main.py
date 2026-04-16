"""FastAPI app voor K&K tekeningvergelijking."""

import base64
import datetime
import logging
import os
import tempfile
from pathlib import Path

import hashlib
import secrets

import httpx
from fastapi import Cookie, Depends, FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse, JSONResponse, RedirectResponse, Response
from fastapi.staticfiles import StaticFiles

import fitz

from .config import DiffConfig
from .diff_engine import run_diff
from .interpreter import interpreteer_diff
from .overlay import generate_overlay_pdf, generate_multi_page_overlay, generate_split_rapport

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

SLACK_WEBHOOK_URL = os.environ.get("SLACK_WEBHOOK_URL", "")
APP_USERNAME = os.environ.get("APP_USERNAME", "GiboTekening")
APP_PASSWORD = os.environ.get("APP_PASSWORD", "")

# Sessie-token = hash van gebruikersnaam + wachtwoord (stabiel, geen opslag nodig)
def _maak_sessie_token() -> str:
    combo = f"{APP_USERNAME}:{APP_PASSWORD}"
    return hashlib.sha256(combo.encode()).hexdigest()


def _check_sessie(kk_sessie: str | None = Cookie(default=None)):
    if not APP_PASSWORD:
        return  # Auth uitgeschakeld als geen wachtwoord ingesteld
    if not kk_sessie or not secrets.compare_digest(kk_sessie, _maak_sessie_token()):
        raise HTTPException(status_code=401, detail="Niet ingelogd")

# Map waar feedback + PDFs worden opgeslagen (/tmp is altijd schrijfbaar)
_feedback_dir = Path("/tmp/feedback-opslag")
_feedback_dir.mkdir(exist_ok=True)

app = FastAPI(
    title="K&K Tekening Diff",
    description="Vergelijk PDF demarcatietekeningen en detecteer wijzigingen.",
    version="2.0.0",
    dependencies=[Depends(_check_sessie)],
)

# Statische bestanden (de webpagina)
_static_dir = Path(__file__).parent / "static"
app.mount("/static", StaticFiles(directory=str(_static_dir)), name="static")

PDF_MAGIC = b"%PDF"
MAX_FILE_SIZE_MB = 50


@app.get("/login", dependencies=[])
def login_pagina():
    return FileResponse(str(_static_dir / "login.html"))


@app.post("/login", dependencies=[])
def login_submit(
    gebruikersnaam: str = Form(...),
    wachtwoord: str = Form(...),
):
    ok_user = secrets.compare_digest(gebruikersnaam.encode(), APP_USERNAME.encode())
    ok_pass = secrets.compare_digest(wachtwoord.encode(), APP_PASSWORD.encode()) if APP_PASSWORD else True
    if not (ok_user and ok_pass):
        raise HTTPException(status_code=401, detail="Onjuiste inloggegevens")
    resp = JSONResponse(content={"ok": True})
    resp.set_cookie(
        key="kk_sessie",
        value=_maak_sessie_token(),
        httponly=True,
        samesite="lax",
        max_age=60 * 60 * 24 * 30,  # 30 dagen
    )
    return resp


@app.get("/")
def index(kk_sessie: str | None = Cookie(default=None)):
    if APP_PASSWORD and (not kk_sessie or not secrets.compare_digest(kk_sessie, _maak_sessie_token())):
        return RedirectResponse(url="/login")
    return FileResponse(str(_static_dir / "index.html"))


@app.post("/feedback")
async def feedback(
    oud_bestand: str = Form(""),
    nieuw_bestand: str = Form(""),
    pagina: str = Form(""),
    type_probleem: str = Form(""),
    locatie: str = Form(""),
    wat_zag_je: str = Form(""),
    wat_fout: str = Form(""),
    wat_had_moeten: str = Form(""),
    oud_pdf: UploadFile | None = File(None),
    nieuw_pdf: UploadFile | None = File(None),
):
    """Ontvang feedback, sla PDFs op en stuur naar Slack."""
    # Timestamp als unieke referentie
    ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    feedback_map = _feedback_dir / ts
    feedback_map.mkdir(exist_ok=True)

    # PDFs opslaan
    for upload, naam in [(oud_pdf, oud_bestand), (nieuw_pdf, nieuw_bestand)]:
        if upload:
            inhoud = await upload.read()
            if inhoud:
                bestandsnaam = naam or upload.filename or "onbekend.pdf"
                (feedback_map / bestandsnaam).write_bytes(inhoud)

    # Downloadlink voor dit feedback-item
    base_url = os.environ.get("RAILWAY_PUBLIC_DOMAIN", "")
    if base_url:
        download_url = f"https://{base_url}/feedback-bestanden/{ts}"
    else:
        download_url = f"/feedback-bestanden/{ts}"

    type_emoji = {
        "gemiste_wijziging": "🔍 Gemiste wijziging",
        "fout_markering": "❌ Fout markering",
        "verkeerd_label": "🏷️ Verkeerd label",
        "anders": "💬 Anders",
    }.get(type_probleem, type_probleem)

    blokken = [
        {
            "type": "header",
            "text": {"type": "plain_text", "text": "📋 Nieuwe feedback — Tekeningvergelijker"},
        },
        {
            "type": "section",
            "fields": [
                {"type": "mrkdwn", "text": f"*Oude tekening:*\n{oud_bestand or '—'}"},
                {"type": "mrkdwn", "text": f"*Nieuwe tekening:*\n{nieuw_bestand or '—'}"},
                {"type": "mrkdwn", "text": f"*Pagina:*\n{pagina or '—'}"},
                {"type": "mrkdwn", "text": f"*Type probleem:*\n{type_emoji}"},
            ],
        },
    ]

    if locatie:
        blokken.append({
            "type": "section",
            "text": {"type": "mrkdwn", "text": f"*📍 Locatie op tekening:*\n{locatie}"},
        })
    if wat_zag_je:
        blokken.append({
            "type": "section",
            "text": {"type": "mrkdwn", "text": f"*👁️ Wat stond er op de tekening:*\n{wat_zag_je}"},
        })
    if wat_fout:
        blokken.append({
            "type": "section",
            "text": {"type": "mrkdwn", "text": f"*⚠️ Wat deed het systeem fout:*\n{wat_fout}"},
        })
    if wat_had_moeten:
        blokken.append({
            "type": "section",
            "text": {"type": "mrkdwn", "text": f"*✅ Wat had het systeem moeten doen:*\n{wat_had_moeten}"},
        })

    blokken.append({
        "type": "section",
        "text": {"type": "mrkdwn", "text": f"*📁 Tekeningen downloaden:*\n<{download_url}|Klik hier — referentie: {ts}>"},
    })

    if not SLACK_WEBHOOK_URL:
        logger.warning("SLACK_WEBHOOK_URL niet ingesteld — feedback niet verstuurd")
        return JSONResponse(status_code=503, content={"error": "Slack niet geconfigureerd."})

    try:
        async with httpx.AsyncClient(timeout=10) as client:
            resp = await client.post(SLACK_WEBHOOK_URL, json={"blocks": blokken})
            resp.raise_for_status()
        logger.info("Feedback verstuurd naar Slack (ref: %s)", ts)
        return JSONResponse(content={"ok": True})
    except Exception as e:
        logger.error("Slack feedback fout: %s", e)
        return JSONResponse(status_code=500, content={"error": "Kon feedback niet versturen."})


@app.get("/feedback-bestanden/{ts}")
def feedback_bestanden_lijst(ts: str):
    """Toon lijst van opgeslagen feedback-bestanden voor een tijdstempel."""
    feedback_map = _feedback_dir / ts
    if not feedback_map.exists():
        return JSONResponse(status_code=404, content={"error": "Niet gevonden."})
    bestanden = [f.name for f in feedback_map.iterdir()]
    return JSONResponse(content={"referentie": ts, "bestanden": bestanden})


@app.get("/feedback-bestanden/{ts}/{bestandsnaam}")
def feedback_bestand_download(ts: str, bestandsnaam: str):
    """Download een opgeslagen feedback-bestand."""
    pad = _feedback_dir / ts / bestandsnaam
    if not pad.exists() or not pad.is_file():
        return JSONResponse(status_code=404, content={"error": "Bestand niet gevonden."})
    return FileResponse(str(pad), media_type="application/pdf",
                        headers={"Content-Disposition": f"attachment; filename={bestandsnaam}"})


@app.get("/health")
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


@app.post("/vergelijk-split")
async def vergelijk_split(
    oud_pdf: UploadFile = File(..., description="Oude versie van de PDF"),
    nieuw_pdf: UploadFile = File(..., description="Nieuwe versie van de PDF"),
    aantal_paginas: int | None = Form(None, description="Aantal pagina's (leeg = automatisch)"),
):
    """Hoofdendpoint voor de webinterface.

    Genereert twee PDF's:
      - rapport_pdf: A4 samenvatting per pagina
      - tekeningen_pdf: gemarkeerde tekeningen met legenda

    Beide worden als base64 teruggegeven zodat de browser ze direct kan downloaden.
    """
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
        if not aantal_paginas:
            oud_doc = fitz.open(oud_path)
            nieuw_doc = fitz.open(nieuw_path)
            aantal_paginas = min(len(oud_doc), len(nieuw_doc))
            oud_doc.close()
            nieuw_doc.close()

        aantal_paginas = max(1, min(aantal_paginas, 50))

        logger.info(
            "Vergelijk-split: '%s' vs '%s', %d pagina('s)",
            oud_pdf.filename, nieuw_pdf.filename, aantal_paginas,
        )

        rapport_bytes, tekeningen_bytes = generate_split_rapport(
            oud_path, nieuw_path, aantal_paginas,
            oud_pdf.filename or "oud.pdf",
            nieuw_pdf.filename or "nieuw.pdf",
        )

        logger.info(
            "Vergelijk-split klaar: rapport=%dKB, tekeningen=%dKB",
            len(rapport_bytes) // 1024, len(tekeningen_bytes) // 1024,
        )

        return JSONResponse(content={
            "rapport_pdf": base64.b64encode(rapport_bytes).decode(),
            "tekeningen_pdf": base64.b64encode(tekeningen_bytes).decode(),
        })

    except Exception as e:
        logger.exception("Fout bij vergelijk-split")
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
