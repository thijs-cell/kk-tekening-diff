"""Preflight checks voor PDF-uploads — vangt te grote of te complexe tekeningen af."""

import logging

import fitz
from fastapi import HTTPException

logger = logging.getLogger(__name__)

MAX_PADEN_PER_PAGINA = 500_000
MAX_BESTAND_MB = 50


def controleer_pdfs(
    oud_bytes: bytes,
    nieuw_bytes: bytes,
    oud_naam: str,
    nieuw_naam: str,
) -> None:
    """Gooit HTTPException 422 als een bestand te groot of te complex is.

    Controleert bestandsgrootte en vectorpad-telling op alle pagina's.
    """
    for pdf_bytes, naam in [(oud_bytes, oud_naam), (nieuw_bytes, nieuw_naam)]:
        _controleer_bestand(pdf_bytes, naam)


def _controleer_bestand(pdf_bytes: bytes, naam: str) -> None:
    mb = len(pdf_bytes) / (1024 * 1024)
    if mb > MAX_BESTAND_MB:
        raise HTTPException(
            status_code=422,
            detail=(
                f"Bestand '{naam}' is te groot ({mb:.0f} MB, max {MAX_BESTAND_MB} MB). "
                "Splits het bestand en probeer opnieuw."
            ),
        )

    doc = fitz.open(stream=pdf_bytes, filetype="pdf")
    try:
        for i, page in enumerate(doc):
            aantal = len(page.get_cdrawings())
            logger.debug("Preflight '%s' pagina %d: %d paden", naam, i + 1, aantal)
            if aantal > MAX_PADEN_PER_PAGINA:
                raise HTTPException(
                    status_code=422,
                    detail=(
                        f"Tekening '{naam}' (pagina {i + 1}) is te complex om te verwerken "
                        f"({aantal:,} paden, max {MAX_PADEN_PER_PAGINA:,}). "
                        "Splits de tekening of neem contact op."
                    ),
                )
    finally:
        doc.close()
