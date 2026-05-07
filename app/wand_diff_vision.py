"""
Vision-based wand-vergelijkingspipeline.

Vervangt het Hungarian-matching algoritme door tegelsgewijs Claude Vision.
Publieke API is identiek aan wand_diff.py: bereken_wand_diff().
Activeer via USE_VISION_PIPELINE = True in config.py.
"""

from __future__ import annotations

import asyncio
import base64
import json
import logging
import time
from dataclasses import dataclass
from typing import TYPE_CHECKING

import anthropic
import fitz

if TYPE_CHECKING:
    from .config import DiffConfig

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Configuratie
# ---------------------------------------------------------------------------

RENDER_DPI         = 200
TILE_PX            = 1500
OVERLAP_PX         = 200
STEP_PX            = TILE_PX - OVERLAP_PX   # 1300 px stap

MODEL              = "claude-sonnet-4-6"
MAX_TOKENS         = 2048
INPUT_PRICE_PER_M  = 3.00
OUTPUT_PRICE_PER_M = 15.00

DEDUP_RADIUS_PT    = 60.0   # PDF-punten; duplicaten uit overlappende tegels

_PROMPT = (
    "Je krijgt twee crops ({w}x{h}px) van dezelfde Nederlandse afbouwtekening — oud en nieuw. "
    "Identificeer alle wandwijzigingen. Per wijziging:\n"
    "- type: \"toegevoegd\" | \"verwijderd\" | \"verplaatst\"\n"
    "- locatie: label/ruimtenaam/rasternummer\n"
    "- wandtype: indien leesbaar (bijv. \"gibo 70mm\", \"lichte scheidingswand\")\n"
    "- wanddikte_mm: integer of null\n"
    "- px_x: geschatte x-pixel in DEZE crop (0=links, {w}=rechts)\n"
    "- px_y: geschatte y-pixel in DEZE crop (0=boven, {h}=onder)\n"
    "Negeer titelblok, revisiedatums, QR-codes. "
    "Als er geen wandwijzigingen zijn: lege array. "
    'JSON: {{"wijzigingen": [...]}}'
)


# ---------------------------------------------------------------------------
# Dataklassen
# ---------------------------------------------------------------------------

@dataclass
class _Tile:
    col:      int
    row:      int
    px_x0:    int
    px_y0:    int
    px_w:     int
    px_h:     int
    old_jpeg: bytes
    new_jpeg: bytes


@dataclass
class _Stats:
    n_tiles:       int   = 0
    n_success:     int   = 0
    n_failed:      int   = 0
    input_tokens:  int   = 0
    output_tokens: int   = 0
    elapsed_s:     float = 0.0

    @property
    def cost_usd(self) -> float:
        return (self.input_tokens  / 1e6 * INPUT_PRICE_PER_M
                + self.output_tokens / 1e6 * OUTPUT_PRICE_PER_M)


# ---------------------------------------------------------------------------
# Tegel-generatie
# ---------------------------------------------------------------------------

def _tile_positions(total: int, tile_size: int, step: int) -> list[int]:
    """Startposities met volledige dekking, inclusief rechter/onderrand."""
    positions, pos = [], 0
    while pos + tile_size <= total:
        positions.append(pos)
        pos += step
    last = max(0, total - tile_size)
    if not positions or positions[-1] != last:
        positions.append(last)
    return positions


def _render_tiles(
    oud_page:  fitz.Page,
    nieuw_page: fitz.Page,
    oud_ori:   dict | None = None,
    nieuw_ori: dict | None = None,
) -> list[_Tile]:
    """
    Rendert overlappende tegels op RENDER_DPI via page.get_pixmap(clip=...).
    Clip-rect in PDF-punten: correct voor rotation=0.
    Voor rotation=90/180/270: display_width/height via ori dicts.
    """
    scale = RENDER_DPI / 72
    mat   = fitz.Matrix(scale, scale)

    dw = float((nieuw_ori or {}).get("display_width")  or oud_page.rect.width)
    dh = float((nieuw_ori or {}).get("display_height") or oud_page.rect.height)

    img_w = int(dw * scale)
    img_h = int(dh * scale)

    tiles: list[_Tile] = []
    for row, py0 in enumerate(_tile_positions(img_h, TILE_PX, STEP_PX)):
        for col, px0 in enumerate(_tile_positions(img_w, TILE_PX, STEP_PX)):
            px1 = min(px0 + TILE_PX, img_w)
            py1 = min(py0 + TILE_PX, img_h)

            clip = fitz.Rect(px0 / scale, py0 / scale, px1 / scale, py1 / scale)
            old_pix = oud_page.get_pixmap(matrix=mat, clip=clip, colorspace=fitz.csRGB)
            new_pix = nieuw_page.get_pixmap(matrix=mat, clip=clip, colorspace=fitz.csRGB)

            tiles.append(_Tile(
                col=col, row=row,
                px_x0=px0, px_y0=py0,
                px_w=old_pix.width, px_h=old_pix.height,
                old_jpeg=old_pix.tobytes("jpeg", jpg_quality=90),
                new_jpeg=new_pix.tobytes("jpeg", jpg_quality=90),
            ))

    logger.info("Vision: %d tegels, %dx%d px totaal, stap %d px",
                len(tiles), img_w, img_h, STEP_PX)
    return tiles


# ---------------------------------------------------------------------------
# Vision API-aanroepen (async)
# ---------------------------------------------------------------------------

def _parse_vision_json(text: str) -> list[dict]:
    text = text.strip()
    if "```" in text:
        for part in text.split("```"):
            part = part.lstrip("json").strip()
            if part.startswith("{"):
                text = part
                break
    try:
        return json.loads(text).get("wijzigingen", [])
    except Exception:
        pass
    start = text.find("{")
    end   = text.rfind("}") + 1
    if start >= 0 and end > start:
        try:
            return json.loads(text[start:end]).get("wijzigingen", [])
        except Exception:
            pass
    return []


async def _call_tile(
    client: anthropic.AsyncAnthropic,
    tile:   _Tile,
    stats:  _Stats,
) -> list[dict]:
    prompt = _PROMPT.format(w=tile.px_w, h=tile.px_h)
    try:
        resp = await client.messages.create(
            model=MODEL,
            max_tokens=MAX_TOKENS,
            messages=[{
                "role": "user",
                "content": [
                    {"type": "text",  "text": "Oud:"},
                    {"type": "image", "source": {
                        "type": "base64", "media_type": "image/jpeg",
                        "data": base64.standard_b64encode(tile.old_jpeg).decode(),
                    }},
                    {"type": "text",  "text": "Nieuw:"},
                    {"type": "image", "source": {
                        "type": "base64", "media_type": "image/jpeg",
                        "data": base64.standard_b64encode(tile.new_jpeg).decode(),
                    }},
                    {"type": "text",  "text": prompt},
                ],
            }],
        )
        stats.input_tokens  += resp.usage.input_tokens
        stats.output_tokens += resp.usage.output_tokens
        stats.n_success     += 1
        text = "".join(b.text for b in resp.content if hasattr(b, "text"))
        return _parse_vision_json(text)
    except Exception as exc:
        logger.warning("Tegel (%d,%d) mislukt: %s", tile.col, tile.row, exc)
        stats.n_failed += 1
        return []


async def _run_async(
    tiles:   list[_Tile],
    api_key: str | None,
) -> tuple[list[list[dict]], _Stats]:
    kwargs = {"api_key": api_key} if api_key else {}
    client  = anthropic.AsyncAnthropic(**kwargs)
    stats   = _Stats(n_tiles=len(tiles))
    results = await asyncio.gather(*[_call_tile(client, t, stats) for t in tiles])
    return list(results), stats


# ---------------------------------------------------------------------------
# Deduplicatie + coördinaten-omrekening
# ---------------------------------------------------------------------------

def _dedupliceer(
    tile_results: list[tuple[_Tile, list[dict]]],
    scale:        float,
) -> list[dict]:
    absolute: list[dict] = []
    for tile, changes in tile_results:
        for c in changes:
            px_x = float(c.get("px_x") or tile.px_w / 2)
            px_y = float(c.get("px_y") or tile.px_h / 2)
            pdf_x = (tile.px_x0 + px_x) / scale
            pdf_y = (tile.px_y0 + px_y) / scale
            absolute.append({**c, "pdf_x": pdf_x, "pdf_y": pdf_y})

    kept = []
    used = [False] * len(absolute)
    for i, a in enumerate(absolute):
        if used[i]:
            continue
        kept.append(a)
        used[i] = True
        for j in range(i + 1, len(absolute)):
            if used[j] or absolute[j].get("type") != a.get("type"):
                continue
            dx = absolute[j]["pdf_x"] - a["pdf_x"]
            dy = absolute[j]["pdf_y"] - a["pdf_y"]
            if (dx * dx + dy * dy) ** 0.5 < DEDUP_RADIUS_PT:
                used[j] = True

    return kept


def _naar_wand_formaat(changes: list[dict]) -> list[dict]:
    TYPE_MAP = {
        "toegevoegd": "nieuw",
        "verwijderd":  "verdwenen",
        "verplaatst":  "gewijzigd",
    }
    out = []
    for c in changes:
        x, y = c["pdf_x"], c["pdf_y"]
        r    = 25.0
        out.append({
            "type":          TYPE_MAP.get(c.get("type", ""), "gewijzigd"),
            "wandtype":      c.get("wandtype") or "",
            "wanddikte_mm":  c.get("wanddikte_mm"),
            "locatie_tekst": c.get("locatie") or "",
            "kleur":         None,
            "positie":       [x, y],
            "bbox":          [x - r, y - r, x + r, y + r],
        })
    return out


# ---------------------------------------------------------------------------
# Publieke API (identieke signatuur als wand_diff.py)
# ---------------------------------------------------------------------------

def bereken_wand_diff(
    oud_page:   fitz.Page,
    nieuw_page: fitz.Page,
    oud_ori:    dict,
    nieuw_ori:  dict,
    legenda:    dict,
    api_key:    str | None = None,
    cfg:        "DiffConfig | None" = None,
    _stats_out: dict | None = None,
) -> list[dict]:
    """
    Vision-based wand-diff. Signatuur identiek aan wand_diff.py.
    Activeer via USE_VISION_PIPELINE = True in config.py.
    """
    t0    = time.time()
    scale = RENDER_DPI / 72

    tiles = _render_tiles(oud_page, nieuw_page, oud_ori, nieuw_ori)

    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        loop = None

    if loop and loop.is_running():
        import concurrent.futures
        with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
            fut = pool.submit(asyncio.run, _run_async(tiles, api_key))
            results, stats = fut.result()
    else:
        results, stats = asyncio.run(_run_async(tiles, api_key))

    stats.elapsed_s = time.time() - t0

    deduped = _dedupliceer(list(zip(tiles, results)), scale)
    output  = _naar_wand_formaat(deduped)

    logger.info(
        "Vision wand-diff: %d tegels, %d wijzigingen, $%.4f, %.1fs",
        stats.n_tiles, len(output), stats.cost_usd, stats.elapsed_s,
    )

    if _stats_out is not None:
        _stats_out.update({
            "n_tiles":       stats.n_tiles,
            "n_success":     stats.n_success,
            "n_failed":      stats.n_failed,
            "input_tokens":  stats.input_tokens,
            "output_tokens": stats.output_tokens,
            "cost_usd":      stats.cost_usd,
            "elapsed_s":     stats.elapsed_s,
            "n_wijzigingen": len(output),
        })

    return output
