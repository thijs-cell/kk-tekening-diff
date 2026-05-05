"""
Test Vision-fallback activatie in vind_legenda_combined().

Gebruik:
    cd kk-tekening-diff
    python test_vision_activatie.py

Vereist: ANTHROPIC_API_KEY in .env of os.environ
"""
import base64 as _b64
import json as _json
import os
import sys

sys.path.insert(0, os.path.dirname(__file__))

import anthropic as _anthropic_mod
import fitz

from app.tekening_profiel import (
    _MAX_SWATCH_AREA,
    _VISION_PROMPT,
    _display_naar_raw_clip,
    _is_neutraal,
    _rnd,
    _vind_legenda_titels,
    detecteer_orientatie,
    vind_legenda,
    vind_legenda_combined,
    vind_legenda_vision,
)

BASE = os.path.join(os.path.dirname(__file__), "..", "Karregat & Koning MVP")

# Laad .env als die bestaat
_env_path = os.path.join(os.path.dirname(__file__), ".env")
if os.path.exists(_env_path):
    with open(_env_path) as _f:
        for _line in _f:
            _line = _line.strip()
            if _line and not _line.startswith("#") and "=" in _line:
                _k, _v = _line.split("=", 1)
                os.environ.setdefault(_k.strip(), _v.strip())

API_KEY = os.environ.get("ANTHROPIC_API_KEY", "")

SEP = "=" * 65


def _open(naam, pdf):
    path = os.path.join(BASE, pdf)
    if not os.path.exists(path):
        print(f"  BESTAND NIET GEVONDEN: {path}")
        return None, None, None
    doc = fitz.open(path)
    page = doc[0]
    ori = detecteer_orientatie(page)
    print(f"  Bestand : {pdf}")
    print(f"  Rotatie : {ori['rotation']}  |  "
          f"Display: {ori['display_width']:.0f} x {ori['display_height']:.0f} pt")
    return doc, page, ori


def _debug_vision(page, ori, api_key):
    """
    Repliceert de stappen van vind_legenda_vision() met volledige debug-output.
    Wijzigt de productiefunctie NIET.
    """
    normalize = ori["normalize"]
    rotation = ori["rotation"]
    dw = ori["display_width"]
    dh = ori["display_height"]

    # Stap 0: legenda-titels
    titel_posities = _vind_legenda_titels(page, normalize)
    if not titel_posities:
        print("  [DEBUG] Geen legenda-titels gevonden")
        return

    # Stap 1: crop-zone (spiegelt productielogica)
    tx, ty = titel_posities[0]
    _STANDARD_W = 600
    _standard_x1 = min(dw, tx - 50 + _STANDARD_W)
    _clipped_w = _standard_x1 - max(0.0, tx - 50)
    if _clipped_w < _STANDARD_W * 0.7:
        crop_x1 = min(dw, tx + 200)
        crop_x0 = max(0.0, crop_x1 - 1000)
        crop_y0 = max(0.0, min(t[1] for t in titel_posities) - 30)
        crop_y1 = min(dh, crop_y0 + 600)
        print(f"  [DEBUG] Crop: ({crop_x0:.0f}, {crop_y0:.0f}) -> ({crop_x1:.0f}, {crop_y1:.0f})"
              f"  [links-flip, clipped_w={_clipped_w:.0f}pt]")
    else:
        crop_x0 = max(0.0, tx - 50)
        crop_y0 = max(0.0, ty - 30)
        crop_x1 = _standard_x1
        crop_y1 = min(dh, crop_y0 + 800)
        print(f"  [DEBUG] Crop: ({crop_x0:.0f}, {crop_y0:.0f}) -> ({crop_x1:.0f}, {crop_y1:.0f})"
              f"  [standaard]")

    # Stap 2: zone_kleuren — dezelfde logica als in vind_legenda_vision stap 7
    zone_kleuren = []
    for d in page.get_drawings():
        rect = d.get("rect")
        if rect is None:
            continue
        area = (rect[2] - rect[0]) * (rect[3] - rect[1])
        if area > _MAX_SWATCH_AREA:
            continue
        rep_kleur = None
        fill = d.get("fill")
        if fill is not None and not _is_neutraal(fill):
            rep_kleur = _rnd(fill)
        if rep_kleur is None:
            kleur = d.get("color")
            breedte = d.get("width") or 0
            if kleur is not None and not _is_neutraal(kleur) and breedte > 0.15:
                rep_kleur = _rnd(kleur)
        if rep_kleur is None:
            continue
        cx = (rect[0] + rect[2]) / 2
        cy = (rect[1] + rect[3]) / 2
        dx, dy = normalize(cx, cy)
        if crop_x0 <= dx <= crop_x1 and crop_y0 <= dy <= crop_y1:
            if rep_kleur not in zone_kleuren:
                zone_kleuren.append(rep_kleur)

    print(f"  [DEBUG] zone_kleuren ({len(zone_kleuren)} uniek):")
    if zone_kleuren:
        for k in zone_kleuren:
            print(f"    {k}")
    else:
        print("    (leeg — geen niet-neutrale swatches in crop)")

    # Stap 3: render crop en API-call (display-coords direct, geen conversie nodig)
    raw_clip = fitz.Rect(crop_x0, crop_y0, crop_x1, crop_y1) & page.rect
    if raw_clip.is_empty:
        print("  [DEBUG] raw_clip is leeg")
        return

    mat = fitz.Matrix(200 / 72, 200 / 72)
    pix = page.get_pixmap(matrix=mat, clip=raw_clip, colorspace=fitz.csRGB)
    png_bytes = pix.tobytes("png")
    print(f"  [DEBUG] PNG crop: {len(png_bytes) // 1024} KB  "
          f"({pix.width} x {pix.height} px)")

    client = _anthropic_mod.Anthropic(api_key=api_key)
    b64_image = _b64.b64encode(png_bytes).decode()

    import time as _time
    _t0 = _time.time()
    response = client.messages.create(
        model="claude-sonnet-4-6",
        max_tokens=1024,
        messages=[{
            "role": "user",
            "content": [
                {
                    "type": "image",
                    "source": {
                        "type": "base64",
                        "media_type": "image/png",
                        "data": b64_image,
                    },
                },
                {"type": "text", "text": _VISION_PROMPT},
            ],
        }],
    )
    _t1 = _time.time()
    _usage = response.usage
    # Sonnet 4.6 prijzen mei 2026: $3/MTok input, $15/MTok output
    _cost = (_usage.input_tokens * 3 + _usage.output_tokens * 15) / 1_000_000
    print(f"  [Vision] {_usage.input_tokens} in + {_usage.output_tokens} out tokens"
          f" = ${_cost:.4f}  ({_t1 - _t0:.1f}s)")

    # Stap 4: raw tekst
    raw_tekst = response.content[0].text.strip()
    print(f"\n  [DEBUG] Raw vision_tekst ({len(raw_tekst)} chars):")
    print(raw_tekst)

    # Stap 5: geparsete JSON
    print()
    start = raw_tekst.find("[")
    end = raw_tekst.rfind("]") + 1
    if start >= 0 and end > start:
        try:
            parsed = _json.loads(raw_tekst[start:end])
            print(f"  [DEBUG] Geparsete JSON ({len(parsed)} items):")
            for item in parsed:
                print(f"    {item}")
        except _json.JSONDecodeError as e:
            print(f"  [DEBUG] JSON parse fout: {e}")
    else:
        print("  [DEBUG] Geen JSON array gevonden in response")


# ---------------------------------------------------------------------------
# TEST A — Geen API key: combined() mag niet crashen
# ---------------------------------------------------------------------------
print(f"\n{SEP}")
print("TEST A — Geen API key (combined moet gewoon vector teruggeven)")
print(SEP)

_backup = os.environ.pop("ANTHROPIC_API_KEY", None)
try:
    result = _open("5102", "5102_Eerste verdieping_05-03-2025_ (2).pdf")
    if result[0] is not None:
        doc, page, ori = result
        combined = vind_legenda_combined(page, ori, api_key=None)
        print(f"\n  Resultaat: {len(combined)} item(s)")
        if combined:
            for rgb, naam in combined.items():
                print(f"    {rgb}  ->  {naam}")
        else:
            print("  (geen resultaten — verwacht voor 5102 vector)")
        doc.close()
finally:
    if _backup:
        os.environ["ANTHROPIC_API_KEY"] = _backup

print("  OK: geen crash")


# ---------------------------------------------------------------------------
# TEST B — 56 de Helling: Vision mag NIET worden aangeroepen
# ---------------------------------------------------------------------------
print(f"\n{SEP}")
print("TEST B — 56 de Helling (vector >= 2 -> Vision NIET aanroepen)")
print(SEP)

if not API_KEY:
    print("  OVERGESLAGEN — geen API key beschikbaar")
else:
    result = _open("56 de Helling",
                   "56 de Helling - plattegronden gibowanden - Nieuwe tekening (2).pdf")
    if result[0] is not None:
        doc, page, ori = result
        print()
        combined = vind_legenda_combined(page, ori, api_key=API_KEY)
        print(f"\n  Resultaat combined: {len(combined)} item(s)")
        doc.close()


# ---------------------------------------------------------------------------
# TEST C — 5102: debug vision_tekst, JSON en zone_kleuren
# ---------------------------------------------------------------------------
print(f"\n{SEP}")
print("TEST C — 5102 (vector = 0 -> Vision als fallback) [DEBUG]")
print(SEP)

if not API_KEY:
    print("  OVERGESLAGEN — geen API key beschikbaar")
else:
    result = _open("5102", "5102_Eerste verdieping_05-03-2025_ (2).pdf")
    if result[0] is not None:
        doc, page, ori = result
        print()
        _debug_vision(page, ori, API_KEY)
        doc.close()


# ---------------------------------------------------------------------------
# TEST D — Muiden D2.1: debug vision_tekst, JSON en zone_kleuren
# ---------------------------------------------------------------------------
print(f"\n{SEP}")
print("TEST D — Muiden D2.1 (directe vergelijking vector vs Vision) [DEBUG]")
print(SEP)

if not API_KEY:
    print("  OVERGESLAGEN — geen API key beschikbaar")
else:
    result = _open("Muiden D2.1", "WT-PLG-D2.1_20260202_E.pdf")
    if result[0] is not None:
        doc, page, ori = result

        print("\n  [VECTOR]")
        vector = vind_legenda(page, ori)
        if vector:
            for rgb, naam in vector.items():
                print(f"    {rgb}  ->  {naam}")
        else:
            print("    (geen resultaten)")

        print("\n  [VISION — DEBUG]")
        _debug_vision(page, ori, API_KEY)

        print(f"\n  Vector: {len(vector)} items")
        doc.close()


print(f"\n{SEP}")
print("Klaar.")
