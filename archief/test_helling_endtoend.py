"""
End-to-end test: 56 de Helling met renvooi-gebaseerde wand-detectie.

Stap 1: Beide renvooien uitlezen via Vision
Stap 2: Pagina renderen en tegelen
Stap 3: Per tegel 1 Vision-call (sequentieel)
Stap 4: Centrale tegel 3x voor consistentie-check

Gebruik: python test_helling_endtoend.py
"""

import asyncio
import base64
import json
import time
from pathlib import Path

import anthropic
import fitz

# ---------------------------------------------------------------------------
# Paden
# ---------------------------------------------------------------------------

SCRIPT_DIR = Path(__file__).parent
REF_DIR    = SCRIPT_DIR / "references" / "helling"
MVP_DIR    = SCRIPT_DIR.parent / "Karregat & Koning MVP"
TMP_DIR    = Path("/tmp/helling_test")
TMP_DIR.mkdir(parents=True, exist_ok=True)

OUD_PDF  = MVP_DIR / "56 de Helling - plattegronden gibowanden - oude tekening (5).pdf"
NIEUW_PDF = MVP_DIR / "56 de Helling - plattegronden gibowanden - Nieuwe tekening (2).pdf"

MODEL      = "claude-sonnet-4-5"
RENDER_DPI = 200
TILE_PX    = 1500
OVERLAP_PX = 200
STEP_PX    = TILE_PX - OVERLAP_PX
PAGINA     = 0

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _laad_b64(pad: Path) -> tuple[str, str]:
    data = pad.read_bytes()
    if data[:3] == b"\xff\xd8\xff":
        return base64.standard_b64encode(data).decode(), "image/jpeg"
    return base64.standard_b64encode(data).decode(), "image/png"


def _pix_naar_b64(pix: fitz.Pixmap) -> str:
    return base64.standard_b64encode(pix.tobytes("jpeg", jpg_quality=90)).decode()


def _parse_json(tekst: str) -> dict:
    tekst = tekst.strip()
    if "```" in tekst:
        for deel in tekst.split("```"):
            deel = deel.lstrip("json").strip()
            if deel.startswith("{"):
                tekst = deel
                break
    try:
        return json.loads(tekst)
    except Exception:
        pass
    s = tekst.find("{")
    e = tekst.rfind("}") + 1
    if s >= 0 and e > s:
        try:
            return json.loads(tekst[s:e])
        except Exception:
            pass
    return {}


def _tile_posities(totaal: int) -> list[int]:
    pos, result = 0, []
    while pos + TILE_PX <= totaal:
        result.append(pos)
        pos += STEP_PX
    laatste = max(0, totaal - TILE_PX)
    if not result or result[-1] != laatste:
        result.append(laatste)
    return result


def _centrum_idx(posities: list[int]) -> int:
    return len(posities) // 2


def _wandtypes_tekst(wandtypes: list[dict]) -> str:
    return "\n".join(
        f"- {w['naam']} ({w.get('categorie', '?')}): {w.get('visuele_kenmerken', '')}"
        for w in wandtypes
    )


# ---------------------------------------------------------------------------
# STAP 1 — Renvooi uitlezen
# ---------------------------------------------------------------------------

_RENVOOI_PROMPT = (
    "Lees alle wandtypes uit dit renvooi. "
    'JSON: {"wandtypes": [{"naam": string, "visuele_kenmerken": string, "categorie": string}]}. '
    "Categorieen: metselwerk, kalkzandsteen, gips, hsb, sandwich, isolatie, voorzetwand, beton, gevelafwerking, anders. "
    "Negeer niet-wand items zoals peilmaat, brandwerendheid, deuren, pijlen, vluchtwegen."
)


async def _call_renvooi(client: anthropic.AsyncAnthropic, pad: Path, label: str) -> list[dict]:
    b64, media = _laad_b64(pad)
    resp = await client.messages.create(
        model=MODEL,
        max_tokens=2048,
        messages=[{
            "role": "user",
            "content": [
                {"type": "image", "source": {"type": "base64", "media_type": media, "data": b64}},
                {"type": "text", "text": _RENVOOI_PROMPT},
            ],
        }],
    )
    tekst = "".join(b.text for b in resp.content if hasattr(b, "text"))
    data  = _parse_json(tekst)
    wandtypes = data.get("wandtypes", [])
    print(f"  {label}: {len(wandtypes)} wandtypes, {resp.usage.input_tokens + resp.usage.output_tokens} tokens")
    return wandtypes


async def stap1_renvooien() -> tuple[list[dict], list[dict], list[dict]]:
    print("\n" + "=" * 60)
    print("STAP 1 -- Renvooi uitlezing")
    print("=" * 60)

    client = anthropic.AsyncAnthropic()
    oud_wt, nieuw_wt = await asyncio.gather(
        _call_renvooi(client, REF_DIR / "renvooi_oud.png",  "Oud renvooi"),
        _call_renvooi(client, REF_DIR / "renvooi_nieuw.png", "Nieuw renvooi"),
    )

    # Opslaan
    for wt, naam in [(oud_wt, "vision_renvooi_oud.json"), (nieuw_wt, "vision_renvooi_nieuw.json")]:
        with open(REF_DIR / naam, "w", encoding="utf-8") as f:
            json.dump({"wandtypes": wt}, f, ensure_ascii=False, indent=2)

    # Vergelijking
    oud_namen  = {w["naam"].lower().strip() for w in oud_wt}
    nieuw_namen = {w["naam"].lower().strip() for w in nieuw_wt}
    alleen_nieuw = nieuw_namen - oud_namen
    alleen_oud   = oud_namen - nieuw_namen
    gedeeld      = oud_namen & nieuw_namen

    print(f"\n  Gedeeld ({len(gedeeld)}): {', '.join(sorted(gedeeld))}")
    if alleen_nieuw:
        print(f"  Alleen in nieuw ({len(alleen_nieuw)}): {', '.join(sorted(alleen_nieuw))}")
    if alleen_oud:
        print(f"  Alleen in oud ({len(alleen_oud)}): {', '.join(sorted(alleen_oud))}")

    # Gecombineerde unie (oud als basis, nieuw vult aan)
    gecombineerd = list(oud_wt)
    oud_lower = {w["naam"].lower().strip() for w in oud_wt}
    for w in nieuw_wt:
        if w["naam"].lower().strip() not in oud_lower:
            gecombineerd.append(w)

    print(f"\n  Gecombineerde lijst: {len(gecombineerd)} wandtypes")
    for w in gecombineerd:
        print(f"    - {w['naam']} [{w.get('categorie','?')}]")

    return oud_wt, nieuw_wt, gecombineerd


# ---------------------------------------------------------------------------
# STAP 2 -- Tegelen
# ---------------------------------------------------------------------------

def stap2_tegelen() -> tuple[list[dict], int, int]:
    """Render pagina 1 van beide PDFs op 200 DPI en sla alle crops op.

    Returns: (tegels, centrum_col_idx, centrum_row_idx)
    Elke tegel: {col, row, px_x0, py_y0, oud_b64, nieuw_b64}
    """
    print("\n" + "=" * 60)
    print("STAP 2 -- Tegelen")
    print("=" * 60)

    scale = RENDER_DPI / 72
    mat   = fitz.Matrix(scale, scale)

    oud_doc   = fitz.open(str(OUD_PDF))
    nieuw_doc = fitz.open(str(NIEUW_PDF))
    oud_page  = oud_doc[PAGINA]
    nieuw_page = nieuw_doc[PAGINA]

    img_w = int(nieuw_page.rect.width  * scale)
    img_h = int(nieuw_page.rect.height * scale)

    x_pos = _tile_posities(img_w)
    y_pos = _tile_posities(img_h)

    print(f"  Paginaformaat: {nieuw_page.rect.width:.0f} x {nieuw_page.rect.height:.0f} pt")
    print(f"  Render: {img_w} x {img_h} px bij {RENDER_DPI} DPI")
    print(f"  Tegel-grid: {len(x_pos)} kolommen x {len(y_pos)} rijen = {len(x_pos) * len(y_pos)} tegels")

    cx_idx = _centrum_idx(x_pos)
    cy_idx = _centrum_idx(y_pos)

    tegels = []
    for ri, py0 in enumerate(y_pos):
        for ci, px0 in enumerate(x_pos):
            px1 = min(px0 + TILE_PX, img_w)
            py1 = min(py0 + TILE_PX, img_h)
            clip = fitz.Rect(px0 / scale, py0 / scale, px1 / scale, py1 / scale)

            oud_pix   = oud_page.get_pixmap(matrix=mat, clip=clip, colorspace=fitz.csRGB)
            nieuw_pix = nieuw_page.get_pixmap(matrix=mat, clip=clip, colorspace=fitz.csRGB)

            # Opslaan als PNG
            oud_pix.save(str(TMP_DIR / f"tegel_r{ri:02d}_c{ci:02d}_oud.png"))
            nieuw_pix.save(str(TMP_DIR / f"tegel_r{ri:02d}_c{ci:02d}_nieuw.png"))

            centrum = (ri == cy_idx and ci == cx_idx)
            label = " [CENTRUM]" if centrum else ""
            print(f"  Tegel r{ri}c{ci}: px_x0={px0}, py_y0={py0}, breedte={px1-px0}, hoogte={py1-py0}{label}")

            tegels.append({
                "col":     ci,
                "row":     ri,
                "px_x0":   px0,
                "py_y0":   py0,
                "oud_b64":  _pix_naar_b64(oud_pix),
                "nieuw_b64": _pix_naar_b64(nieuw_pix),
                "centrum": centrum,
            })

    oud_doc.close()
    nieuw_doc.close()
    print(f"  Alle crops opgeslagen in {TMP_DIR}/")
    return tegels, cx_idx, cy_idx


# ---------------------------------------------------------------------------
# STAP 3 -- Detectie per tegel
# ---------------------------------------------------------------------------

_DETECTIE_PROMPT = """\
Twee crops van dezelfde Nederlandse afbouwtekening (oud + nieuw).
De wandtypes op deze tekening volgens het renvooi:
{wandtypes_tekst}

Identificeer wandwijzigingen tussen oud en nieuw. Per wijziging:
- type: toegevoegd | verdwenen
- wandtype: kies uit de lijst, of 'onbekend'
- locatie: globale beschrijving binnen crop

NEGEER rode markeringen -- dat zijn architect-aantekeningen, geen wand-wijzigingen.
JSON: {{"wijzigingen": [{{"type": string, "wandtype": string, "locatie": string}}]}}\
"""


async def _call_detectie_een(
    client: anthropic.AsyncAnthropic,
    tegel: dict,
    wandtypes_tekst: str,
) -> dict:
    prompt = _DETECTIE_PROMPT.format(wandtypes_tekst=wandtypes_tekst)
    try:
        resp = await client.messages.create(
            model=MODEL,
            max_tokens=2048,
            messages=[{
                "role": "user",
                "content": [
                    {"type": "text",  "text": "Oud:"},
                    {"type": "image", "source": {"type": "base64", "media_type": "image/jpeg", "data": tegel["oud_b64"]}},
                    {"type": "text",  "text": "Nieuw:"},
                    {"type": "image", "source": {"type": "base64", "media_type": "image/jpeg", "data": tegel["nieuw_b64"]}},
                    {"type": "text",  "text": prompt},
                ],
            }],
        )
        tekst = "".join(b.text for b in resp.content if hasattr(b, "text"))
        data  = _parse_json(tekst)
        return {
            "wijzigingen": data.get("wijzigingen", []),
            "tokens":      resp.usage.input_tokens + resp.usage.output_tokens,
            "fout":        None,
        }
    except Exception as e:
        return {"wijzigingen": [], "tokens": 0, "fout": str(e)}


async def stap3_detectie(tegels: list[dict], gecombineerd: list[dict]) -> list[dict]:
    print("\n" + "=" * 60)
    print("STAP 3 -- Wand-detectie per tegel (sequentieel)")
    print("=" * 60)

    client  = anthropic.AsyncAnthropic()
    wt_tekst = _wandtypes_tekst(gecombineerd)
    resultaten = []
    totaal_tokens = 0

    for tegel in tegels:
        r, c = tegel["row"], tegel["col"]
        res = await _call_detectie_een(client, tegel, wt_tekst)
        totaal_tokens += res["tokens"]
        n = len(res["wijzigingen"])
        fout_label = f"  FOUT: {res['fout']}" if res["fout"] else ""

        print(f"  Tegel r{r}c{c}: {n} wijziging(en), {res['tokens']} tokens{fout_label}")
        for w in res["wijzigingen"]:
            print(f"    {w.get('type','?'):10s} | {w.get('wandtype','?'):<25} | {w.get('locatie','?')}")

        resultaten.append({**tegel, "detectie": res})

    print(f"\n  Totaal tokens stap 3: {totaal_tokens}")
    return resultaten


# ---------------------------------------------------------------------------
# STAP 4 -- Centrale tegel 3x
# ---------------------------------------------------------------------------

async def stap4_consistentie(centrum_tegel: dict, gecombineerd: list[dict]) -> None:
    print("\n" + "=" * 60)
    print("STAP 4 -- Centrale tegel consistentie-check (3 runs parallel)")
    print("=" * 60)

    client   = anthropic.AsyncAnthropic()
    wt_tekst = _wandtypes_tekst(gecombineerd)

    runs = list(await asyncio.gather(*[
        _call_detectie_een(client, centrum_tegel, wt_tekst)
        for _ in range(3)
    ]))

    breedte = 36
    header = "  " + "  ".join(
        f"{'Run %d (%d wijz.)' % (i + 1, len(runs[i]['wijzigingen'])):<{breedte}}"
        for i in range(3)
    )
    print(header)
    print("  " + ("  " + "-" * breedte) * 3)

    max_len = max(len(r["wijzigingen"]) for r in runs)
    for i in range(max_len):
        def fmt(r):
            if i >= len(r["wijzigingen"]):
                return ""
            w = r["wijzigingen"][i]
            return f"{w.get('type','?')} | {w.get('wandtype','?')[:22]}"
        print("  " + "  ".join(f"{fmt(r):<{breedte}}" for r in runs))

    def run_set(r):
        return {(w.get("type", ""), w.get("wandtype", "").lower()) for w in r["wijzigingen"]}

    sets    = [run_set(r) for r in runs]
    in_alle = sets[0] & sets[1] & sets[2]
    in_min2 = ((sets[0] & sets[1]) | (sets[0] & sets[2]) | (sets[1] & sets[2])) - in_alle

    totaal_uniek = len(sets[0] | sets[1] | sets[2])
    pct = len(in_alle) / totaal_uniek * 100 if totaal_uniek else 0.0

    print(f"\n  In alle 3 runs ({len(in_alle)}):")
    for t, wt in sorted(in_alle):
        print(f"    OK {t} | {wt}")
    print(f"  In 2 van 3 runs: {len(in_min2)}")
    print(f"  Consistentie: {pct:.0f}% ({len(in_alle)}/{totaal_uniek} unieke wijzigingen)")

    tokens = [r["tokens"] for r in runs]
    print(f"  Tokens: run1={tokens[0]}, run2={tokens[1]}, run3={tokens[2]}")

    # Check sandwichpaneel
    sandwich_hits = [
        any("sandwich" in w.get("wandtype", "").lower() for w in r["wijzigingen"])
        for r in runs
    ]
    print(f"\n  Sandwichpaneel gevonden: run1={sandwich_hits[0]}, run2={sandwich_hits[1]}, run3={sandwich_hits[2]}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

async def main():
    t0 = time.time()

    print("=" * 60)
    print("END-TO-END TEST: 56 de Helling")
    print("=" * 60)

    for pad, label in [(OUD_PDF, "Oud PDF"), (NIEUW_PDF, "Nieuw PDF")]:
        status = "OK" if pad.exists() else "NIET GEVONDEN"
        print(f"  {label}: {status}")
    for pad, label in [
        (REF_DIR / "renvooi_oud.png",  "Renvooi oud"),
        (REF_DIR / "renvooi_nieuw.png", "Renvooi nieuw"),
    ]:
        status = "OK" if pad.exists() else "NIET GEVONDEN"
        print(f"  {label}: {status}")

    # Stap 1
    oud_wt, nieuw_wt, gecombineerd = await stap1_renvooien()

    # Stap 2
    tegels, cx, cy = stap2_tegelen()

    # Stap 3
    resultaten = await stap3_detectie(tegels, gecombineerd)

    # Stap 4 — centrale tegel
    centrum_tegel = next(t for t in tegels if t["centrum"])
    await stap4_consistentie(centrum_tegel, gecombineerd)

    # Eindrapport
    elapsed = time.time() - t0
    totaal_wijzigingen = sum(len(r["detectie"]["wijzigingen"]) for r in resultaten)
    totaal_tokens_s3   = sum(r["detectie"]["tokens"] for r in resultaten)

    print("\n" + "=" * 60)
    print("EINDRAPPORT")
    print("=" * 60)
    print(f"  Renvooi oud:     {len(oud_wt)} wandtypes")
    print(f"  Renvooi nieuw:   {len(nieuw_wt)} wandtypes")
    print(f"  Gecombineerd:    {len(gecombineerd)} wandtypes")
    print(f"  Tegels:          {len(tegels)}")
    print(f"  Wijzigingen:     {totaal_wijzigingen} (stap 3, 1 call per tegel)")
    print(f"  Tokens stap 1:   2 calls (renvooi)")
    print(f"  Tokens stap 3:   {totaal_tokens_s3}")
    print(f"  Totale tijd:     {elapsed:.1f}s")

    # Sandwichpaneel check over alle tegels
    sandwich_tegels = [
        f"r{r['row']}c{r['col']}"
        for r in resultaten
        if any("sandwich" in w.get("wandtype", "").lower() for w in r["detectie"]["wijzigingen"])
    ]
    if sandwich_tegels:
        print(f"\n  Sandwichpaneel gedetecteerd in tegel(s): {', '.join(sandwich_tegels)}")
    else:
        print("\n  Sandwichpaneel: NIET gedetecteerd in stap 3")


if __name__ == "__main__":
    asyncio.run(main())
