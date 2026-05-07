"""
Standalone test: Vision-met-zelf-uitgelezen-renvooi voor wanddetectie.

Fase 1: 3x parallelle Vision-calls per renvooi (9 calls totaal)
Fase 2: 3x parallelle wand-detectie-calls per tekening met renvooi als context

Gebruik: python test_renvooi_vision.py
"""

import asyncio
import base64
import json
import os
import time
from pathlib import Path

import anthropic
import fitz

# ---------------------------------------------------------------------------
# Paden
# ---------------------------------------------------------------------------

SCRIPT_DIR  = Path(__file__).parent
REF_DIR     = SCRIPT_DIR / "references"
MVP_DIR     = SCRIPT_DIR.parent / "Karregat & Koning MVP"
TMP_DIR     = Path("/tmp/renvooi_test")
TMP_DIR.mkdir(parents=True, exist_ok=True)

RENVOOI_PADEN = {
    "muiden":  REF_DIR / "muiden"  / "renvooi.png",
    "bd_n101": REF_DIR / "bd_n101" / "renvooi.png",
    "5102":    REF_DIR / "5102"    / "renvooi.png",
}

PDF_PAREN = {
    "muiden": (
        MVP_DIR / "WT-PLG-D2.1_20250324_B.pdf",
        MVP_DIR / "WT-PLG-D2.1_20260202_E.pdf",
    ),
    "bd_n101": (
        MVP_DIR / "BD-N-101 - PLATTEGROND 1e VERDIEPING.pdf (1).pdf",
        MVP_DIR / "BU-N-101 - PLATTEGROND 1e VERDIEPING----V2 (1).pdf",
    ),
    "5102": (
        MVP_DIR / "5102_Eerste verdieping_17-01-2025_ (3).pdf",
        MVP_DIR / "5102_Eerste verdieping_05-03-2025_ (2).pdf",
    ),
}

MODEL      = "claude-sonnet-4-5"
MAX_TOKENS = 2048
RENDER_DPI = 200
TILE_PX    = 1500
OVERLAP_PX = 200
STEP_PX    = TILE_PX - OVERLAP_PX

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _laad_b64(pad: Path) -> tuple[str, str]:
    data = pad.read_bytes()
    # Detecteer werkelijk formaat via magic bytes — extensie kan misleidend zijn
    if data[:3] == b"\xff\xd8\xff":
        media = "image/jpeg"
    elif data[:4] == b"\x89PNG":
        media = "image/png"
    elif data[:4] == b"RIFF" and data[8:12] == b"WEBP":
        media = "image/webp"
    elif data[:4] == b"GIF8":
        media = "image/gif"
    else:
        media = {".png": "image/png", ".jpg": "image/jpeg", ".jpeg": "image/jpeg"}.get(
            pad.suffix.lower(), "image/png"
        )
    return base64.standard_b64encode(data).decode(), media


def _pix_naar_b64(pix: fitz.Pixmap) -> str:
    return base64.standard_b64encode(pix.tobytes("jpeg", jpg_quality=90)).decode()


def _tile_posities(totaal: int) -> list[int]:
    posities, pos = [], 0
    while pos + TILE_PX <= totaal:
        posities.append(pos)
        pos += STEP_PX
    laatste = max(0, totaal - TILE_PX)
    if not posities or posities[-1] != laatste:
        posities.append(laatste)
    return posities


def _centrum_tegel_start(totaal: int) -> int:
    posities = _tile_posities(totaal)
    return posities[len(posities) // 2] if posities else 0


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
    start = tekst.find("{")
    eind  = tekst.rfind("}") + 1
    if start >= 0 and eind > start:
        try:
            return json.loads(tekst[start:eind])
        except Exception:
            pass
    return {}


def _render_centrum_tegel(
    oud_path: Path, nieuw_path: Path, pagina: int = 0
) -> tuple[str, str, int, int]:
    """Render de centrale tegel op 200 DPI voor beide PDFs. Sla ook op in /tmp."""
    scale = RENDER_DPI / 72
    mat   = fitz.Matrix(scale, scale)

    oud_doc   = fitz.open(str(oud_path))
    nieuw_doc = fitz.open(str(nieuw_path))
    oud_page  = oud_doc[pagina]
    nieuw_page = nieuw_doc[pagina]

    img_w = int(nieuw_page.rect.width  * scale)
    img_h = int(nieuw_page.rect.height * scale)

    px0 = _centrum_tegel_start(img_w)
    py0 = _centrum_tegel_start(img_h)
    px1 = min(px0 + TILE_PX, img_w)
    py1 = min(py0 + TILE_PX, img_h)

    clip = fitz.Rect(px0 / scale, py0 / scale, px1 / scale, py1 / scale)

    oud_pix   = oud_page.get_pixmap(matrix=mat, clip=clip, colorspace=fitz.csRGB)
    nieuw_pix = nieuw_page.get_pixmap(matrix=mat, clip=clip, colorspace=fitz.csRGB)

    oud_doc.close()
    nieuw_doc.close()

    # Opslaan in /tmp
    naam_base = oud_path.stem[:20].replace(" ", "_")
    oud_pix.save(str(TMP_DIR / f"{naam_base}_oud_centrum.png"))
    nieuw_pix.save(str(TMP_DIR / f"{naam_base}_nieuw_centrum.png"))

    return _pix_naar_b64(oud_pix), _pix_naar_b64(nieuw_pix), px0, py0


# ---------------------------------------------------------------------------
# FASE 1 — Renvooi Vision-calls
# ---------------------------------------------------------------------------

_RENVOOI_PROMPT = (
    "Dit is het renvooi van een Nederlandse afbouwtekening. "
    "Extraheer alle wandtypes als JSON:\n"
    '{"wandtypes": [{"naam": string, "visuele_kenmerken": string, "categorie": string}]}\n'
    "Categorieën: metselwerk, kalkzandsteen, gips, metalstud, beton, isolatie, hsb, prefab, anders.\n"
    "Geen niet-wand elementen meenemen (deuren, symbolen, brandveiligheid)."
)


async def _call_renvooi(
    client: anthropic.AsyncAnthropic,
    b64: str,
    media: str,
    run_nr: int,
) -> dict:
    try:
        resp = await client.messages.create(
            model=MODEL,
            max_tokens=MAX_TOKENS,
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
        return {
            "run":       run_nr,
            "wandtypes": data.get("wandtypes", []),
            "tokens":    resp.usage.input_tokens + resp.usage.output_tokens,
            "fout":      None,
        }
    except Exception as e:
        return {"run": run_nr, "wandtypes": [], "tokens": 0, "fout": str(e)}


def _analyseer_consistentie(runs: list[dict]) -> dict:
    lijsten = [
        {w["naam"].lower().strip() for w in r["wandtypes"]}
        for r in runs
    ]
    alle = set().union(*lijsten)
    consistent   = {n for n in alle if all(n in l for l in lijsten)}
    inconsistent = alle - consistent
    pct = len(consistent) / len(alle) * 100 if alle else 0.0
    return {
        "runs":        runs,
        "consistent":  sorted(consistent),
        "inconsistent": sorted(inconsistent),
        "pct":         pct,
        "go":          pct >= 80.0,
    }


async def fase1() -> dict[str, dict]:
    client = anthropic.AsyncAnthropic()

    # Laad alle afbeeldingen
    taken: list[tuple[str, str, str, int]] = []
    for naam, pad in RENVOOI_PADEN.items():
        if not pad.exists():
            print(f"  SKIP {naam}: {pad} niet gevonden")
            continue
        b64, media = _laad_b64(pad)
        for run in range(1, 4):
            taken.append((naam, b64, media, run))

    # 9 calls parallel
    async def _run_one(naam, b64, media, run):
        return naam, await _call_renvooi(client, b64, media, run)

    responses = await asyncio.gather(*[_run_one(n, b, m, r) for n, b, m, r in taken])

    # Groepeer + analyseer
    per_tekening: dict[str, list] = {}
    for naam, res in responses:
        per_tekening.setdefault(naam, []).append(res)

    return {
        naam: _analyseer_consistentie(sorted(runs, key=lambda x: x["run"]))
        for naam, runs in per_tekening.items()
    }


def _print_fase1(naam: str, res: dict):
    runs = res["runs"]
    lijsten = [[w["naam"] for w in r["wandtypes"]] for r in runs]
    tokens  = [r["tokens"] for r in runs]

    print(f"\n{'='*70}")
    print(f"RENVOOI: {naam.upper()}")
    print(f"{'='*70}")

    breedte = 26
    header = "  " + "  ".join(
        f"{'Run %d (%d types)' % (i+1, len(lijsten[i])):<{breedte}}"
        for i in range(3)
    )
    print(header)
    print("  " + ("  " + "-" * breedte) * 3)

    max_len = max(len(l) for l in lijsten) if lijsten else 0
    for i in range(max_len):
        rij = "  " + "  ".join(
            f"{(lijsten[j][i] if i < len(lijsten[j]) else ''):<{breedte}}"
            for j in range(3)
        )
        print(rij)

    print(f"\n  Tokens: run1={tokens[0]}, run2={tokens[1]}, run3={tokens[2]}")

    # Fouten zichtbaar maken
    for r in runs:
        if r["fout"]:
            print(f"  FOUT run {r['run']}: {r['fout']}")

    print(f"\n  Consistent in alle 3 runs - {len(res['consistent'])} types:")
    for n in res["consistent"]:
        print(f"    OK {n}")

    if res["inconsistent"]:
        print(f"\n  Inconsistent - {len(res['inconsistent'])} types:")
        for n in res["inconsistent"]:
            in_runs = [
                str(i + 1)
                for i, r in enumerate(runs)
                if n in {w["naam"].lower().strip() for w in r["wandtypes"]}
            ]
            print(f"    ~ {n}  [in run(s): {', '.join(in_runs)}]")

    label = "GO OK" if res["go"] else "NO-GO X"
    print(f"\n  Consistentie: {res['pct']:.0f}%  ->  {label}")


# ---------------------------------------------------------------------------
# FASE 2 — Detectie met renvooi-context
# ---------------------------------------------------------------------------

_DETECTIE_PROMPT = (
    "Hier zie je twee crops van dezelfde Nederlandse afbouwtekening — oud en nieuw. "
    "De wandtypes die op deze tekening voorkomen volgens het renvooi:\n\n"
    "{wandtypes_tekst}\n\n"
    "Identificeer alle wandwijzigingen tussen oud en nieuw. Per wijziging:\n"
    "- type: toegevoegd | verdwenen\n"
    "- wandtype: kies uit de lijst hierboven, of 'onbekend'\n"
    "- locatie: globale beschrijving binnen crop\n\n"
    '{{"wijzigingen": [{{"type": string, "wandtype": string, "locatie": string}}]}}'
)


async def _call_detectie(
    client: anthropic.AsyncAnthropic,
    oud_b64: str,
    nieuw_b64: str,
    wandtypes_tekst: str,
    run_nr: int,
) -> dict:
    prompt = _DETECTIE_PROMPT.format(wandtypes_tekst=wandtypes_tekst)
    try:
        resp = await client.messages.create(
            model=MODEL,
            max_tokens=MAX_TOKENS,
            messages=[{
                "role": "user",
                "content": [
                    {"type": "text",  "text": "Oud:"},
                    {"type": "image", "source": {"type": "base64", "media_type": "image/jpeg", "data": oud_b64}},
                    {"type": "text",  "text": "Nieuw:"},
                    {"type": "image", "source": {"type": "base64", "media_type": "image/jpeg", "data": nieuw_b64}},
                    {"type": "text",  "text": prompt},
                ],
            }],
        )
        tekst = "".join(b.text for b in resp.content if hasattr(b, "text"))
        data  = _parse_json(tekst)
        return {
            "run":        run_nr,
            "wijzigingen": data.get("wijzigingen", []),
            "tokens":     resp.usage.input_tokens + resp.usage.output_tokens,
            "fout":       None,
        }
    except Exception as e:
        return {"run": run_nr, "wijzigingen": [], "tokens": 0, "fout": str(e)}


def _wandtypes_tekst(res1: dict) -> str:
    beste = max(res1["runs"], key=lambda r: len(r["wandtypes"]))
    return "\n".join(
        f"- {w['naam']} ({w.get('categorie','?')}): {w.get('visuele_kenmerken','')}"
        for w in beste["wandtypes"]
    )


async def fase2(fase1_res: dict[str, dict]) -> None:
    client = anthropic.AsyncAnthropic()

    for naam in ["muiden", "bd_n101", "5102"]:
        res1 = fase1_res.get(naam)
        if not res1:
            print(f"\n  SKIP {naam}: niet in fase 1 resultaten")
            continue
        if not res1["go"]:
            print(f"\n  SKIP fase 2 voor {naam}: fase 1 NO-GO ({res1['pct']:.0f}%)")
            continue

        oud_path, nieuw_path = PDF_PAREN[naam]
        if not oud_path.exists() or not nieuw_path.exists():
            print(f"\n  SKIP {naam}: PDF niet gevonden")
            print(f"    oud:  {oud_path}")
            print(f"    nieuw: {nieuw_path}")
            continue

        print(f"\n{'='*70}")
        print(f"FASE 2 DETECTIE: {naam.upper()}")
        print(f"{'='*70}")

        # Render centrale tegel
        try:
            oud_b64, nieuw_b64, px0, py0 = _render_centrum_tegel(oud_path, nieuw_path)
        except Exception as e:
            print(f"  FOUT bij renderen: {e}")
            continue

        wt_tekst = _wandtypes_tekst(res1)
        n_wt     = len(res1["runs"][0]["wandtypes"])
        print(f"  Tegel: px_x0={px0}, py_y0={py0} | Renvooi-context: {n_wt} wandtypes")
        print(f"  Crops opgeslagen in {TMP_DIR}/")

        # 3 parallelle calls
        runs = list(await asyncio.gather(*[
            _call_detectie(client, oud_b64, nieuw_b64, wt_tekst, r)
            for r in range(1, 4)
        ]))
        runs.sort(key=lambda x: x["run"])

        _print_fase2(naam, runs)


def _print_fase2(naam: str, runs: list[dict]):
    breedte = 38
    header = "  " + "  ".join(
        f"{'Run %d (%d wijz.)' % (r['run'], len(r['wijzigingen'])):<{breedte}}"
        for r in runs
    )
    print(header)
    print("  " + ("  " + "-" * breedte) * 3)

    max_len = max(len(r["wijzigingen"]) for r in runs)
    for i in range(max_len):
        def fmt(r):
            if i >= len(r["wijzigingen"]):
                return ""
            w = r["wijzigingen"][i]
            return f"{w.get('type','?')} | {w.get('wandtype','?')[:20]}"
        print("  " + "  ".join(f"{fmt(r):<{breedte}}" for r in runs))

    # Consistentie
    def run_set(r):
        return {(w.get("type",""), w.get("wandtype","").lower()) for w in r["wijzigingen"]}

    sets = [run_set(r) for r in runs]
    in_alle = sets[0] & sets[1] & sets[2]
    in_2    = ((sets[0] & sets[1]) | (sets[0] & sets[2]) | (sets[1] & sets[2])) - in_alle
    alleen_1 = (sets[0] | sets[1] | sets[2]) - in_alle - in_2

    print(f"\n  In alle 3 runs ({len(in_alle)}):")
    for t, wt in sorted(in_alle):
        print(f"    OK {t} | {wt}")
    print(f"  In 2 van 3 runs: {len(in_2)}")
    print(f"  Alleen in 1 run: {len(alleen_1)}")

    tokens = [r["tokens"] for r in runs]
    print(f"  Tokens: run1={tokens[0]}, run2={tokens[1]}, run3={tokens[2]}")

    if any(r["fout"] for r in runs):
        for r in runs:
            if r["fout"]:
                print(f"  FOUT run {r['run']}: {r['fout']}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

async def main():
    t0 = time.time()

    print("=" * 70)
    print("FASE 1 — Renvooi consistentie test (9 Vision-calls parallel)")
    print("=" * 70)

    fase1_res = await fase1()

    for naam in sorted(fase1_res):
        _print_fase1(naam, fase1_res[naam])

    go_count = sum(1 for r in fase1_res.values() if r["go"])
    print(f"\n{'='*70}")
    print(f"FASE 1 SAMENVATTING: {go_count}/{len(fase1_res)} tekeningen -> GO")

    if go_count == 0:
        print("-> STOP: geen enkele tekening haalt 80% consistentie.")
        print(f"Totale tijd: {time.time() - t0:.1f}s")
        return

    print(f"\n{'='*70}")
    print("FASE 2 — Wand-detectie met renvooi-context (9 Vision-calls parallel)")
    print("=" * 70)

    await fase2(fase1_res)

    elapsed = time.time() - t0
    print(f"\n{'='*70}")
    print(f"KLAAR — Totale tijd: {elapsed:.1f}s")
    print()
    print("EINDCONCLUSIE:")

    fase2_beschikbaar = {
        naam for naam, (o, n) in PDF_PAREN.items() if o.exists() and n.exists()
    }
    for naam in sorted(fase1_res):
        res = fase1_res[naam]
        pct = res["pct"]
        if not res["go"]:
            status = f"C — renvooi instabiel ({pct:.0f}% < 80%)"
        elif naam not in fase2_beschikbaar:
            status = f"— fase 1 GO ({pct:.0f}%), fase 2 overgeslagen (PDF niet gevonden)"
        else:
            status = f"zie fase 2 output ({pct:.0f}% renvooi-consistentie)"
        print(f"  {naam:<10}: {status}")


if __name__ == "__main__":
    asyncio.run(main())
