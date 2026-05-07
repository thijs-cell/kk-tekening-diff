"""
Eenmalige extractie: alle inhoud uit elk renvooi via Vision.
Een call per renvooi, complete output, opslaan als vision_complete.json.

Gebruik: python test_renvooi_extract.py
"""

import asyncio
import base64
import json
import os
from pathlib import Path

import anthropic

SCRIPT_DIR = Path(__file__).parent
REF_DIR    = SCRIPT_DIR / "references"

RENVOOIEN = {
    "muiden":  REF_DIR / "muiden"  / "renvooi.png",
    "bd_n101": REF_DIR / "bd_n101" / "renvooi.png",
    "5102":    REF_DIR / "5102"    / "renvooi.png",
}

MODEL      = "claude-sonnet-4-5"
MAX_TOKENS = 4096

PROMPT = (
    "Lees alles uit dit renvooi. Geef per item:\n"
    "- naam/beschrijving zoals in renvooi\n"
    "- visuele weergave (kleur, arcering, symbool)\n\n"
    'JSON: {"items": [{"naam": string, "visuele_weergave": string}]}'
)


def _laad_b64(pad: Path) -> tuple[str, str]:
    data = pad.read_bytes()
    if data[:3] == b"\xff\xd8\xff":
        media = "image/jpeg"
    elif data[:4] == b"\x89PNG":
        media = "image/png"
    else:
        media = "image/jpeg"
    return base64.standard_b64encode(data).decode(), media


async def _call(client: anthropic.AsyncAnthropic, naam: str, pad: Path) -> dict:
    b64, media = _laad_b64(pad)
    resp = await client.messages.create(
        model=MODEL,
        max_tokens=MAX_TOKENS,
        messages=[{
            "role": "user",
            "content": [
                {"type": "image", "source": {"type": "base64", "media_type": media, "data": b64}},
                {"type": "text", "text": PROMPT},
            ],
        }],
    )
    tekst = "".join(b.text for b in resp.content if hasattr(b, "text"))

    # JSON parsen
    tekst_clean = tekst.strip()
    if "```" in tekst_clean:
        for deel in tekst_clean.split("```"):
            deel = deel.lstrip("json").strip()
            if deel.startswith("{"):
                tekst_clean = deel
                break
    try:
        data = json.loads(tekst_clean)
    except Exception:
        start = tekst_clean.find("{")
        eind  = tekst_clean.rfind("}") + 1
        data  = json.loads(tekst_clean[start:eind]) if start >= 0 and eind > start else {}

    return {
        "naam":   naam,
        "items":  data.get("items", []),
        "tokens": resp.usage.input_tokens + resp.usage.output_tokens,
        "raw":    tekst,
    }


async def main():
    client = anthropic.AsyncAnthropic()

    taken = [
        _call(client, naam, pad)
        for naam, pad in RENVOOIEN.items()
        if pad.exists()
    ]
    resultaten = await asyncio.gather(*taken)

    for res in resultaten:
        naam  = res["naam"]
        items = res["items"]

        print(f"\n{'='*60}")
        print(f"RENVOOI: {naam.upper()}  ({len(items)} items, {res['tokens']} tokens)")
        print(f"{'='*60}")
        for i, item in enumerate(items, 1):
            print(f"  {i:>2}. {item.get('naam','?')}")
            print(f"      {item.get('visuele_weergave','?')}")

        # Opslaan
        uit_pad = REF_DIR / naam / "vision_complete.json"
        with open(uit_pad, "w", encoding="utf-8") as f:
            json.dump({"items": items}, f, ensure_ascii=False, indent=2)
        print(f"\n  Opgeslagen: {uit_pad}")


if __name__ == "__main__":
    asyncio.run(main())
