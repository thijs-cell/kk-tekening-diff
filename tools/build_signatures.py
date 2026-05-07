"""
Bouwt references/<project>/signatures.json + templates/<naam>.png crops.

Per renvooi-afbeelding: één Vision-call (claude-sonnet-4-5) voor naam +
arcering_bbox per wandtype. Lokaal: dominante_rgb uit crop, dikte_mm uit naam.

Helling: oud + nieuw renvooi, unie van wandtypes (sandwichpaneel alleen in nieuw).

Caching: Vision-respons opgeslagen in references/<project>/_vision_cache_<hash>.json.
Tweede run = gratis (geen API-call).

Gebruik:
    python tools/build_signatures.py             # alle projecten
    python tools/build_signatures.py helling     # specifiek project
"""
from __future__ import annotations

import base64
import hashlib
import json
import os
import re
import sys
import time
from pathlib import Path

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

REFERENCES = ROOT / "references"
MODEL = "claude-sonnet-4-5"
MODEL_FALLBACK = "claude-sonnet-4-6"

VISION_PROMPT = """\
Hier zie je een renvooi-tabel uit een Nederlandse bouwtekening (PNG, breedte=__W__px, hoogte=__H__px).

Identificeer alle wandtype-rijen. Voor elke wandtype geef terug:
- naam: letterlijke naam zoals in renvooi (bv. "Gibo zwaar 70mm", "kalkzandsteen 120mm", "hardschuimisolatie", "sandwichpaneel")
- arcering_bbox: [x, y, w, h] in PIXEL-coördinaten van DEZE afbeelding, voor het rechthoekje met de arcering/kleurvulling (NIET het tekstlabel, NIET de hele rij — alleen het visuele symbool zelf)

Negeer (geen wandtype):
- Peilmaat (1200+, etc)
- Brandwerendheid markeringen (30 minuten, 60 minuten)
- Geluidsisolatie (Rw, Rw,p)
- Pijlen, deurmarkeringen, vluchtwegaanduiding
- Zelfsluitende deur, entree
- Architect-aantekeningen of D-pijlen

Antwoord UITSLUITEND met geldige JSON, geen prose, geen markdown-code-fences:
{"wandtypes": [{"naam": "...", "arcering_bbox": [x, y, w, h]}, ...]}
"""


def _safe_filename(naam: str) -> str:
    s = re.sub(r"[^\w\s-]", "", naam, flags=re.UNICODE).strip().lower()
    s = re.sub(r"[\s_]+", "_", s)
    return s or "wandtype"


def _parse_dikte_mm(naam: str) -> int | None:
    m = re.search(r"(\d+)\s*mm", naam, flags=re.IGNORECASE)
    return int(m.group(1)) if m else None


def _dedup_key(naam: str) -> str:
    return re.sub(r"\s+", " ", naam.strip().lower())


def _dominante_rgb(image: Image.Image, bbox: list[int]) -> list[int]:
    x, y, w, h = [int(v) for v in bbox]
    crop = image.crop((x, y, x + w, y + h)).convert("RGB")
    arr = np.array(crop).reshape(-1, 3)
    if len(arr) == 0:
        return [255, 255, 255]
    near_white = (arr[:, 0] > 240) & (arr[:, 1] > 240) & (arr[:, 2] > 240)
    non_white = arr[~near_white]
    if len(non_white) >= 5:
        med = np.median(non_white, axis=0)
    else:
        med = np.median(arr, axis=0)
    return [int(med[0]), int(med[1]), int(med[2])]


def _hash_image(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()[:12]


def _call_vision(png_path: Path, api_key: str) -> dict:
    """Eén Vision-call per renvooi met lokale cache."""
    proj_dir = png_path.parent
    h = _hash_image(png_path)
    cache_path = proj_dir / f"_vision_cache_{h}.json"

    if cache_path.exists():
        print(f"  [cache hit] {cache_path.name}")
        return json.loads(cache_path.read_text(encoding="utf-8"))

    import anthropic
    import io

    img = Image.open(png_path).convert("RGB")
    W, H = img.size
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    img.close()
    png_bytes = buf.getvalue()

    b64 = base64.b64encode(png_bytes).decode()
    client = anthropic.Anthropic(api_key=api_key)

    prompt = VISION_PROMPT.replace("__W__", str(W)).replace("__H__", str(H))
    used_model = MODEL
    t0 = time.time()
    try:
        resp = client.messages.create(
            model=MODEL,
            max_tokens=2048,
            messages=[{
                "role": "user",
                "content": [
                    {"type": "image", "source": {
                        "type": "base64", "media_type": "image/png", "data": b64,
                    }},
                    {"type": "text", "text": prompt},
                ],
            }],
        )
    except Exception as e:
        print(f"  [WARN] {MODEL} faalde: {e!r} — fallback naar {MODEL_FALLBACK}")
        resp = client.messages.create(
            model=MODEL_FALLBACK,
            max_tokens=2048,
            messages=[{
                "role": "user",
                "content": [
                    {"type": "image", "source": {
                        "type": "base64", "media_type": "image/png", "data": b64,
                    }},
                    {"type": "text", "text": prompt},
                ],
            }],
        )
        used_model = MODEL_FALLBACK
    t1 = time.time()

    text = "".join(b.text for b in resp.content if b.type == "text").strip()
    text = re.sub(r"^```(?:json)?\s*|\s*```$", "", text.strip(), flags=re.MULTILINE)
    try:
        data = json.loads(text)
    except json.JSONDecodeError:
        m = re.search(r"\{.*\}", text, re.DOTALL)
        if not m:
            raise RuntimeError(f"Geen JSON in Vision-respons: {text[:200]!r}")
        data = json.loads(m.group(0))

    cost = (resp.usage.input_tokens * 3 + resp.usage.output_tokens * 15) / 1_000_000
    payload = {
        "model": used_model,
        "input_tokens": resp.usage.input_tokens,
        "output_tokens": resp.usage.output_tokens,
        "cost_usd": round(cost, 4),
        "duur_s": round(t1 - t0, 2),
        "image_size": [W, H],
        "wandtypes": data.get("wandtypes", []),
    }
    cache_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"  [vision] {used_model}  in={resp.usage.input_tokens} out={resp.usage.output_tokens}  ${cost:.4f}  {t1-t0:.1f}s")
    return payload


def _verzamel_renvooien(project_dir: Path) -> list[Path]:
    cands = []
    for nm in ("renvooi.png", "renvooi_oud.png", "renvooi_nieuw.png"):
        p = project_dir / nm
        if p.exists():
            cands.append(p)
    return cands


def build_for_project(project_dir: Path, api_key: str) -> dict:
    project = project_dir.name
    pngs = _verzamel_renvooien(project_dir)
    if not pngs:
        return {"project": project, "skipped": True, "reden": "geen renvooi-PNG"}

    templates_dir = project_dir / "templates"
    templates_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n=== {project} ===")
    print(f"  renvooien: {[p.name for p in pngs]}")

    seen: dict[str, dict] = {}
    total_cost = 0.0
    total_dur = 0.0
    total_in = 0
    total_out = 0

    for png in pngs:
        vis = _call_vision(png, api_key)
        total_cost += vis.get("cost_usd", 0)
        total_dur += vis.get("duur_s", 0)
        total_in += vis.get("input_tokens", 0)
        total_out += vis.get("output_tokens", 0)

        img = Image.open(png).convert("RGB")
        for wt in vis.get("wandtypes", []):
            naam = (wt.get("naam") or "").strip()
            bbox = wt.get("arcering_bbox")
            if not naam or not bbox or len(bbox) != 4:
                continue

            key = _dedup_key(naam)
            rgb = _dominante_rgb(img, bbox)
            dikte = _parse_dikte_mm(naam)

            x, y, w, h = [int(v) for v in bbox]
            template_path = templates_dir / f"{_safe_filename(naam)}.png"
            crop = img.crop((x, y, x + w, y + h))
            crop.save(template_path)

            if key in seen:
                continue
            seen[key] = {
                "naam": naam,
                "dikte_mm": dikte,
                "dominante_rgb": rgb,
                "arcering_bbox": [x, y, w, h],
                "bron_renvooi": png.name,
                "template": str(template_path.relative_to(project_dir)).replace("\\", "/"),
            }
        img.close()

    sigs_path = project_dir / "signatures.json"
    sigs_path.write_text(
        json.dumps({"wandtypes": list(seen.values())}, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    return {
        "project": project,
        "skipped": False,
        "wandtypes_gevonden": len(seen),
        "renvooien_verwerkt": len(pngs),
        "cost_usd": round(total_cost, 4),
        "duur_s": round(total_dur, 2),
        "input_tokens": total_in,
        "output_tokens": total_out,
        "signatures_path": str(sigs_path.relative_to(ROOT)).replace("\\", "/"),
    }


def _load_env(env_path: Path) -> None:
    if not env_path.exists():
        return
    for line in env_path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        k, _, v = line.partition("=")
        os.environ.setdefault(k.strip(), v.strip().strip('"').strip("'"))


def main(argv: list[str]) -> int:
    _load_env(ROOT / ".env")
    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        print("ERROR: ANTHROPIC_API_KEY niet gezet")
        return 2

    if argv:
        targets = [REFERENCES / p for p in argv]
    else:
        targets = sorted([d for d in REFERENCES.iterdir() if d.is_dir() and not d.name.endswith("_ground_truth")])

    rapporten = []
    for proj_dir in targets:
        if not proj_dir.exists():
            print(f"[skip] {proj_dir.name}: bestaat niet")
            continue
        try:
            rap = build_for_project(proj_dir, api_key)
            rapporten.append(rap)
        except Exception as e:
            import traceback
            traceback.print_exc()
            rapporten.append({"project": proj_dir.name, "error": str(e)})

    print("\n=== samenvatting ===")
    for r in rapporten:
        if r.get("skipped"):
            print(f"  {r['project']:<25}  SKIP   ({r['reden']})")
        elif "error" in r:
            print(f"  {r['project']:<25}  ERROR  {r['error']}")
        else:
            print(f"  {r['project']:<25}  {r['wandtypes_gevonden']:>3} wandtypes  ${r['cost_usd']:.4f}  {r['duur_s']:.1f}s")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
