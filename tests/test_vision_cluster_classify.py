"""
Vision-validatie op cluster-niveau voor data/helling.

Pipeline:
1. Re-run pixel-diff + BFS-clustering + kleur-matching (zoals test_kleur_dikte_matching).
2. Filter clusters waar kleur-matching "onbekend" gaf.
3. Top 5 grootste daarvan: render oud-crop + nieuw-crop op 200 DPI.
4. Eén Vision-call per cluster (claude-sonnet-4-5) met beide crops + namenlijst
   uit signatures.json.
5. Vergelijk met ground truth (Gibo zwaar dikte-wijziging + Hardschuimisolatie).

Standalone — geen integratie in app/.
"""
from __future__ import annotations

import base64
import io
import json
import math
import os
import re
import sys
import time
from collections import defaultdict
from pathlib import Path

import fitz

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).parent))

from app.diff_engine import (
    strip_annotations,
    _extract_lijnen,
    _vergelijk_fills,
    _vergelijk_lijnen,
)
from test_kleur_dikte_matching import (
    bfs_cluster,
    cluster_bbox,
    to_255,
    best_kleur_match,
)

PROJECT = "helling"
PAGINA = 0
MODEL = "claude-sonnet-4-5"
MODEL_FALLBACK = "claude-sonnet-4-6"
DPI = 200
PADDING_PT = 20
TOP_N = 5
CLUSTER_AFSTAND_PT = 80.0

PDF_OUD = ROOT / "data" / PROJECT / "oud.pdf"
PDF_NIEUW = ROOT / "data" / PROJECT / "nieuw.pdf"
SIGS_PATH = ROOT / "references" / PROJECT / "signatures.json"


def _load_env(env_path: Path) -> None:
    if not env_path.exists():
        return
    for line in env_path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        k, _, v = line.partition("=")
        os.environ.setdefault(k.strip(), v.strip().strip('"').strip("'"))


def render_crop_b64(page: fitz.Page, bbox_pt: list[float], padding: float, dpi: int) -> str:
    x0, y0, x1, y1 = bbox_pt
    clip = fitz.Rect(
        max(0.0, x0 - padding),
        max(0.0, y0 - padding),
        min(page.rect.width, x1 + padding),
        min(page.rect.height, y1 + padding),
    ) & page.rect
    if clip.is_empty:
        clip = page.rect
    mat = fitz.Matrix(dpi / 72, dpi / 72)
    pix = page.get_pixmap(matrix=mat, clip=clip, colorspace=fitz.csRGB)
    return base64.b64encode(pix.tobytes("png")).decode()


def vision_classify(client, b64_oud: str, b64_nieuw: str, naam_dikte_lijst: str, model: str):
    prompt = (
        "Hier zijn twee crops van dezelfde zone uit een Nederlandse afbouwtekening:\n"
        "- eerste afbeelding: oude versie\n"
        "- tweede afbeelding: nieuwe versie\n\n"
        "Identificeer welk wandtype zichtbaar is en of er een zichtbare wijziging is "
        "tussen oud en nieuw.\n\n"
        f"Bekende wandtypes uit het renvooi:\n{naam_dikte_lijst}\n\n"
        "Antwoord UITSLUITEND met geldige JSON (geen prose, geen markdown-fences):\n"
        '{"wandtype": "...", "wijziging": "toegevoegd|verdwenen|gewijzigd|geen", "redenering": "..."}\n\n'
        "wandtype: exact één van de bekende namen, of \"onbekend\".\n"
        "wijziging: alleen op basis van zichtbaar verschil tussen oud en nieuw.\n"
        "redenering: max 1 korte zin."
    )
    return client.messages.create(
        model=model,
        max_tokens=512,
        messages=[{
            "role": "user",
            "content": [
                {"type": "text", "text": "Oude versie:"},
                {"type": "image", "source": {"type": "base64", "media_type": "image/png", "data": b64_oud}},
                {"type": "text", "text": "Nieuwe versie:"},
                {"type": "image", "source": {"type": "base64", "media_type": "image/png", "data": b64_nieuw}},
                {"type": "text", "text": prompt},
            ],
        }],
    )


def parse_vision_json(text: str) -> dict:
    text = re.sub(r"^```(?:json)?\s*|\s*```$", "", text.strip(), flags=re.MULTILINE)
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        m = re.search(r"\{.*\}", text, re.DOTALL)
        if not m:
            return {"wandtype": "parse_error", "wijziging": "geen", "redenering": text[:120]}
        return json.loads(m.group(0))


def main():
    _load_env(ROOT / ".env")
    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        print("FAIL: ANTHROPIC_API_KEY niet gezet")
        return 2

    if not (PDF_OUD.exists() and PDF_NIEUW.exists() and SIGS_PATH.exists()):
        print("FAIL: data PDFs of signatures.json ontbreken")
        return 1

    sigs = json.loads(SIGS_PATH.read_text(encoding="utf-8"))["wandtypes"]
    naam_dikte = "\n".join(
        f"- {s['naam']}" + (f" ({s['dikte_mm']}mm)" if s.get('dikte_mm') else "")
        for s in sigs
    )

    print(f"[load] {len(sigs)} signatures, {PDF_OUD.name} vs {PDF_NIEUW.name}")
    t_start = time.time()

    # Pipeline (zelfde als test_kleur_dikte_matching) — kort versie
    oud_clean = strip_annotations(str(PDF_OUD))
    nieuw_clean = strip_annotations(str(PDF_NIEUW))
    oud_doc = fitz.open(oud_clean)
    nieuw_doc = fitz.open(nieuw_clean)

    try:
        oud_page = oud_doc[PAGINA]
        nieuw_page = nieuw_doc[PAGINA]
        oud_items = _extract_lijnen(oud_page)
        nieuw_items = _extract_lijnen(nieuw_page)

        fill_gewijzigd, fills_toegevoegd, fills_verdwenen = _vergelijk_fills(oud_items, nieuw_items)
        _, _, lijnen_toegevoegd, lijnen_verdwenen = _vergelijk_lijnen(oud_items, nieuw_items)

        wijzigingen = []
        for f in fills_toegevoegd:
            wijzigingen.append({"kind": "fill_toegevoegd",
                                "pos": tuple(f["pos"]),
                                "bbox": list(f.get("bbox") or [f["pos"][0], f["pos"][1], f["pos"][0]+1, f["pos"][1]+1]),
                                "rgb255": to_255(f["rgb"])})
        for f in fills_verdwenen:
            wijzigingen.append({"kind": "fill_verdwenen",
                                "pos": tuple(f["pos"]),
                                "bbox": list(f.get("bbox") or [f["pos"][0], f["pos"][1], f["pos"][0]+1, f["pos"][1]+1]),
                                "rgb255": to_255(f["rgb"])})
        for f in fill_gewijzigd:
            wijzigingen.append({"kind": "fill_gewijzigd",
                                "pos": tuple(f["pos"]),
                                "bbox": list(f.get("bbox") or [f["pos"][0], f["pos"][1], f["pos"][0]+1, f["pos"][1]+1]),
                                "rgb255_oud": to_255(f["oud_rgb"]),
                                "rgb255_nieuw": to_255(f["nieuw_rgb"])})
        for l in lijnen_toegevoegd:
            wijzigingen.append({"kind": "lijn_toegevoegd",
                                "pos": tuple(l["van"]),
                                "bbox": [min(l["van"][0], l["naar"][0]), min(l["van"][1], l["naar"][1]),
                                         max(l["van"][0], l["naar"][0]), max(l["van"][1], l["naar"][1])]})
        for l in lijnen_verdwenen:
            wijzigingen.append({"kind": "lijn_verdwenen",
                                "pos": tuple(l["van"]),
                                "bbox": [min(l["van"][0], l["naar"][0]), min(l["van"][1], l["naar"][1]),
                                         max(l["van"][0], l["naar"][0]), max(l["van"][1], l["naar"][1])]})

        clusters = bfs_cluster(wijzigingen, CLUSTER_AFSTAND_PT)
        clusters = [c for c in clusters if len(c) >= 2]
        clusters.sort(key=len, reverse=True)
        print(f"[pipeline] {len(wijzigingen)} wijzigingen -> {len(clusters)} clusters")

        # Bepaal welke clusters "onbekend" zijn (geen kleur-match binnen drempel)
        onbekend_clusters = []
        for cid, idx_list in enumerate(clusters):
            heeft_match = False
            for i in idx_list:
                w = wijzigingen[i]
                rgbs = []
                if w["kind"] in ("fill_toegevoegd", "fill_verdwenen"):
                    rgbs = [w["rgb255"]]
                elif w["kind"] == "fill_gewijzigd":
                    rgbs = [w["rgb255_oud"], w["rgb255_nieuw"]]
                for r in rgbs:
                    sig, d = best_kleur_match(r, sigs)
                    if sig is not None:
                        heeft_match = True
                        break
                if heeft_match:
                    break
            if not heeft_match:
                onbekend_clusters.append((cid, idx_list))

        print(f"[filter] {len(onbekend_clusters)} clusters zijn 'onbekend' (geen kleur-match)")

        top = onbekend_clusters[:TOP_N]
        if not top:
            print("[skip] geen onbekend-clusters")
            return 0

        # Vision per cluster
        import anthropic
        client = anthropic.Anthropic(api_key=api_key)

        resultaten = []
        total_in = 0
        total_out = 0
        total_cost = 0.0
        used_model = MODEL

        print(f"\n[vision] top {len(top)} onbekend-clusters @ {DPI} DPI...")
        for cid, idx_list in top:
            bbox = cluster_bbox(wijzigingen, idx_list)
            t0 = time.time()
            b64_oud = render_crop_b64(oud_page, bbox, PADDING_PT, DPI)
            b64_nieuw = render_crop_b64(nieuw_page, bbox, PADDING_PT, DPI)

            try:
                resp = vision_classify(client, b64_oud, b64_nieuw, naam_dikte, MODEL)
            except Exception as e:
                print(f"  c{cid}: {MODEL} faalde ({e!r}), fallback naar {MODEL_FALLBACK}")
                resp = vision_classify(client, b64_oud, b64_nieuw, naam_dikte, MODEL_FALLBACK)
                used_model = MODEL_FALLBACK

            text = "".join(b.text for b in resp.content if b.type == "text").strip()
            verdict = parse_vision_json(text)
            cost = (resp.usage.input_tokens * 3 + resp.usage.output_tokens * 15) / 1_000_000
            total_in += resp.usage.input_tokens
            total_out += resp.usage.output_tokens
            total_cost += cost

            resultaten.append({
                "cluster_id": cid,
                "n_items": len(idx_list),
                "bbox": [round(v, 1) for v in bbox],
                "wandtype": verdict.get("wandtype", "?"),
                "wijziging": verdict.get("wijziging", "?"),
                "redenering": verdict.get("redenering", "")[:160],
                "duur_s": round(time.time() - t0, 1),
                "cost": round(cost, 4),
            })
            print(f"  c{cid:>3}  n={len(idx_list):>4}  bbox=[{bbox[0]:.0f},{bbox[1]:.0f},{bbox[2]:.0f},{bbox[3]:.0f}]  "
                  f"-> {verdict.get('wandtype','?'):<35}  {verdict.get('wijziging','?'):<10}  ${cost:.4f}  {time.time()-t0:.1f}s")

        # Print per-cluster output
        print(f"\n--- detail ---")
        for r in resultaten:
            print(f"  c{r['cluster_id']:>3}  n={r['n_items']:>4}  wandtype={r['wandtype']!r}")
            print(f"        wijziging={r['wijziging']!r}")
            print(f"        redenering: {r['redenering']}")

        # GT-validatie
        print(f"\n=== GROUND TRUTH VERGELIJKING ===")
        gibo = [r for r in resultaten if "gibo zwaar" in r["wandtype"].lower()]
        hard = [r for r in resultaten if "hardschuimisolatie" in r["wandtype"].lower()]

        print(f"  GT-1: Gibo zwaar dikte 70mm -> 100mm")
        if gibo:
            for r in gibo:
                print(f"        WEL — c{r['cluster_id']} (n={r['n_items']}, wijziging={r['wijziging']})")
        else:
            print(f"        NIET — geen 'Gibo zwaar' onder de top {TOP_N} onbekend-clusters")

        print(f"  GT-2: Hardschuimisolatie toegevoegd")
        if hard:
            for r in hard:
                print(f"        WEL — c{r['cluster_id']} (n={r['n_items']}, wijziging={r['wijziging']})")
        else:
            print(f"        NIET — geen 'hardschuimisolatie' onder de top {TOP_N} onbekend-clusters")

        gevonden = (1 if gibo else 0) + (1 if hard else 0)

        # Kosten + tijd
        print(f"\n=== KOSTEN + TIJD ===")
        print(f"  Vision-calls:  {len(resultaten)}  ({used_model})")
        print(f"  Tokens:        in={total_in}  out={total_out}")
        print(f"  Kosten:        ${total_cost:.4f}")
        print(f"  Totale tijd:   {time.time() - t_start:.1f}s")
        print(f"  Recall (GT):   {gevonden}/2")
        return 0
    finally:
        oud_doc.close()
        nieuw_doc.close()
        for p in (oud_clean, nieuw_clean):
            try:
                os.unlink(p)
            except OSError:
                pass


if __name__ == "__main__":
    sys.exit(main())
