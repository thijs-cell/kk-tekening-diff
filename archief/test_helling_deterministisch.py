"""
Deterministische wanddetectie — 56 de Helling pagina 1.
Stap 1: Vision leest renvooien -> signatures.json (RGB + dikte per wandtype)
Stap 2: Pixel diff + BFS + kleur-matching op fills + dikte-detectie via maattekst
Stap 3: Rapport + ground truth check + top-10 crops

Gebruik: python test_helling_deterministisch.py
"""
import asyncio, base64, json, math, re, sys, time
from collections import defaultdict
from pathlib import Path

import anthropic, fitz

SCRIPT_DIR = Path(__file__).parent
sys.path.insert(0, str(SCRIPT_DIR))
from app.diff_engine import _extract_lijnen, _vergelijk_lijnen, _vergelijk_fills, strip_annotations

MVP_DIR  = SCRIPT_DIR.parent / "Karregat & Koning MVP"
REF_DIR  = SCRIPT_DIR / "references" / "helling"
SIG_PAD  = REF_DIR / "signatures.json"
UIT_DIR  = Path("/tmp/helling_detecties")
MODEL    = "claude-sonnet-4-5"
PAGINA   = 0

OUD_PDF   = MVP_DIR / "56 de Helling - plattegronden gibowanden - oude tekening (5).pdf"
NIEUW_PDF = MVP_DIR / "56 de Helling - plattegronden gibowanden - Nieuwe tekening (2).pdf"

CLUSTER_AFSTAND = 80.0
MIN_ITEMS       = 3
MIN_BBOX_DIM    = 50.0
CROP_PAD        = 80.0
KLEUR_DREMPEL   = 55.0   # euclidische afstand in 0-255 ruimte
WALL_DIKTES     = {30,40,50,60,70,80,90,100,110,120,130,140,150,175,200,250}
DIKTE_RE        = re.compile(r'(?<!\d)(\d{2,3})(?!\d)')

# ===========================================================================
# STAP 1 — Renvooi -> signatures.json
# ===========================================================================
_PROMPT_SIG = """\
Lees dit renvooi van een Nederlandse afbouwtekening.
Per wandtype geef je:
- naam: exact zoals in het renvooi
- dikte_mm: getal in mm uit de naam (bijv. "Gibo 70mm" -> 70), null als niet vermeld
- fill_rgb: dominante VULKLEUR van het arceringsblokje als [R,G,B] integers 0-255.
  Wit/leeg blokje -> [255,255,255]. Zwart -> [0,0,0].
- heeft_arcering: true als het blokje een arcering/hatchpatroon heeft (naast kleur)

Negeer legenda-items die GEEN wanden zijn (deuren, ramen, pijlen, teksten).

JSON: {"wandtypes": [{"naam": str, "dikte_mm": int|null, "fill_rgb": [R,G,B], "heeft_arcering": bool}]}
"""

def _laad_b64(pad: Path) -> tuple[str, str]:
    data = pad.read_bytes()
    media = "image/jpeg" if data[:3] == b"\xff\xd8\xff" else "image/png"
    return base64.standard_b64encode(data).decode(), media

async def _vision_renvooi(client, naam: str, pad: Path) -> list[dict]:
    b64, media = _laad_b64(pad)
    resp = await client.messages.create(
        model=MODEL, max_tokens=2048,
        messages=[{"role": "user", "content": [
            {"type": "image", "source": {"type": "base64", "media_type": media, "data": b64}},
            {"type": "text",  "text": _PROMPT_SIG},
        ]}],
    )
    tekst = "".join(b.text for b in resp.content if hasattr(b, "text")).strip()
    if "```" in tekst:
        for deel in tekst.split("```"):
            deel = deel.lstrip("json").strip()
            if deel.startswith("{"):
                tekst = deel; break
    try:
        data = json.loads(tekst)
    except Exception:
        s = tekst.find("{"); e = tekst.rfind("}") + 1
        data = json.loads(tekst[s:e]) if s >= 0 and e > s else {}
    types = data.get("wandtypes", [])
    tokens = resp.usage.input_tokens + resp.usage.output_tokens
    print(f"  {naam}: {len(types)} types, {tokens} tokens")
    return types

async def stap1_signatures() -> list[dict]:
    print("\n" + "=" * 64)
    print("STAP 1 -- Renvooi signatures (Vision)")
    print("=" * 64)
    client = anthropic.AsyncAnthropic()
    oud_types, nieuw_types = await asyncio.gather(
        _vision_renvooi(client, "oud",  REF_DIR / "renvooi_oud.png"),
        _vision_renvooi(client, "nieuw", REF_DIR / "renvooi_nieuw.png"),
    )
    gecombineerd: dict[str, dict] = {}
    for wt in oud_types + nieuw_types:
        gecombineerd[wt["naam"].lower().strip()] = wt
    sigs = list(gecombineerd.values())
    with open(SIG_PAD, "w", encoding="utf-8") as f:
        json.dump({"wandtypes": sigs}, f, ensure_ascii=False, indent=2)
    print(f"  Opgeslagen: {SIG_PAD} ({len(sigs)} types)")
    for s in sigs:
        print(f"    {s['naam']:<40} rgb={s.get('fill_rgb')}  dikte={s.get('dikte_mm')}mm")
    return sigs

# ===========================================================================
# STAP 2 — Detectie
# ===========================================================================
def _kleur_dist(rgb_float: tuple, sig_rgb: list) -> float:
    return math.sqrt(sum((v * 255 - c) ** 2 for v, c in zip(rgb_float, sig_rgb)))

def _match_kleur(rgb_float: tuple, sigs: list[dict]) -> tuple[str | None, float]:
    beste, dist = None, float("inf")
    for s in sigs:
        sr = s.get("fill_rgb", [])
        if not sr or sr in ([255,255,255],[0,0,0]):
            continue
        d = _kleur_dist(rgb_float, sr)
        if d < dist:
            dist = d; beste = s["naam"]
    return beste, dist

def _tekst_in_bbox(page, bbox: tuple) -> list[str]:
    # get_text("words") geeft: (x0,y0,x1,y1,word,block_no,line_no,word_no)
    x0, y0, x1, y1 = bbox
    out = []
    for w in page.get_text("words"):
        cx, cy = (w[0]+w[2])/2, (w[1]+w[3])/2
        if x0 <= cx <= x1 and y0 <= cy <= y1:
            out.append(w[4].strip())
    return out

def _dikte_check(tekst_oud: list[str], tekst_nieuw: list[str]) -> str | None:
    def diktes(teksten):
        result = set()
        for t in teksten:
            for m in DIKTE_RE.findall(t):
                if int(m) in WALL_DIKTES:
                    result.add(int(m))
        return result
    d_oud = diktes(tekst_oud)
    d_nieuw = diktes(tekst_nieuw)
    verdwenen = d_oud - d_nieuw
    toegevoegd = d_nieuw - d_oud
    if verdwenen and toegevoegd:
        return f"{'/'.join(str(v) for v in sorted(verdwenen))}mm -> {'/'.join(str(v) for v in sorted(toegevoegd))}mm"
    return None

def stap2_detecteer(sigs: list[dict]) -> list[dict]:
    print("\n" + "=" * 64)
    print("STAP 2 -- Kleur + dikte detectie (hele pagina 1)")
    print("=" * 64)

    oud_c   = strip_annotations(str(OUD_PDF))
    nieuw_c = strip_annotations(str(NIEUW_PDF))
    d_oud   = fitz.open(oud_c);  d_nieuw = fitz.open(nieuw_c)
    p_oud   = d_oud[PAGINA];     p_nieuw = d_nieuw[PAGINA]

    oi = _extract_lijnen(p_oud);  ni = _extract_lijnen(p_nieuw)
    _, _, lt, lv = _vergelijk_lijnen(oi, ni)
    _, ft, fv    = _vergelijk_fills(oi, ni)
    print(f"  lt={len(lt)}  lv={len(lv)}  ft={len(ft)}  fv={len(fv)}")
    d_oud.close(); d_nieuw.close()

    alle = []
    for items, tag in [(lt,"lijn_toegevoegd"),(lv,"lijn_verdwenen"),
                       (ft,"fill_toegevoegd"),(fv,"fill_verdwenen")]:
        for it in items:
            x = ((it["van"][0]+it["naar"][0])/2) if "van" in it else it["pos"][0]
            y = ((it["van"][1]+it["naar"][1])/2) if "van" in it else it["pos"][1]
            alle.append({**it, "_tag": tag, "_x": float(x), "_y": float(y)})

    cel = CLUSTER_AFSTAND
    raster: dict = defaultdict(list)
    for i, it in enumerate(alle):
        raster[(int(it["_x"]//cel), int(it["_y"]//cel))].append(i)

    bezoekt = [False]*len(alle)
    clusters_raw = []
    for start in range(len(alle)):
        if bezoekt[start]: continue
        groep=[start]; bezoekt[start]=True; front=[start]
        while front:
            h=front.pop(); hx,hy=alle[h]["_x"],alle[h]["_y"]
            gx,gy=int(hx//cel),int(hy//cel)
            for dx in(-1,0,1):
                for dy in(-1,0,1):
                    for j in raster.get((gx+dx,gy+dy),[]):
                        if not bezoekt[j] and math.hypot(alle[j]["_x"]-hx,alle[j]["_y"]-hy)<=cel:
                            bezoekt[j]=True; groep.append(j); front.append(j)
        xs=[alle[i]["_x"] for i in groep]; ys=[alle[i]["_y"] for i in groep]
        clusters_raw.append({"items":[alle[i] for i in groep],
                              "bbox":(min(xs),min(ys),max(xs),max(ys))})

    clusters = [c for c in clusters_raw
                if len(c["items"])>=MIN_ITEMS
                and c["bbox"][2]-c["bbox"][0]>=MIN_BBOX_DIM
                and c["bbox"][3]-c["bbox"][1]>=MIN_BBOX_DIM]
    clusters.sort(key=lambda c:-len(c["items"]))
    print(f"  Clusters na filter: {len(clusters)}")

    d_oud2  = fitz.open(str(OUD_PDF))
    d_nieuw2 = fitz.open(str(NIEUW_PDF))
    p_oud2  = d_oud2[PAGINA]; p_nieuw2 = d_nieuw2[PAGINA]

    detecties = []
    for c in clusters:
        bx0,by0,bx1,by1 = c["bbox"]
        pad = (bx0-30, by0-30, bx1+30, by1+30)

        fills = [it for it in c["items"] if "fill" in it["_tag"]]
        kleur_scores: dict = defaultdict(lambda:{"toegevoegd":0,"verdwenen":0})
        for it in fills:
            rgb = it.get("rgb")
            if not rgb: continue
            naam, dist = _match_kleur(rgb, sigs)
            if naam and dist < KLEUR_DREMPEL:
                key = "toegevoegd" if "toegevoegd" in it["_tag"] else "verdwenen"
                kleur_scores[naam][key] += 1

        tekst_oud  = _tekst_in_bbox(p_oud2,  pad)
        tekst_nieuw = _tekst_in_bbox(p_nieuw2, pad)
        dikte = _dikte_check(tekst_oud, tekst_nieuw)

        if not kleur_scores and not dikte:
            continue

        if kleur_scores:
            beste = max(kleur_scores.items(), key=lambda kv:kv[1]["toegevoegd"]+kv[1]["verdwenen"])
            wt = beste[0]
            ta, tv = beste[1]["toegevoegd"], beste[1]["verdwenen"]
            wtype = "toegevoegd" if ta>0 and tv==0 else "verdwenen" if tv>0 and ta==0 else "gewijzigd"
            conf = "hoog" if dikte else "midden"
        else:
            wt = "onbekend"; wtype = "gewijzigd"; conf = "laag"

        detecties.append({
            "bbox":      c["bbox"],
            "n":         len(c["items"]),
            "wandtype":  wt,
            "wijziging": wtype,
            "dikte":     dikte,
            "confidence":conf,
        })

    d_oud2.close(); d_nieuw2.close()
    print(f"  Detecties: {len(detecties)}")
    return detecties

# ===========================================================================
# STAP 3 — Rapport
# ===========================================================================
def stap3_rapport(detecties: list[dict]):
    print("\n" + "=" * 64)
    print("STAP 3 -- RAPPORT")
    print("=" * 64)
    UIT_DIR.mkdir(parents=True, exist_ok=True)

    volgorde = {"hoog":0,"midden":1,"laag":2}
    gesorteerd = sorted(detecties, key=lambda d:volgorde[d["confidence"]])

    print(f"\n  {'Nr':<4} {'Wandtype':<35} {'Wijz':<12} {'Dikte':<20} {'Conf':<7} Bbox")
    print("  "+"-"*100)
    for i,d in enumerate(gesorteerd):
        b = d["bbox"]
        bstr = f"({b[0]:.0f},{b[1]:.0f})-({b[2]:.0f},{b[3]:.0f})"
        print(f"  {i:<4} {d['wandtype']:<35} {d['wijziging']:<12} {d['dikte'] or '-':<20} {d['confidence']:<7} {bstr}")

    print("\n  --- Per wandtype ---")
    per_type: dict = defaultdict(lambda:defaultdict(int))
    for d in detecties:
        per_type[d["wandtype"]][d["wijziging"]] += 1
    for wt, counts in sorted(per_type.items()):
        print(f"  {wt}: " + ", ".join(f"{v}x {k}" for k,v in counts.items()))
    print(f"  Onbekend: {sum(1 for d in detecties if d['wandtype']=='onbekend')}")

    print("\n  --- Ground truth check (cluster 0 deel 4: x=3449-3891, y=608-1156) ---")
    d4 = (3449,608,3891,1156)
    in_d4 = [d for d in detecties
             if d["bbox"][0]>=d4[0]-80 and d["bbox"][2]<=d4[2]+80
             and d["bbox"][1]>=d4[1]-80 and d["bbox"][3]<=d4[3]+80]
    buiten = [d for d in detecties if d not in in_d4]

    w1 = [d for d in in_d4 if "gibo" in d["wandtype"].lower() and d["dikte"]
          and "70" in d["dikte"] and "100" in d["dikte"]]
    w2 = [d for d in in_d4 if "hardschuim" in d["wandtype"].lower() and d["wijziging"]=="toegevoegd"]

    print(f"  Detecties in deel 4     : {len(in_d4)}")
    print(f"  W1 Gibo zwaar 70->100mm : {'GEVONDEN' if w1 else 'NIET GEVONDEN'}")
    print(f"  W2 Hardschuimisolatie   : {'GEVONDEN' if w2 else 'NIET GEVONDEN'}")
    print(f"  False positives buiten  : {len(buiten)}")

    print(f"\n  --- Top 10 crops -> {UIT_DIR} ---")
    d_oud  = fitz.open(str(OUD_PDF))
    d_nieuw = fitz.open(str(NIEUW_PDF))
    mat = fitz.Matrix(200/72, 200/72)
    for i,d in enumerate(gesorteerd[:10]):
        bx0,by0,bx1,by1 = d["bbox"]
        clip = fitz.Rect(bx0-CROP_PAD, by0-CROP_PAD, bx1+CROP_PAD, by1+CROP_PAD)
        for doc,lbl in [(d_oud,"oud"),(d_nieuw,"nieuw")]:
            doc[PAGINA].get_pixmap(matrix=mat,clip=clip).save(str(UIT_DIR/f"det_{i:02d}_{lbl}.png"))
        print(f"  {i:02d}: {d['wandtype']:<35} {d['wijziging']:<12} {d['dikte'] or '-':<20} {d['confidence']}")
    d_oud.close(); d_nieuw.close()

# ===========================================================================
# Main
# ===========================================================================
async def main():
    t0 = time.time()
    if SIG_PAD.exists():
        with open(SIG_PAD, encoding="utf-8") as f:
            sigs = json.load(f)["wandtypes"]
        print(f"signatures.json geladen ({len(sigs)} types)")
        for s in sigs:
            print(f"  {s['naam']:<40} rgb={s.get('fill_rgb')}  dikte={s.get('dikte_mm')}mm")
    else:
        sigs = await stap1_signatures()
    detecties = stap2_detecteer(sigs)
    stap3_rapport(detecties)
    print(f"\n  Totale tijd: {time.time()-t0:.1f}s")

if __name__ == "__main__":
    asyncio.run(main())
