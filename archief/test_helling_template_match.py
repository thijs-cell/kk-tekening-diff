"""
Template matching wanddetectie — 56 de Helling pagina 1.
Stap 1: Vision -> bboxes arceringsblokjes -> templates/*.png (via fitz.Pixmap)
Stap 2: Pixel diff + BFS + NCC template matching op meerdere schalen
Stap 3: Rapport + ground truth check

Gebruik: python test_helling_template_match.py
"""
import asyncio, base64, json, math, re, sys, time
from collections import defaultdict
from pathlib import Path

import anthropic, fitz
import numpy as np
from numpy.lib.stride_tricks import sliding_window_view

SCRIPT_DIR = Path(__file__).parent
sys.path.insert(0, str(SCRIPT_DIR))
from app.diff_engine import _extract_lijnen, _vergelijk_lijnen, _vergelijk_fills, strip_annotations

MVP_DIR   = SCRIPT_DIR.parent / "Karregat & Koning MVP"
REF_DIR   = SCRIPT_DIR / "references" / "helling"
TPL_DIR   = REF_DIR / "templates"
UIT_DIR   = Path("/tmp/helling_templates")
MODEL     = "claude-sonnet-4-5"
PAGINA    = 0

OUD_PDF   = MVP_DIR / "56 de Helling - plattegronden gibowanden - oude tekening (5).pdf"
NIEUW_PDF = MVP_DIR / "56 de Helling - plattegronden gibowanden - Nieuwe tekening (2).pdf"

CLUSTER_AFSTAND = 80.0
MIN_ITEMS       = 3
MIN_BBOX_DIM    = 50.0
CROP_PAD        = 80.0
RENDER_DPI      = 200
NCC_DREMPEL     = 0.40
NCC_STRIDE      = 4
SCALES          = (0.5, 0.75, 1.0, 1.5, 2.0)
WALL_DIKTES     = {30,40,50,60,70,80,90,100,110,120,130,140,150,175,200,250}
DIKTE_RE        = re.compile(r'(?<!\d)(\d{2,3})(?!\d)')

# ===========================================================================
# STAP 1 — Templates extraheren via Vision
# ===========================================================================
_PROMPT_BBOX = """\
Dit is een renvooi (legenda) van een bouwkundige afbouwtekening.
Per wandtype: geef de pixel-coördinaten van het ARCERINGSBLOKJE — het kleine
visuele sample links van de naam, NIET de tekst zelf.

JSON: {"items": [{"naam": str, "x0": int, "y0": int, "x1": int, "y1": int}]}
"""

def _laad_b64(pad: Path) -> tuple[str, str]:
    data = pad.read_bytes()
    media = "image/jpeg" if data[:3] == b"\xff\xd8\xff" else "image/png"
    return base64.standard_b64encode(data).decode(), media

async def _vision_bboxes(client, label: str, pad: Path) -> list[dict]:
    b64, media = _laad_b64(pad)
    resp = await client.messages.create(
        model=MODEL, max_tokens=2048,
        messages=[{"role": "user", "content": [
            {"type": "image", "source": {"type": "base64", "media_type": media, "data": b64}},
            {"type": "text",  "text": _PROMPT_BBOX},
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
    items = data.get("items", [])
    print(f"  {label}: {len(items)} bboxes, {resp.usage.input_tokens + resp.usage.output_tokens} tokens")
    return items

def _sla_templates_op(renvooi_pad: Path, items: list[dict]):
    """Crop arceringsblokjes uit renvooi PNG via fitz document-open + clip."""
    doc = fitz.open(str(renvooi_pad))  # PNG als 1-page document
    page = doc.load_page(0)
    W, H = int(page.rect.width), int(page.rect.height)
    mat = fitz.Matrix(1, 1)  # 1pt = 1px voor image-documenten
    for it in items:
        x0 = max(0, int(it.get("x0", 0)))
        y0 = max(0, int(it.get("y0", 0)))
        x1 = min(W, int(it.get("x1", 0)))
        y1 = min(H, int(it.get("y1", 0)))
        if x1 - x0 < 8 or y1 - y0 < 8:
            continue
        naam_c = re.sub(r'[^a-z0-9]', '_', it["naam"].lower()).strip('_')
        try:
            pix = page.get_pixmap(matrix=mat, clip=fitz.Rect(x0, y0, x1, y1))
            pix.save(str(TPL_DIR / f"{naam_c}.png"))
            print(f"    {naam_c}.png  {x1-x0}x{y1-y0}px")
        except Exception as e:
            print(f"    SKIP {naam_c}: {e}")
    doc.close()

async def stap1_templates() -> dict[str, np.ndarray]:
    print("\n" + "=" * 64)
    print("STAP 1 -- Templates extraheren (Vision)")
    print("=" * 64)
    TPL_DIR.mkdir(parents=True, exist_ok=True)

    bestaand = list(TPL_DIR.glob("*.png"))
    if not bestaand:
        client = anthropic.AsyncAnthropic()
        oud_items, nieuw_items = await asyncio.gather(
            _vision_bboxes(client, "oud",  REF_DIR / "renvooi_oud.png"),
            _vision_bboxes(client, "nieuw", REF_DIR / "renvooi_nieuw.png"),
        )
        _sla_templates_op(REF_DIR / "renvooi_oud.png",  oud_items)
        _sla_templates_op(REF_DIR / "renvooi_nieuw.png", nieuw_items)
        bestaand = list(TPL_DIR.glob("*.png"))
    else:
        print(f"  {len(bestaand)} templates al aanwezig, Vision overgeslagen")

    # Templates laden als numpy grayscale arrays
    tpls: dict[str, np.ndarray] = {}
    for pad in bestaand:
        pix = fitz.Pixmap(str(pad))
        arr = np.frombuffer(bytes(pix.samples), dtype=np.uint8).reshape(pix.height, pix.width, pix.n)
        if pix.n >= 3:
            gray = arr[:,:,0].astype(np.float32)*0.299 + arr[:,:,1].astype(np.float32)*0.587 + arr[:,:,2].astype(np.float32)*0.114
        else:
            gray = arr[:,:,0].astype(np.float32)
        if gray.std() > 3.0:  # skip uniforme (all-white/all-black) templates
            tpls[pad.stem] = gray
        else:
            print(f"  Skip uniform: {pad.stem}")
    print(f"  {len(tpls)} bruikbare templates geladen")
    return tpls

# ===========================================================================
# NCC helpers
# ===========================================================================
def _resize_nn(arr: np.ndarray, factor: float) -> np.ndarray:
    H, W = arr.shape
    nh, nw = max(5, int(H*factor)), max(5, int(W*factor))
    return arr[np.ix_(np.linspace(0,H-1,nh).astype(int),
                      np.linspace(0,W-1,nw).astype(int))]

def _ncc(image: np.ndarray, tpl: np.ndarray) -> float:
    h, w = tpl.shape
    if image.shape[0] < h or image.shape[1] < w:
        return -1.0
    t_std = float(tpl.std())
    if t_std < 2.0:
        return -1.0
    t_norm = (tpl - tpl.mean()) / t_std
    wins = sliding_window_view(image, (h, w))[::NCC_STRIDE, ::NCC_STRIDE]
    stds = np.clip(wins.std(axis=(2,3), keepdims=True), 1.0, None)
    scores = ((wins - wins.mean(axis=(2,3), keepdims=True)) / stds * t_norm).mean(axis=(2,3))
    return float(scores.max())

def _match(img: np.ndarray, tpl: np.ndarray) -> float:
    return max(_ncc(img, _resize_nn(tpl, s)) for s in SCALES)

def _pix_gray(pix: fitz.Pixmap) -> np.ndarray:
    arr = np.frombuffer(bytes(pix.samples), dtype=np.uint8).reshape(pix.height, pix.width, pix.n)
    if pix.n >= 3:
        return arr[:,:,0].astype(np.float32)*0.299 + arr[:,:,1].astype(np.float32)*0.587 + arr[:,:,2].astype(np.float32)*0.114
    return arr[:,:,0].astype(np.float32)

# ===========================================================================
# STAP 2 — Detectie
# ===========================================================================
def _tekst(page, bbox) -> list[str]:
    x0,y0,x1,y1 = bbox
    return [w[4] for w in page.get_text("words") if x0<=(w[0]+w[2])/2<=x1 and y0<=(w[1]+w[3])/2<=y1]

def _dikte(t_oud, t_nieuw):
    def dd(ts):
        return {int(m) for t in ts for m in DIKTE_RE.findall(t) if int(m) in WALL_DIKTES}
    v, a = dd(t_oud)-dd(t_nieuw), dd(t_nieuw)-dd(t_oud)
    return (f"{'/'.join(str(x) for x in sorted(v))}mm -> {'/'.join(str(x) for x in sorted(a))}mm") if v and a else None

def stap2_detecteer(tpls: dict[str, np.ndarray]) -> list[dict]:
    print("\n" + "=" * 64)
    print("STAP 2 -- NCC matching + dikte (hele pagina 1)")
    print("=" * 64)

    oud_c = strip_annotations(str(OUD_PDF)); nieuw_c = strip_annotations(str(NIEUW_PDF))
    d_oud = fitz.open(oud_c); d_nieuw = fitz.open(nieuw_c)
    oi = _extract_lijnen(d_oud[PAGINA]); ni = _extract_lijnen(d_nieuw[PAGINA])
    _, _, lt, lv = _vergelijk_lijnen(oi, ni)
    _, ft, fv    = _vergelijk_fills(oi, ni)
    print(f"  lt={len(lt)} lv={len(lv)} ft={len(ft)} fv={len(fv)}")
    d_oud.close(); d_nieuw.close()

    alle = []
    for items, tag in [(lt,"lijn_toegevoegd"),(lv,"lijn_verdwenen"),(ft,"fill_toegevoegd"),(fv,"fill_verdwenen")]:
        for it in items:
            x = ((it["van"][0]+it["naar"][0])/2) if "van" in it else it["pos"][0]
            y = ((it["van"][1]+it["naar"][1])/2) if "van" in it else it["pos"][1]
            alle.append({**it, "_tag": tag, "_x": float(x), "_y": float(y)})

    cel = CLUSTER_AFSTAND; raster: dict = defaultdict(list)
    for i, it in enumerate(alle):
        raster[(int(it["_x"]//cel), int(it["_y"]//cel))].append(i)
    bezoekt = [False]*len(alle); clusters_raw = []
    for start in range(len(alle)):
        if bezoekt[start]: continue
        groep=[start]; bezoekt[start]=True; front=[start]
        while front:
            h=front.pop(); hx,hy=alle[h]["_x"],alle[h]["_y"]; gx,gy=int(hx//cel),int(hy//cel)
            for dx in(-1,0,1):
                for dy in(-1,0,1):
                    for j in raster.get((gx+dx,gy+dy),[]):
                        if not bezoekt[j] and math.hypot(alle[j]["_x"]-hx,alle[j]["_y"]-hy)<=cel:
                            bezoekt[j]=True; groep.append(j); front.append(j)
        xs=[alle[i]["_x"] for i in groep]; ys=[alle[i]["_y"] for i in groep]
        clusters_raw.append({"items":[alle[i] for i in groep],"bbox":(min(xs),min(ys),max(xs),max(ys))})

    clusters = sorted([c for c in clusters_raw if len(c["items"])>=MIN_ITEMS
        and c["bbox"][2]-c["bbox"][0]>=MIN_BBOX_DIM and c["bbox"][3]-c["bbox"][1]>=MIN_BBOX_DIM],
        key=lambda c:-len(c["items"]))
    print(f"  Clusters: {len(clusters)}")

    mat = fitz.Matrix(RENDER_DPI/72, RENDER_DPI/72)
    d_oud2 = fitz.open(str(OUD_PDF)); d_nieuw2 = fitz.open(str(NIEUW_PDF))
    p_oud2 = d_oud2[PAGINA]; p_nieuw2 = d_nieuw2[PAGINA]

    detecties = []
    for i, c in enumerate(clusters):
        bx0,by0,bx1,by1 = c["bbox"]
        clip = fitz.Rect(bx0-CROP_PAD, by0-CROP_PAD, bx1+CROP_PAD, by1+CROP_PAD)
        g_oud  = _pix_gray(p_oud2.get_pixmap(matrix=mat, clip=clip))
        g_nieuw = _pix_gray(p_nieuw2.get_pixmap(matrix=mat, clip=clip))

        scores_n = {n: _match(g_nieuw, t) for n, t in tpls.items()}
        scores_o = {n: _match(g_oud,   t) for n, t in tpls.items()}
        beste = max(scores_n, key=lambda n: scores_n[n])
        sn, so = scores_n[beste], scores_o[beste]

        if sn < NCC_DREMPEL and so < NCC_DREMPEL:
            wt, wij, conf = "onbekend", "onbekend", "laag"
        else:
            wt = beste.replace("_"," ")
            wij = "toegevoegd" if sn>NCC_DREMPEL and so<NCC_DREMPEL else \
                  "verdwenen"  if so>NCC_DREMPEL and sn<NCC_DREMPEL else "gewijzigd"
            conf = "midden"

        dk = _dikte(_tekst(p_oud2,(bx0-30,by0-30,bx1+30,by1+30)),
                    _tekst(p_nieuw2,(bx0-30,by0-30,bx1+30,by1+30)))
        if dk and conf=="midden": conf="hoog"

        detecties.append({"bbox":c["bbox"],"n":len(c["items"]),"wandtype":wt,
                           "wijziging":wij,"dikte":dk,"confidence":conf,
                           "score_n":round(sn,3),"score_o":round(so,3)})
        print(f"  [{i:02d}] {wt:<33} {wij:<12} n={sn:.3f}/o={so:.3f}  dk={dk or '-'}")

    d_oud2.close(); d_nieuw2.close()
    return detecties

# ===========================================================================
# STAP 3 — Rapport
# ===========================================================================
def stap3_rapport(detecties: list[dict]):
    print("\n" + "=" * 64)
    print("STAP 3 -- RAPPORT")
    print("=" * 64)
    UIT_DIR.mkdir(parents=True, exist_ok=True)
    volgorde = {"hoog":0,"midden":1,"laag":2,"onbekend":3}
    ges = sorted(detecties, key=lambda d:volgorde.get(d["confidence"],9))

    print(f"\n  {'#':<4} {'Wandtype':<33} {'Wijz':<12} {'Dikte':<20} {'Score n/o':<12} Conf")
    print("  "+"-"*90)
    for i,d in enumerate(ges):
        print(f"  {i:<4} {d['wandtype']:<33} {d['wijziging']:<12} {d['dikte'] or '-':<20} "
              f"{d['score_n']:.3f}/{d['score_o']:.3f}  {d['confidence']}")

    print("\n  --- Ground truth check (deel 4: x=3449-3891, y=608-1156) ---")
    d4=(3449,608,3891,1156)
    in4=[d for d in detecties if d["bbox"][0]>=d4[0]-80 and d["bbox"][2]<=d4[2]+80
         and d["bbox"][1]>=d4[1]-80 and d["bbox"][3]<=d4[3]+80]
    w1=[d for d in in4 if "gibo" in d["wandtype"].lower() and d["dikte"]
        and "70" in d["dikte"] and "100" in d["dikte"]]
    w2=[d for d in in4 if "hardschuim" in d["wandtype"].lower() and d["wijziging"] in ("toegevoegd","gewijzigd")]
    print(f"  In deel 4 : {len(in4)}")
    print(f"  W1 Gibo 70->100mm        : {'GEVONDEN' if w1 else 'NIET GEVONDEN'}")
    print(f"  W2 Hardschuimisolatie    : {'GEVONDEN' if w2 else 'NIET GEVONDEN'}")
    print(f"  Buiten deel 4            : {len(detecties)-len(in4)}")

    print(f"\n  Top 10 crops -> {UIT_DIR}")
    d_o=fitz.open(str(OUD_PDF)); d_n=fitz.open(str(NIEUW_PDF))
    mat=fitz.Matrix(200/72,200/72)
    for i,d in enumerate(ges[:10]):
        bx0,by0,bx1,by1=d["bbox"]; clip=fitz.Rect(bx0-CROP_PAD,by0-CROP_PAD,bx1+CROP_PAD,by1+CROP_PAD)
        for doc,lbl in [(d_o,"oud"),(d_n,"nieuw")]:
            doc[PAGINA].get_pixmap(matrix=mat,clip=clip).save(str(UIT_DIR/f"det_{i:02d}_{lbl}.png"))
    d_o.close(); d_n.close()

# ===========================================================================
# Main
# ===========================================================================
async def main():
    t0=time.time()
    tpls = await stap1_templates()
    if not tpls:
        print("Geen bruikbare templates. Stop."); return
    detecties = stap2_detecteer(tpls)
    stap3_rapport(detecties)
    print(f"\n  Totale tijd: {time.time()-t0:.1f}s")

if __name__=="__main__":
    asyncio.run(main())
