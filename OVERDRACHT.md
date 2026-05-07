# OVERDRACHT — K&K Tekening-diff

Voor: Cas
Datum: 2026-05-07
Projectmap: `volledige_oude_map/kk-tekening-diff/`

---

## 1. PROJECT-STATUS

**Doel.** Wand-detectie pipeline voor K&K Afbouw bouwen die automatisch wand-wijzigingen tussen oude en nieuwe bouwtekening kan herkennen.

**Eindproduct (visie).** Dicky uploadt drie bestanden: een renvooi (legenda) + een oude tekening + een nieuwe tekening. Hij krijgt automatisch een rapport terug met:
- Oppervlakte-veranderingen
- Ruimtenamen
- Maatwijzigingen
- **Wand-wijzigingen (toegevoegd / verdwenen / type gewijzigd)** ← dit deel werkt nog niet

**Wat werkt al in productie:**
- Oppervlakte-detectie
- Ruimtenamen
- Maatwijzigingen

**Wat nog niet werkt:**
- Wand toegevoegd
- Wand verdwenen
- Wand-type gewijzigd (bijv. Gibo zwaar 70 → 100mm, isolatie toegevoegd)

---

## 2. PROJECT-STRUCTUUR

Hoofdmap: `volledige_oude_map/kk-tekening-diff/`

### Mappen
- `app/` — applicatiecode (FastAPI + diff-engine)
- `archief/` — eerdere experimenten en weggegooide aanpakken (raadplegen voordat je opnieuw begint)
- `data/` — testtekeningen (4 paren oud/nieuw)
- `output/` — gegenereerde rapporten (build artifacts)
- `references/` — gecachete renvooi-uitlezing, signatures en ground truth
- `tests/` — losse test-/diagnose-scripts (geen pytest-suite)
- `tools/` — utility scripts
- `Dockerfile`, `requirements.txt`, `README.md`

### Endpoints in `app/main.py`
- `/vergelijk-split` — productie-endpoint dat Dicky gebruikt (oppervlakte + ruimtenamen + maten). **Niet aanraken.**
- (overige endpoints zijn ondersteunend / experimenteel)

### Belangrijke modules in `app/`
- `diff_engine.py` — kern van vergelijking (oppervlakte/maat/ruimtenaam) — **niet aanraken**
- `preflight.py` — crash-preventie / input-validatie — **niet aanraken**
- `interpreter.py` — renvooi-uitlezing
- `layout_detect.py` — pagina-layout / clustering
- `tekening_profiel.py` — per-tekening configuratie
- `wall_detect.py`, `wand_diff.py`, `wand_diff_vision.py` — experimentele wand-detectie (work in progress)
- `overlay.py` — visualisatie

### Testscripts in `tests/`
- `test_renvooi_extract.py` — Vision-keten voor renvooi-uitlezing (werkt)
- `test_kleur_dikte_matching.py` — deterministische kleur+dikte matching (faalt op huidige GT)
- `test_template_matching.py` — template matching ronde 0 / 1
- `test_template_full_page.py` — schaalbaarheid template matching
- `test_vision_cluster_classify.py` — Vision per-tegel classificatie
- `test_diagnose_gt_helling.py` — diagnose script voor ground truth helling

### Data in `data/` (4 paren oud/nieuw, totaal 8 PDFs)
- `helling/` — kleurtekening, hoofdtestcase
- `muiden/`
- `bd_n101/`
- `5102/` — zwart-wit (kleurmatching werkt sowieso niet)

### References (opgebouwd)
- `references/helling/`, `references/muiden/`, `references/bd_n101/`, `references/5102/` — gecachete renvooi-uitlezing en signatures per tekening
- `references/helling_ground_truth/cluster_0_deel4/` — handmatig gedocumenteerde ground truth (2 wijzigingen, zie sectie 6)

---

## 3. WAT IS GEPROBEERD VOOR WAND-DETECTIE

Chronologische lijst van experimenten + resultaat:

| # | Aanpak | Resultaat |
|---|--------|-----------|
| 1 | Vision-tegelen, open detectie ("vind alle wand-wijzigingen") | Variabel, recall 5/1/4/4/4/2 over 6 runs op dezelfde GT — niet reproduceerbaar |
| 2 | Vision per tegel mét renvooi-context | Vond sandwichpaneel correct, maar **veel false positives** |
| 3 | Pixel-diff + Vision classificatie (hybride) | **0 echte wand-wijzigingen** gevonden op plattegrond |
| 4 | Renvooi-extractie via Vision | **WERKT** — alle items gevonden, gecached in `references/` |
| 5 | Kleur + dikte deterministisch matching | **0/2 GT recall** — meeste wanden hebben geen solid fill, alleen arcering |
| 6 | Template matching ronde 0 (lossere drempel) | **2/2 GT recall**, maar **4805 false positives** op 3% van pagina (cluster_0_deel4) |
| 7 | Template matching ronde 1 (strengere drempel) | **0/2 GT recall** — GT zat onder drempel, terug naar af |

---

## 4. KERNPROBLEEM

Drie open vragen die op moment van overdracht niet beantwoord zijn:

1. **Cross-correlation onderscheidingsvermogen niet bewezen.** We weten niet of er überhaupt een score-separatie bestaat tussen "echte template-match" en "ruis op de tekening".
2. **Pagina-brede schaalbaarheid niet bewezen.** Alle template matching tests draaiden op `cluster_0_deel4` — dat is ~3% van pagina 1. We weten niet hoe het schaalt over de hele pagina.
3. **Mogelijk fundamenteel probleem.** In ronde 0 zat de GT op marginale scores tussen 4805 false positives. Dat suggereert dat de GT-wandwijziging niet duidelijk uniek scoort — geen drempel scheidt echt van ruis.

---

## 5. OPEN OPTIES OP MOMENT VAN OVERDRACHT

### A. Diagnostische test cross-correlation
Meten of er score-separatie bestaat tussen echte match en ruis. **Goedkope sanity-check** voordat je verder bouwt op template matching.

### B. Per-wandtype Vision-detectie
Eén Vision-call per wandtype met gerichte vraag ("zie je ergens een sandwichpaneel verschenen?"). Verkleint zoekruimte, verhoogt waarschijnlijk precision.

### C. Vision converteert tekening of clusters naar code-representatie
Vision leest de tekening uit naar een gestructureerde representatie (bijv. JSON met wand-segmenten) waar diff-logica deterministisch op draait.

### D. Pagina-brede template matching test
Schaalbaarheid van template matching meten op hele pagina i.p.v. alleen cluster_0_deel4.

### E. Stoppen met automatische wand-detectie
Lever het product zonder wand-detectie. Oppervlakte/ruimtenamen/maten zijn al waardevol genoeg.

### Aanbeveling op moment van overdracht
**Eerst A of B**, voordat C of D verder uitgewerkt worden.
- A geeft je een ja/nee op de kernvraag of template matching überhaupt kan werken.
- B gebruikt Vision waar het al bewezen werkt (renvooi-extractie), maar dan gericht.

---

## 6. GROUND TRUTH BESCHIKBAAR

- `references/helling_ground_truth/cluster_0_deel4/` — bevat **2 GT-wijzigingen**:
  1. Gibo zwaar 70mm → 100mm
  2. Hardschuimisolatie toegevoegd

### Niet beschikbaar / nog te doen
- Geen ground truth voor `muiden`, `bd_n101`, `5102`
- Ground truth voor de rest van pagina 1 helling is ook nog niet gedocumenteerd
- → bij elk nieuw experiment heb je dus alleen 2 GT-punten om tegen te valideren

---

## 7. GETESTE TEKENINGEN

| Tekening | Type | Status |
|----------|------|--------|
| helling | kleur | Diepgaand getest op `cluster_0_deel4` |
| muiden | kleur | Niet diepgaand getest met huidige aanpak |
| bd_n101 | kleur | Niet diepgaand getest |
| 5102 | zwart-wit | Kleurmatching werkt sowieso niet |

---

## 8. WAT IN PRODUCTIE BLIJFT (NIET TOUCHEREN)

- `/vergelijk-split` endpoint — werkt voor Dicky
- `app/diff_engine.py` kern — oppervlakte / maat / ruimtenaam detectie
- `app/preflight.py` — crash-preventie
- `vind_legenda_combined` — renvooi-fallback voor 5102

Wand-detectie experimenten draaien in losse modules (`wall_detect.py`, `wand_diff*.py`) en testscripts — die kun je vrij wijzigen.

---

## 9. AANBEVELING VOOR CAS — HOE TE BEGINNEN

1. **Run** `tests/test_renvooi_extract.py` om te verifiëren dat de Vision-keten nog werkt.
2. **Run** `tests/test_kleur_dikte_matching.py` om de deterministische kleur-matching met eigen ogen te zien (en te zien waarom hij 0/2 scoort).
3. **Open** `archief/` — daar staat eerdere experiment-code. Doorlopen voorkomt dat je iets opnieuw bouwt dat al gefaald heeft.
4. **Kies** één van de 5 opties uit sectie 5.
5. **Documenteer** je keuze onderaan dit OVERDRACHT.md vóór je begint te coden, zodat we beiden weten waar we staan.

---

## 10. KEUZE CAS (in te vullen)

> _Vul hier in welke optie je kiest en waarom, vóór je begint._

- Gekozen optie:
- Reden:
- Verwachte eerste milestone:
