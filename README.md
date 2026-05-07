# K&K Tekening Diff

FastAPI service voor K&K Afbouw die twee versies van een bouwtekening (PDF) vergelijkt en de wijzigingen visualiseert. Gebruikt PyMuPDF voor PDF-parsing en optioneel Anthropic Vision voor renvooi-uitlezing.

## Doel

Bij K&K Afbouw worden demarcatietekeningen regelmatig herzien. Voor de offerte- en uitvoeringsfase moet duidelijk zijn welke wanden, ruimtes, materialen of revisies tussen twee versies veranderd zijn. Dit project automatiseert die vergelijking en levert een JSON-rapport plus een overlay-PDF met de wijzigingen.

## Folder-structuur

```
kk-tekening-diff/
├── app/             productie-code (FastAPI service)
├── references/      renvooi-afbeeldingen, context.md en vision-output per project
├── data/            bron-PDFs (oud.pdf / nieuw.pdf per project) — git-ignored
├── tests/           actieve testscripts
├── archief/         oude experimenten (bewaard, niet meer actief)
├── output/          gegenereerde overlay-PDFs
├── README.md
├── requirements.txt
├── Dockerfile
├── .env             API-keys (git-ignored)
└── .gitignore
```

### app/
- `main.py` — FastAPI entrypoint en endpoints
- `diff_engine.py` — kern van de tekstuele/grafische diff
- `overlay.py` — genereert overlay-PDF
- `interpreter.py`, `layout_detect.py`, `wall_detect.py`, `wand_diff.py`, `wand_diff_vision.py`, `tekening_profiel.py`, `preflight.py`, `config.py`

### data/
Bevat per project een `oud.pdf` en `nieuw.pdf`. Niet in git — kopieer lokaal vanuit de gedeelde projectmap.

### references/
Per project: `renvooi.png` (de uitgesneden renvooi-tabel), `context.md` (handmatige context), en `vision_complete.json` (Vision-output van de renvooi). Helling heeft gesplitste oud/nieuw varianten en een `signatures.json`.

## Beschikbare testtekening-projecten

| Project | Map | Pagina-formaat | Bron |
|---|---|---|---|
| helling | `data/helling/` | 1700 × 594 mm (7 vs 8 pagina's) | 56 de Helling — plattegronden gibowanden |
| muiden  | `data/muiden/`  | 1350 × 841 mm  | WT-PLG-D2.1 (revisie B → E) |
| bd_n101 | `data/bd_n101/` | 1189 × 841 mm (A0) | BD-N-101 → BU-N-101 V2 |
| 5102    | `data/5102/`    | 1189 × 841 mm (A0) | 5102 Eerste verdieping (jan → maart 2025) |

Renvooi-uitsneden en context staan in `references/<project>/`.

## Snel starten

### Lokaal

```bash
pip install -r requirements.txt
uvicorn app.main:app --reload
```

Service draait op http://localhost:8000 — Swagger UI op http://localhost:8000/docs.

### Docker

```bash
docker build -t kk-tekening-diff .
docker run -p 8000:8000 --env-file .env kk-tekening-diff
```

### Diff aanroepen

```bash
curl -X POST http://localhost:8000/diff \
  -F "oud_pdf=@data/helling/oud.pdf" \
  -F "nieuw_pdf=@data/helling/nieuw.pdf" \
  -F "pagina=1"
```

## Endpoints

| Methode | Pad | Beschrijving |
|---|---|---|
| GET  | `/` | Hoofdpagina (login of UI) |
| GET  | `/health` | Health check |
| GET  | `/login` / POST `/login` | Inloggen |
| POST | `/diff` | Vergelijk twee PDF's, retourneer JSON met alle wijzigingen |
| POST | `/overlay` | Genereer overlay-PDF |
| POST | `/rapport` | Diff-rapport |
| POST | `/rapport-volledig` | Diff-rapport inclusief alle pagina's |
| POST | `/vergelijk-split` | Vergelijking met split-output |
| POST | `/feedback` | Feedback opslaan |
| GET  | `/feedback-bestanden/{ts}` | Feedback-bestandenlijst |
| GET  | `/feedback-bestanden/{ts}/{bestandsnaam}` | Feedback-bestand download |

## Output secties (van `/diff`)

- `tekst_gewijzigd` — tekst met andere inhoud (auto-categorie: maat, revisieletter, oppervlakte, etc.)
- `tekst_toegevoegd` / `tekst_verdwenen` — nieuwe of verdwenen tekst
- `tekst_kleur_gewijzigd` — zelfde tekst, andere kleur
- `lijn_kleur_gewijzigd` — niet-zwarte lijnen met kleurwijziging
- `vul_kleur_gewijzigd` — gevulde vlakken met kleurwijziging
- `nieuwe_gekleurde_vlakken` / `verdwenen_gekleurde_vlakken` — vlakken > 100 opp
- `lijnen_linewidth_gewijzigd` / `lijnen_toegevoegd` / `lijnen_verdwenen` — aantallen + sample
- `kleur_inventaris` — alle kleuren per PDF (tekst, lijnen, vullingen)
- `totalen` — samenvatting

## Stack

- Python 3.12+
- FastAPI 0.115, uvicorn 0.34
- PyMuPDF 1.24+
- anthropic 0.50+ (voor optionele Vision-renvooi-uitlezing)

## Tests

Actieve tests staan in `tests/`. Oudere experimenten zijn naar `archief/` verplaatst — niet weggegooid maar uit de actieve werkruimte.

```bash
python tests/test_renvooi_extract.py
```
