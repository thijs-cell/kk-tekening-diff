# K&K Tekening Diff

FastAPI service die twee PDF demarcatietekeningen vergelijkt op tekst, lijnen, vullingen en kleuren via PyMuPDF.

## Snel starten

### Lokaal

```bash
pip install -r requirements.txt
uvicorn app.main:app --reload
```

### Docker

```bash
docker build -t kk-tekening-diff .
docker run -p 8000:8000 kk-tekening-diff
```

## Gebruik

```bash
curl -X POST http://localhost:8000/diff \
  -F "oud_pdf=@tekening_v1.pdf" \
  -F "nieuw_pdf=@tekening_v2.pdf" \
  -F "pagina=1"
```

Swagger UI: http://localhost:8000/docs

## Endpoints

| Methode | Pad | Beschrijving |
|---------|------|-------------|
| GET | `/` | Health check |
| POST | `/diff` | Vergelijk twee PDF's, retourneer JSON met alle wijzigingen |

## Output secties

De JSON response bevat:

- `tekst_gewijzigd` — tekst met andere inhoud (met auto-categorie: maat, revisieletter, oppervlakte, etc.)
- `tekst_toegevoegd` / `tekst_verdwenen` — nieuwe of verdwenen tekst
- `tekst_kleur_gewijzigd` — zelfde tekst, andere kleur (scope-wijziging)
- `lijn_kleur_gewijzigd` — niet-zwarte lijnen met kleurwijziging
- `vul_kleur_gewijzigd` — gevulde vlakken met kleurwijziging
- `nieuwe_gekleurde_vlakken` / `verdwenen_gekleurde_vlakken` — vlakken > 100 opp
- `lijnen_linewidth_gewijzigd` / `lijnen_toegevoegd` / `lijnen_verdwenen` — aantallen + sample
- `kleur_inventaris` — alle kleuren per PDF (tekst, lijnen, vullingen)
- `totalen` — samenvatting

## Stack

- Python 3.12, FastAPI, PyMuPDF
- Geen externe dependencies (geen poppler, opencv, anthropic)
