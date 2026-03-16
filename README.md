# K&K Tekening Diff

FastAPI service die twee PDF bouwtekeningen vergelijkt en per pagina de wijzigingen detecteert via OpenCV.

## Snel starten

### Lokaal

```bash
pip install -r requirements.txt
uvicorn app.main:app --host 0.0.0.0 --port 8080
```

**Let op:** Poppler moet geïnstalleerd zijn (`apt-get install poppler-utils` of `brew install poppler`).

### Docker

```bash
docker build -t kk-tekening-diff .
docker run -p 8080:8080 kk-tekening-diff
```

## Gebruik

```bash
curl -X POST http://localhost:8080/compare \
  -F "old_pdf=@tekening_v1.pdf" \
  -F "new_pdf=@tekening_v2.pdf" \
  -o result.json
```

Optionele parameters: `?dpi=200&sensitivity=30`

## Configuratie

Zie `.env.example` voor beschikbare environment variables.
