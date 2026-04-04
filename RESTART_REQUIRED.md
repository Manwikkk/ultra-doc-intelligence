# ⚠️ IMPORTANT — Always restart the backend after any .env change

The backend caches all settings at startup. If you change `.env`,
you MUST stop and restart uvicorn or the old model name stays active.

```bash
# Stop the running server (Ctrl+C), then:
cd backend
uvicorn main:app --reload --port 8000
```

## Verify the correct model is loading

Look for this in the server startup logs:
```
LLM Router initialised
  GROQ_MODEL  : llama-3.3-70b-versatile    ← must say this
  GROQ_KEY    : gsk_jdt7yBMb...(hidden)
```

If you see `llama3-8b-8192` here, the old .env is still being read.
Make sure you run uvicorn from inside the `backend/` directory, not the project root.

## Current valid Groq models (April 2026)
- llama-3.3-70b-versatile  ← recommended, set in .env
- llama-3.1-8b-instant
- mixtral-8x7b-32768
- gemma2-9b-it

## ❌ Decommissioned (will fail)
- llama3-8b-8192
- llama2-70b-4096
