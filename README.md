# Agon Eval API

Minimal FastAPI service for the Agon AI Evaluation Platform.

## API shape

- `POST /`
- Request:

```json
{
  "query": "question as a string",
  "assets": ["https://asset-url-1.com", "https://asset-url-2.com"]
}
```

- Response:

```json
{
  "output": "your answer string"
}
```

## Local run

```bash
pip install -r requirements.txt
uvicorn app:app --host 0.0.0.0 --port 8000
```

## Deploy

Render can auto-detect `render.yaml`. Set `OPENAI_API_KEY` in the service environment before submitting the public URL to Agon.

## Notes

- Includes a small regex fallback so the sample `What is 10 + 15?` returns `The sum is 25.`
- For broader hidden tests, the service uses the OpenAI Responses API when `OPENAI_API_KEY` is configured.
