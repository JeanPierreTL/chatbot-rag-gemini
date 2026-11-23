# Chatbot RAG con Gemini, MongoDB Atlas y Streamlit Cloud

1. Ejecute `app_llm.ipynb` en Colab (secretos GOOGLE_API_KEY y MONGODB_URI) con el PDF de clase.
2. Suba a un repositorio público: `app.py`, `requirements.txt`, `.gitignore`, `README.md`.
3. En share.streamlit.io: Create app → repo, rama `main`, `app.py` → Advanced settings → Secrets:

```toml
[app]
GOOGLE_API_KEY = "<su_api_key>"
MONGODB_URI = "mongodb+srv://<usuario>:<password>@<cluster>.mongodb.net/"
```

Nunca suba `.streamlit/secrets.toml`.
