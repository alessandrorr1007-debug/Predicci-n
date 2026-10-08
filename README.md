# AulaPredict - Backend (API de Detección de Emociones)

API en FastAPI para el análisis en memoria de emociones faciales y estado académico mediante DeepFace, CLAHE, alineación ocular y Test-Time Augmentation (TTA).

---

## 🚀 Despliegue en Render (Web Service)

Para desplegar este backend en [Render](https://dashboard.render.com):

1. Haz clic en **New +** ➔ **Web Service**.
2. Conecta este repositorio: `https://github.com/alessandrorr1007-debug/Prediccion_Backend`.
3. Configuración del servicio:
   - **Name:** `prediccion-backend` (o el nombre que elijas)
   - **Region:** Ohio (US East) o Frankfurt
   - **Branch:** `main`
   - **Root Directory:** *(dejar vacío)*
   - **Runtime:** `Python 3`
   - **Build Command:**
     ```bash
     pip install -r requirements.txt
     ```
   - **Start Command:**
     ```bash
     uvicorn main:app --host 0.0.0.0 --port $PORT
     ```
   - **Plan:** `Free`
4. *(Opcional)* En **Environment Variables**, añade:
   - `PYTHON_VERSION`: `3.11.9` *(ya incluido en `.python-version`)*

Al finalizar, Render te dará la URL de tu API (ejemplo: `https://prediccion-backend.onrender.com`).

---

## 📡 Endpoints Disponibles

- `GET /salud`: Estado del servidor y preparación del modelo (`{"estado": "ok", "modelo_listo": true}`).
- `POST /analizar`: Recibe imagen (`multipart/form-data`) y retorna predicción de emoción, confianza, probabilidades y estado académico.
- `GET /docs`: Documentación interactiva Swagger.
