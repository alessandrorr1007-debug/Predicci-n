"""
AulaPredict - Módulo Imagen
Servidor API Backend con FastAPI (main.py)
"""

import os
import sys

# Configurar UTF-8 en consola de Windows para evitar errores con emojis
os.environ["PYTHONUTF8"] = "1"
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
if sys.platform.startswith("win"):
    try:
        if hasattr(sys.stdout, "reconfigure"):
            sys.stdout.reconfigure(encoding="utf-8")
        if hasattr(sys.stderr, "reconfigure"):
            sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass

import logging
import threading
from contextlib import asynccontextmanager
from typing import Dict, Any

from fastapi import FastAPI, File, UploadFile, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from emotion_model import detectar_emocion, precargar_modelo, esta_modelo_listo

# Configuración del sistema de registro (Logging)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] [%(name)s]: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("aulapredict.main")

# Límite máximo de tamaño de archivo (5 MB)
MAX_FILE_SIZE = 5 * 1024 * 1024
# Tipos MIME permitidos
ALLOWED_MIME_TYPES = {
    "image/jpeg",
    "image/png",
    "image/webp",
    "image/bmp",
    "application/octet-stream", # Algunos navegadores envían octet-stream para blobs
}


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Gestión del ciclo de vida de la aplicación.
    Inicia la precarga del modelo en un hilo separado al arrancar el servidor
    para que la API responda inmediatamente y realice el warm-up sin bloquear.
    """
    logger.info("Iniciando servicio AulaPredict - Módulo Imagen...")
    # Ejecutamos el warm-up del modelo en un hilo en segundo plano
    hilo_precarga = threading.Thread(
        target=precargar_modelo,
        name="HiloWarmupModelo",
        daemon=True,
    )
    hilo_precarga.start()
    yield
    logger.info("Cerrando servicio AulaPredict.")


# Inicialización de la aplicación FastAPI
app = FastAPI(
    title="AulaPredict - Módulo Imagen",
    description="API para la detección de emociones en imágenes capturadas con cámara web.",
    version="1.0.0",
    lifespan=lifespan,
)

# Configuración de CORS para permitir la comunicación con el frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # En producción se puede restringir a dominios específicos
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get(
    "/salud",
    summary="Verificar estado de salud del servidor",
    response_description="Estado del servidor y preparación del modelo",
)
async def verificar_salud() -> Dict[str, Any]:
    """
    Endpoint GET /salud:
    Permite al frontend verificar si el servidor está activo y si el modelo
    de emociones ya completó su descarga/warm-up inicial.
    """
    listo = esta_modelo_listo()
    return {
        "estado": "ok",
        "modelo_listo": listo,
        "mensaje": "Servicio operando normalmente" if listo else "El modelo se está preparando, espera unos segundos...",
    }


@app.post(
    "/analizar",
    summary="Analizar emoción en una imagen facial",
    response_description="Emoción detectada, probabilidades, confianza y estado académico",
)
async def analizar_imagen(
    archivo: UploadFile = File(..., description="Archivo de imagen facial (JPEG, PNG, WebP)")
) -> Dict[str, Any]:
    """
    Endpoint POST /analizar:
    Recibe una imagen (multipart/form-data), valida su tamaño y formato,
    la procesa enteramente en memoria RAM (sin tocar el disco duro)
    y retorna la emoción detectada, confianza y estado académico.
    """
    # 1. Validación de tipo MIME
    tipo_contenido = archivo.content_type or ""
    if tipo_contenido and tipo_contenido not in ALLOWED_MIME_TYPES:
        logger.warning(f"Tipo de archivo no permitido recibido: {tipo_contenido}")
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Formato de imagen no soportado ({tipo_contenido}). Usa JPEG, PNG o WebP.",
        )

    # 2. Lectura en memoria RAM
    try:
        contenido_bytes = await archivo.read()
    except Exception as e:
        logger.error(f"Error al leer el archivo recibido: {e}")
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="No se pudo leer el archivo transmitido.",
        )

    # 3. Validación de tamaño (máx. 5 MB)
    tamano_bytes = len(contenido_bytes)
    if tamano_bytes == 0:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="El archivo recibido está vacío.",
        )

    if tamano_bytes > MAX_FILE_SIZE:
        logger.warning(f"Archivo excedió el límite: {tamano_bytes} bytes")
        raise HTTPException(
            status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
            detail=f"La imagen supera el límite de 5 MB ({tamano_bytes / (1024 * 1024):.2f} MB).",
        )

    # 4. Verificación de preparación del modelo
    if not esta_modelo_listo():
        logger.info("Petición recibida mientras el modelo está calentando.")
        return JSONResponse(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            content={
                "estado": "preparando",
                "modelo_listo": False,
                "mensaje": "El modelo se está preparando, espera unos segundos e intenta nuevamente.",
            },
        )

    # 5. Inferencia en memoria usando emotion_model.py
    logger.info(f"Procesando imagen en memoria ({tamano_bytes / 1024:.1f} KB)...")
    resultado = detectar_emocion(contenido_bytes)

    return resultado


# Montaje del frontend estático para despliegue unificado (ej. en Render)
ruta_frontend = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "frontend"))
if os.path.exists(ruta_frontend):
    from fastapi.staticfiles import StaticFiles
    app.mount("/", StaticFiles(directory=ruta_frontend, html=True), name="frontend")


# Punto de entrada para ejecución directa
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
