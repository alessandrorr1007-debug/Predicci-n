"""
AulaPredict - Módulo Imagen
Módulo Avanzado de Detección de Emociones (emotion_model.py)

================================================================================
MEJORAS DE PRECISIÓN Y CALIDAD IMPLEMENTADAS:
================================================================================
1. Preprocesamiento CLAHE (Contrast Limited Adaptive Histogram Equalization):
   - Normaliza la iluminación y contraste en el espacio de color LAB.
   - Elimina sombras oscuras bajo cejas y comisuras labiales típicas de webcams.
   - Denoising bilateral para limpiar ruido de sensor sin degradar bordes faciales.
2. Alineación Facial Automática (align=True):
   - Alinea los ojos horizontalmente para máxima precisión de la red neuronal.
3. Test-Time Augmentation (TTA):
   - Evalúa la imagen original y su versión invertida (espejo), promediando probabilidades.
   - Evita falsos positivos por iluminación asimétrica.
4. Calibración de Sesgo de Iluminación:
   - Suaviza la hipersensibilidad de FER2013 a sombras en cuencas de ojos (falsos "miedo"/"triste")
     estabilizando la detección en rostros en reposo (neutrales).
5. Detección Automática de Modelo Propio Entrenado en Google Colab:
   - Si existe 'modelo_emociones.keras' o 'modelo_emociones.h5' en backend/, se carga
     automáticamente para inferencia de máxima precisión.
================================================================================
"""

import io
import os
import sys
import logging

# Configurar UTF-8 en consola de Windows para evitar errores con emojis del logger de DeepFace
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

import cv2
import numpy as np
from PIL import Image

# Configuración de logging
logger = logging.getLogger("aulapredict.emotion_model")

# Variable de estado global para controlar la preparación del modelo
_modelo_listo = False
_modelo_propio_colab = None

# Rutas para modelo personalizado exportado desde Google Colab
RUTA_MODELO_KERAS = os.path.join(os.path.dirname(__file__), "modelo_emociones.keras")
RUTA_MODELO_H5 = os.path.join(os.path.dirname(__file__), "modelo_emociones.h5")

# Diccionario de traducción de emociones a español
TRADUCCION_EMOCIONES = {
    "happy": "feliz",
    "sad": "triste",
    "angry": "enojado",
    "neutral": "neutral",
    "surprise": "sorprendido",
    "fear": "miedo",
    "disgust": "disgusto",
}

# Mapeo a estado académico
MAPEO_ESTADO_ACADEMICO = {
    "triste": "posible riesgo",
    "miedo": "posible riesgo",
    "enojado": "posible riesgo",
    "disgusto": "posible riesgo",
    "neutral": "estable",
    "feliz": "positivo",
    "sorprendido": "positivo",
}

# Pesos de calibración para cámaras web (mitiga sombras cenitales de webcam)
PESOS_CALIBRACION = {
    "happy": 1.05,
    "neutral": 1.35,      # Aumenta la estabilidad en rostros en reposo
    "surprise": 1.05,
    "sad": 0.82,          # Suaviza falsos positivos por sombras bajo los labios
    "fear": 0.62,         # Suaviza falsos positivos por sombras en cuencas oculares
    "angry": 0.85,
    "disgust": 0.75,
}

# Clasificador Haar Cascade de OpenCV para validación propia de presencia de rostro
_face_cascade = None


def _obtener_clasificador_rostros():
    """Carga de forma perezosa el clasificador Haar Cascade de OpenCV."""
    global _face_cascade
    if _face_cascade is None:
        try:
            cascade_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
            _face_cascade = cv2.CascadeClassifier(cascade_path)
            logger.info("Clasificador Haar Cascade cargado para validación de rostros.")
        except Exception as e:
            logger.warning(f"No se pudo cargar Haar Cascade: {e}")
    return _face_cascade


def esta_modelo_listo() -> bool:
    """Devuelve True si el modelo ha completado su precarga y warmup."""
    return _modelo_listo


def mejorar_calidad_imagen(img_bgr: np.ndarray) -> np.ndarray:
    """
    Pipeline de mejora visual para optimizar la entrada de la red neuronal:
    1. Filtrado bilateral para limpiar el ruido del sensor de la webcam sin difuminar bordes.
    2. Ecualización adaptativa de contraste (CLAHE) en el canal L del espacio LAB.
       Compensa la iluminación deficiente y elimina sombras en los ojos.
    """
    try:
        # Filtro bilateral para reducir ruido de alta frecuencia
        suavizada = cv2.bilateralFilter(img_bgr, d=5, sigmaColor=25, sigmaSpace=25)
        # Separación a espacio LAB
        lab = cv2.cvtColor(suavizada, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        # CLAHE sobre luminosidad
        clahe = cv2.createCLAHE(clipLimit=2.2, tileGridSize=(8, 8))
        l_ecualizado = clahe.apply(l)
        lab_final = cv2.merge((l_ecualizado, a, b))
        return cv2.cvtColor(lab_final, cv2.COLOR_LAB2BGR)
    except Exception as e:
        logger.warning(f"Error al aplicar mejora CLAHE: {e}")
        return img_bgr


def precargar_modelo():
    """
    Precarga el modelo de emociones y realiza un warm-up con una imagen sintética.
    Detecta automáticamente si existe un modelo entrenado en Colab (.keras o .h5)
    o de lo contrario utiliza DeepFace con pipeline optimizado.
    """
    global _modelo_listo, _modelo_propio_colab
    logger.info("Cargando modelo de inteligencia artificial...")

    # 1. Comprobar si existe un modelo propio exportado desde Colab
    ruta_modelo = None
    if os.path.exists(RUTA_MODELO_KERAS):
        ruta_modelo = RUTA_MODELO_KERAS
    elif os.path.exists(RUTA_MODELO_H5):
        ruta_modelo = RUTA_MODELO_H5

    if ruta_modelo:
        try:
            logger.info(f"Cargando modelo personalizado desde: {ruta_modelo}")
            import tensorflow as tf
            _modelo_propio_colab = tf.keras.models.load_model(ruta_modelo)
            # Warm-up del modelo personalizado
            dummy_input = np.zeros((1, 224, 224, 3), dtype=np.float32)
            _ = _modelo_propio_colab.predict(dummy_input, verbose=0)
            _modelo_listo = True
            logger.info("¡Modelo personalizado de Google Colab cargado exitosamente!")
            return
        except Exception as e:
            logger.error(f"Error al cargar modelo personalizado: {e}. Usando DeepFace como respaldo.", exc_info=True)

    # 2. Respaldo o modelo estándar con DeepFace
    try:
        from deepface import DeepFace

        dummy_img = np.zeros((224, 224, 3), dtype=np.uint8)
        cv2.circle(dummy_img, (112, 112), 60, (200, 200, 200), -1)
        cv2.circle(dummy_img, (90, 95), 8, (50, 50, 50), -1)
        cv2.circle(dummy_img, (134, 95), 8, (50, 50, 50), -1)
        cv2.ellipse(dummy_img, (112, 135), (25, 12), 0, 0, 180, (50, 50, 50), 3)

        _ = DeepFace.analyze(
            img_path=dummy_img,
            actions=["emotion"],
            enforce_detection=False,
            align=True,
            expand_percentage=10,
            silent=True,
        )

        _modelo_listo = True
        logger.info("¡Modelo de emociones DeepFace optimizado listo para inferencia!")
    except Exception as e:
        logger.error(f"Error durante la precarga del modelo: {e}", exc_info=True)
        _modelo_listo = True


def _validar_presencia_rostro(img_bgr: np.ndarray, analisis: dict) -> bool:
    """
    Validación propia para comprobar si realmente hay un rostro presente en la imagen.
    """
    alto, ancho = img_bgr.shape[:2]
    
    conf_rostro = analisis.get("face_confidence", 0.0)
    region = analisis.get("region", {})
    rw = region.get("w", 0)
    rh = region.get("h", 0)
    
    es_pantalla_completa = (rw >= ancho * 0.95) and (rh >= alto * 0.95)
    
    if conf_rostro and conf_rostro > 0.3 and not es_pantalla_completa:
        return True

    cascade = _obtener_clasificador_rostros()
    if cascade is not None and not cascade.empty():
        gris = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
        rostros_haar = cascade.detectMultiScale(
            gris,
            scaleFactor=1.1,
            minNeighbors=4,
            minSize=(int(ancho * 0.12), int(alto * 0.12)),
        )
        if len(rostros_haar) > 0:
            return True

    if region.get("left_eye") is not None and region.get("right_eye") is not None:
        return True

    if conf_rostro and conf_rostro > 0.0 and not es_pantalla_completa:
        return True

    return False


def _inferir_con_deepface_tta(img_mejorada: np.ndarray) -> dict:
    """
    Ejecuta inferencia con Test-Time Augmentation (TTA):
    Promedia las predicciones de la imagen original y la versión espejo horizontal.
    Aplica alineación ocular (align=True) y expansión de borde.
    """
    from deepface import DeepFace

    # 1. Pase original
    res_orig = DeepFace.analyze(
        img_path=img_mejorada,
        actions=["emotion"],
        enforce_detection=False,
        align=True,
        expand_percentage=10,
        silent=True,
    )
    analisis_orig = res_orig[0] if isinstance(res_orig, list) else res_orig

    # 2. Pase invertido (TTA espejo)
    img_flip = cv2.flip(img_mejorada, 1)
    res_flip = DeepFace.analyze(
        img_path=img_flip,
        actions=["emotion"],
        enforce_detection=False,
        align=True,
        expand_percentage=10,
        silent=True,
    )
    analisis_flip = res_flip[0] if isinstance(res_flip, list) else res_flip

    # Promediar probabilidades TTA
    emociones_1 = analisis_orig.get("emotion", {})
    emociones_2 = analisis_flip.get("emotion", {})
    
    emociones_promedio = {}
    todas_claves = set(emociones_1.keys()).union(emociones_2.keys())
    for k in todas_claves:
        val1 = float(emociones_1.get(k, 0.0))
        val2 = float(emociones_2.get(k, 0.0))
        emociones_promedio[k] = (val1 + val2) / 2.0

    return {
        "emotion": emociones_promedio,
        "region": analisis_orig.get("region", {}),
        "face_confidence": max(analisis_orig.get("face_confidence", 0.0), analisis_flip.get("face_confidence", 0.0)),
    }


def detectar_emocion(imagen_bytes: bytes) -> dict:
    """
    Función principal de análisis de emociones de alta precisión.
    """
    respuesta_sin_rostro = {
        "emocion": None,
        "confianza": 0.0,
        "probabilidades": {
            "feliz": 0.0,
            "triste": 0.0,
            "neutral": 0.0,
            "enojado": 0.0,
            "sorprendido": 0.0,
            "miedo": 0.0,
            "disgusto": 0.0,
        },
        "rostro_detectado": False,
        "estado_academico": "no determinado",
        "mensaje": "No se detectó ningún rostro en la imagen. Por favor, encuadra tu rostro de frente y con buena iluminación.",
    }

    try:
        # Decodificación en memoria
        np_arr = np.frombuffer(imagen_bytes, np.uint8)
        img_bgr = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

        if img_bgr is None or img_bgr.size == 0:
            try:
                pil_img = Image.open(io.BytesIO(imagen_bytes)).convert("RGB")
                img_bgr = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)
            except Exception:
                logger.warning("Fallo al decodificar la imagen.")
                resp = respuesta_sin_rostro.copy()
                resp["mensaje"] = "El archivo enviado no es una imagen válida o está dañado."
                return resp

        # 1. Aplicar preprocesamiento de mejora visual (CLAHE + Denoising)
        img_optimizada = mejorar_calidad_imagen(img_bgr)

        # 2. Inferencia con Modelo Personalizado de Colab o DeepFace TTA
        if _modelo_propio_colab is not None:
            # Modelo Colab
            cascade = _obtener_clasificador_rostros()
            gris = cv2.cvtColor(img_optimizada, cv2.COLOR_BGR2GRAY)
            rostros = cascade.detectMultiScale(gris, 1.1, 4) if cascade is not None else []
            if len(rostros) == 0:
                return respuesta_sin_rostro
            x, y, w, h = rostros[0]
            cara_recortada = cv2.resize(img_optimizada[y:y+h, x:x+w], (224, 224))
            inp = np.expand_dims(cara_recortada / 255.0, axis=0)
            preds = _modelo_propio_colab.predict(inp, verbose=0)[0]
            etiquetas = ["enojado", "disgusto", "miedo", "feliz", "triste", "sorprendido", "neutral"]
            emociones_raw = {etiquetas[i]: float(preds[i]) for i in range(len(etiquetas))}
            analisis = {"face_confidence": 0.95, "region": {"x": int(x), "y": int(y), "w": int(w), "h": int(h)}}
        else:
            # DeepFace con Test-Time Augmentation y alineación
            analisis = _inferir_con_deepface_tta(img_optimizada)
            emociones_raw = analisis.get("emotion", {})

        # 3. Validación de presencia de rostro
        rostro_presente = _validar_presencia_rostro(img_bgr, analisis)
        if not rostro_presente:
            logger.info("Validación propia: no se detectó un rostro válido en la captura.")
            return respuesta_sin_rostro

        # 4. Calibración y normalización de probabilidades
        # Aplicamos pesos de calibración para corregir sesgos de sombras en webcam
        probabilidades_ponderadas = {}
        for key_en, valor in emociones_raw.items():
            k_lower = key_en.lower()
            peso = PESOS_CALIBRACION.get(k_lower, 1.0)
            probabilidades_ponderadas[k_lower] = max(0.0, float(valor) * peso)

        suma_ponderada = sum(probabilidades_ponderadas.values()) or 1.0

        probabilidades_normalizadas = {}
        for key_en, valor in probabilidades_ponderadas.items():
            key_es = TRADUCCION_EMOCIONES.get(key_en.lower(), key_en.lower())
            prob = round(float(valor) / suma_ponderada, 4)
            probabilidades_normalizadas[key_es] = max(0.0, min(1.0, prob))

        # Asegurar las 7 emociones estándar
        for em in ["feliz", "triste", "neutral", "enojado", "sorprendido", "miedo", "disgusto"]:
            if em not in probabilidades_normalizadas:
                probabilidades_normalizadas[em] = 0.0

        # Determinar emoción dominante con probabilidades calibradas
        emocion_es = max(probabilidades_normalizadas, key=probabilidades_normalizadas.get)
        confianza = probabilidades_normalizadas[emocion_es]

        # Mapeo a estado académico
        estado_academico = MAPEO_ESTADO_ACADEMICO.get(emocion_es, "estable")

        logger.info(f"Emoción detectada: {emocion_es} (confianza: {confianza:.2f}) - Estado: {estado_academico}")

        return {
            "emocion": emocion_es,
            "confianza": confianza,
            "probabilidades": probabilidades_normalizadas,
            "rostro_detectado": True,
            "estado_academico": estado_academico,
            "mensaje": "Análisis completado exitosamente",
        }

    except Exception as e:
        logger.error(f"Error inesperado durante el análisis de emoción: {e}", exc_info=True)
        resp = respuesta_sin_rostro.copy()
        resp["mensaje"] = f"Error al procesar el rostro: {str(e)}"
        return resp
