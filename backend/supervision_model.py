"""
AulaPredict - Módulo PREDICCIÓN & Supervisión de Exámenes
Archivo: supervision_model.py

Cadena de Integración:
[IMAGEN: best.pt YOLOv8 + CLAHE + Métricas] ──► [PREDICCIÓN: Modelo Predictivo de Riesgo de Fraude]

Capacidades:
1. Detección en memoria de objetos no permitidos (YOLOv8 con pesos 'best.pt' entrenados por el equipo).
2. Procesamiento de imagen: evaluación de calidad (brillo, contraste, nitidez) y ecualización adaptativa.
3. Extracción de características espaciales respecto a la persona (solapamiento corporal, distancia Euclidiana).
4. Motor de Inferencia Predictiva:
   - Probabilidad estimada de fraude/infracción (0 a 100%).
   - Clasificación de riesgo: Crítico (🚨), Moderado (⚠️), Normal (🟢).
   - Explicabilidad y justificación automática de la predicción.
5. Predicción en lote a partir de archivos JSON o CSV exportados por el módulo IMAGEN.
"""

import io
import os
import cv2
import base64
import logging
import numpy as np
from PIL import Image
from typing import Dict, Any, List, Optional, Tuple

logger = logging.getLogger("aulapredict.supervision_model")

# Ruta del modelo YOLO entrenado por el equipo
RUTA_MODELO_YOLO = os.path.join(os.path.dirname(__file__), "modelo_ia", "best.pt")

# Clases reconocidas por el modelo YOLOv8 entrenado por el equipo
CLASES_MODELO = [
    "persona",
    "mochila",
    "telefono",
    "cuaderno",
    "libro",
    "audifonos",
    "reloj",
    "laptop",
]

# Objetos considerados infracción / no permitidos durante el examen
OBJETOS_NO_PERMITIDOS = {
    "telefono": {"severidad": 1.0, "nombre": "Teléfono Móvil", "riesgo_base": 0.85},
    "celular": {"severidad": 1.0, "nombre": "Teléfono Móvil", "riesgo_base": 0.85},
    "phone": {"severidad": 1.0, "nombre": "Teléfono Móvil", "riesgo_base": 0.85},
    "audifonos": {"severidad": 0.95, "nombre": "Audífonos / Dispositivo de Audio", "riesgo_base": 0.80},
    "airpods": {"severidad": 0.95, "nombre": "Audífonos Inalámbricos", "riesgo_base": 0.80},
    "cuaderno": {"severidad": 0.85, "nombre": "Cuaderno / Apuntes", "riesgo_base": 0.70},
    "libro": {"severidad": 0.85, "nombre": "Libro de Consulta", "riesgo_base": 0.70},
    "cuaderno/libro": {"severidad": 0.85, "nombre": "Cuaderno o Libro", "riesgo_base": 0.70},
    "reloj": {"severidad": 0.65, "nombre": "Reloj / Smartwatch", "riesgo_base": 0.50},
}

# Objetos contextuales / permitidos
OBJETOS_PERMITIDOS = {"persona", "mochila", "laptop"}

# Umbrales específicos por clase (calibrados acordes al módulo IMAGEN)
# Teléfono, audífonos y reloj se detectan con alta sensibilidad (>= 0.15)
UMBRALES_POR_CLASE = {
    "persona": 0.20,
    "telefono": 0.15,
    "celular": 0.15,
    "phone": 0.15,
    "audifonos": 0.15,
    "airpods": 0.15,
    "reloj": 0.15,
    "mochila": 0.15,
    "laptop": 0.25,
    "cuaderno": 0.55,
    "libro": 0.55,
}
UMBRAL_CORTE_GLOBAL = 0.15

# Modelo YOLO singleton en memoria
_modelo_yolo = None
_modelo_listo = False


def obtener_modelo_yolo():
    """Carga y almacena en caché el modelo YOLOv8 entrenado (best.pt)."""
    global _modelo_yolo, _modelo_listo
    if _modelo_yolo is None:
        if not os.path.exists(RUTA_MODELO_YOLO):
            logger.warning(f"No se encontró el modelo en {RUTA_MODELO_YOLO}")
            return None
        try:
            import torch
            es_render = os.environ.get("RENDER") == "true" or "onrender.com" in os.environ.get("RENDER_EXTERNAL_HOSTNAME", "")
            if es_render:
                torch.set_num_threads(2)
            os.environ["YOLO_VERBOSE"] = "False"
            from ultralytics import YOLO
            logger.info(f"Cargando modelo YOLO de supervisión desde: {RUTA_MODELO_YOLO}")
            _modelo_yolo = YOLO(RUTA_MODELO_YOLO)
            _modelo_listo = True
            logger.info("¡Modelo YOLO (best.pt) cargado exitosamente!")
        except Exception as e:
            logger.error(f"Error al cargar YOLO: {e}", exc_info=True)
            return None
    return _modelo_yolo


def esta_modelo_supervision_listo() -> bool:
    """Verifica si el modelo YOLO está disponible y cargado."""
    global _modelo_listo
    if not _modelo_listo and os.path.exists(RUTA_MODELO_YOLO):
        obtener_modelo_yolo()
    return _modelo_listo


def medir_calidad_imagen(img_bgr: np.ndarray) -> Dict[str, float]:
    """Calcula métricas de calidad de imagen acordes al pipeline del módulo IMAGEN."""
    gris = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    brillo = float(np.mean(gris))
    contraste = float(np.std(gris))
    nitidez = float(cv2.Laplacian(gris, cv2.CV_64F).var())
    return {
        "brillo": round(brillo, 2),
        "contraste": round(contraste, 2),
        "nitidez": round(nitidez, 2),
    }


def preprocesar_y_mejorar_imagen(img_bgr: np.ndarray) -> np.ndarray:
    """
    Aplica técnicas de mejoramiento del módulo IMAGEN:
    1. Ajuste adaptativo de contraste CLAHE en luminancia (LAB) para rescatar objetos en sombras o contraluz.
    2. Máscara de enfoque (unsharp masking) para acentuar bordes y siluetas de dispositivos y manos.
    """
    try:
        # CLAHE en luminancia (espacio LAB) para no distorsionar colores
        lab = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        l_clahe = clahe.apply(l)
        lab_clahe = cv2.merge((l_clahe, a, b))
        img_clahe = cv2.cvtColor(lab_clahe, cv2.COLOR_LAB2BGR)

        # Máscara de enfoque para bordes nítidos
        suavizada = cv2.GaussianBlur(img_clahe, (0, 0), sigmaX=1.8)
        img_enfocada = cv2.addWeighted(img_clahe, 1.25, suavizada, -0.25, 0)
        return img_enfocada
    except Exception as e:
        logger.warning(f"Error en mejoramiento de imagen, usando original: {e}")
        return img_bgr


def calcular_interseccion_y_distancia(
    caja_obj: List[float],
    caja_persona: List[float],
    ancho_img: int,
    alto_img: int
) -> Tuple[float, float]:
    """
    Calcula:
    1. Proporción del objeto dentro del área corporal de la persona (0.0 a 1.0).
    2. Distancia Euclidiana normalizada entre centros (0.0 a 1.0).
    """
    ox1, oy1, ox2, oy2 = caja_obj
    px1, py1, px2, py2 = caja_persona

    # Área del objeto
    area_obj = max(0.0, ox2 - ox1) * max(0.0, oy2 - oy1)
    if area_obj <= 0:
        return 0.0, 1.0

    # Intersección
    ix1 = max(ox1, px1)
    iy1 = max(oy1, py1)
    ix2 = min(ox2, px2)
    iy2 = min(oy2, py2)
    inter_ancho = max(0.0, ix2 - ix1)
    inter_alto = max(0.0, iy2 - iy1)
    area_inter = inter_ancho * inter_alto

    proporcion_dentro = area_inter / area_obj

    # Distancia entre centros normalizada
    ocx = (ox1 + ox2) / 2.0 / ancho_img
    ocy = (oy1 + oy2) / 2.0 / alto_img
    pcx = (px1 + px2) / 2.0 / ancho_img
    pcy = (py1 + py2) / 2.0 / alto_img

    distancia = float(np.hypot(ocx - pcx, ocy - pcy))
    return round(proporcion_dentro, 3), round(distancia, 3)


def predecir_riesgo_fraude(
    objetos_detectados: List[Dict[str, Any]],
    calidad_img: Dict[str, float]
) -> Dict[str, Any]:
    """
    Modelo de Inferencia Predictiva de AulaPredict:
    Calcula la probabilidad de fraude en el examen y clasifica el riesgo.
    """
    if not objetos_detectados:
        return {
            "nivel_riesgo": "normal",
            "etiqueta_riesgo": "Sin Actividad Sospechosa",
            "probabilidad_fraude": 0.0,
            "color_estado": "green",
            "dictamen": "No se detectaron objetos ni irregularidades en la imagen analizada.",
            "factores_clave": ["Zona de trabajo despejada."],
            "objetos_infractores": [],
            "recomendacion": "Continuar con la evaluación normalmente."
        }

    # Separar personas de objetos
    personas = [o for o in objetos_detectados if o["clase"].lower() == "persona"]
    infractores = []
    probabilidad_acumulada = 0.0
    factores = []

    for obj in objetos_detectados:
        clase = obj["clase"].lower()
        if clase in OBJETOS_NO_PERMITIDOS:
            meta = OBJETOS_NO_PERMITIDOS[clase]
            conf = obj["confianza"]
            prop_dentro = obj.get("proporcion_dentro_persona", 0.0)
            dist_persona = obj.get("distancia_a_persona", 0.5)

            # Algoritmo de scoring predictivo
            score_base = meta["riesgo_base"] * conf

            # Modificador espacial: si el objeto está en posesión directa
            if prop_dentro > 0.35:
                modificador_espacio = 1.35
                motivo_posicion = f"{meta['nombre']} sostenido directamente en el área corporal (solapamiento {int(prop_dentro*100)}%)"
            elif dist_persona is not None and dist_persona < 0.25:
                modificador_espacio = 1.15
                motivo_posicion = f"{meta['nombre']} al alcance inmediato de la mano (distancia {dist_persona})"
            else:
                modificador_espacio = 0.75
                motivo_posicion = f"{meta['nombre']} presente en la escena (distancia {dist_persona or 'N/A'})"

            score_objeto = min(1.0, score_base * modificador_espacio)
            infractores.append({
                "objeto": meta["nombre"],
                "clase_raw": clase,
                "confianza": conf,
                "score_infraccion": round(score_objeto, 2),
                "posicion": motivo_posicion
            })
            factores.append(f"{motivo_posicion} con confianza del {int(conf * 100)}%.")

    if not infractores:
        tiene_persona = len(personas) > 0
        return {
            "nivel_riesgo": "normal",
            "etiqueta_riesgo": "Normal - Sin Infracciones",
            "probabilidad_fraude": 0.05 if tiene_persona else 0.0,
            "color_estado": "green",
            "dictamen": "Estudiante presente en la evaluación sin elementos no autorizados." if tiene_persona else "Espacio de trabajo sin elementos sospechosos.",
            "factores_clave": ["Solo se detectaron elementos autorizados."] if tiene_persona else ["Sin elementos."],
            "objetos_infractores": [],
            "recomendacion": "Evaluación regular en curso."
        }

    # Fusión probabilística (Ensemble Soft-OR)
    # P_total = 1 - (1 - P1) * (1 - P2) ...
    prod_complementos = 1.0
    for inf in infractores:
        prod_complementos *= (1.0 - inf["score_infraccion"])
    probabilidad_final = round((1.0 - prod_complementos) * 100.0, 1)

    # Clasificación predictiva
    if probabilidad_final >= 75.0:
        nivel = "critico"
        etiqueta = "🚨 Fraude Inminente / Riesgo Crítico"
        color = "red"
        dictamen = "Alta probabilidad de uso de material no autorizado durante la prueba. Evidencia contundente de infracción."
        recomendacion = "Notificar inmediatamente al docente / supervisor para verificar la pantalla o cámara del estudiante."
    elif probabilidad_final >= 40.0:
        nivel = "moderado"
        etiqueta = "⚠️ Conducta Sospechosa / Riesgo Moderado"
        color = "orange"
        dictamen = "Se detectó presencia de elementos restringidos cerca de la persona. Posible intento de consulta."
        recomendacion = "Emitir llamado de atención preventivo y monitorear los próximos fotogramas."
    else:
        nivel = "bajo"
        etiqueta = "🟡 Riesgo Bajo / Posible Falso Positivo"
        color = "yellow"
        dictamen = "Objeto detectado con baja probabilidad de interacción directa o a distancia segura."
        recomendacion = "Mantener supervisión pasiva sin interrumpir al estudiante."

    return {
        "nivel_riesgo": nivel,
        "etiqueta_riesgo": etiqueta,
        "probabilidad_fraude": probabilidad_final,
        "color_estado": color,
        "dictamen": dictamen,
        "factores_clave": factores,
        "objetos_infractores": infractores,
        "recomendacion": recomendacion
    }


def analizar_imagen_supervision(imagen_bytes: bytes) -> Dict[str, Any]:
    """
    Pipeline completo:
    1. Decodificación en memoria.
    2. Medición de calidad y CLAHE.
    3. Inferencia con YOLOv8 (best.pt).
    4. Extracción de características de IMAGEN.
    5. Inferencia del modelo de PREDICCIÓN de Fraude.
    6. Renderizado de imagen anotada en Base64.
    """
    np_arr = np.frombuffer(imagen_bytes, np.uint8)
    img_bgr = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

    if img_bgr is None or img_bgr.size == 0:
        pil_img = Image.open(io.BytesIO(imagen_bytes)).convert("RGB")
        img_bgr = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)

    alto, ancho = img_bgr.shape[:2]
    calidad = medir_calidad_imagen(img_bgr)

    modelo = obtener_modelo_yolo()
    if modelo is None:
        return {
            "exito": False,
            "mensaje": "El modelo de supervisión YOLO ('best.pt') no está disponible o no se pudo cargar.",
            "prediccion": {
                "nivel_riesgo": "no_disponible",
                "etiqueta_riesgo": "Modelo No Disponible",
                "probabilidad_fraude": 0.0,
            }
        }

    # 2. Reescalado inteligente para inferencia ultrarrápida en Render / CPU
    es_render = os.environ.get("RENDER") == "true" or "onrender.com" in os.environ.get("RENDER_EXTERNAL_HOSTNAME", "")
    max_dim = 360 if es_render else 640
    factor_escala = 1.0
    if max(alto, ancho) > max_dim:
        factor_escala = max_dim / float(max(alto, ancho))
        nuevo_w = max(16, int(ancho * factor_escala))
        nuevo_h = max(16, int(alto * factor_escala))
        img_para_yolo = cv2.resize(img_bgr, (nuevo_w, nuevo_h), interpolation=cv2.INTER_AREA)
    else:
        img_para_yolo = img_bgr

    # Preprocesamiento y mejoramiento de imagen para potenciar objetos
    img_mejorada = preprocesar_y_mejorar_imagen(img_para_yolo)

    # 3. Inferencia con YOLOv8 (usando corte sensible 0.15, imgsz calibrado e iou=0.45)
    tamanio_inferencia = 320 if es_render else 640
    try:
        import torch
        with torch.inference_mode():
            resultados = modelo.predict(
                source=img_mejorada,
                conf=UMBRAL_CORTE_GLOBAL,
                iou=0.45,
                imgsz=tamanio_inferencia,
                device="cpu",
                verbose=False
            )
    except Exception as e:
        logger.warning(f"Error en predict YOLO: {e}")
        resultados = []
    detecciones = []
    personas_cajas = []

    # Extraer detecciones aplicando umbrales calibrados por clase y reescalando a dimensiones originales
    for r in resultados:
        if r.boxes is None:
            continue
        for b in r.boxes:
            cls_id = int(b.cls[0].item())
            conf = float(b.conf[0].item())
            x1, y1, x2, y2 = [float(v) for v in b.xyxy[0].tolist()]

            # Proyectar coordenadas a resolución original si fue reescalada
            if factor_escala != 1.0:
                x1 = x1 / factor_escala
                y1 = y1 / factor_escala
                x2 = x2 / factor_escala
                y2 = y2 / factor_escala

            # Asegurar límites válidos dentro de la imagen
            x1 = max(0.0, min(float(ancho), x1))
            y1 = max(0.0, min(float(alto), y1))
            x2 = max(0.0, min(float(ancho), x2))
            y2 = max(0.0, min(float(alto), y2))

            clase_nombre = CLASES_MODELO[cls_id] if cls_id < len(CLASES_MODELO) else f"clase_{cls_id}"
            clase_key = clase_nombre.lower()

            # Umbral de corte calibrado por objeto
            umbral_min = UMBRALES_POR_CLASE.get(clase_key, 0.20)
            if conf < umbral_min:
                continue

            det = {
                "id": len(detecciones) + 1,
                "clase": clase_nombre,
                "confianza": round(conf, 3),
                "caja": [round(x1, 1), round(y1, 1), round(x2, 1), round(y2, 1)],
            }
            detecciones.append(det)
            if clase_key == "persona":
                personas_cajas.append([x1, y1, x2, y2])

    # Calcular relaciones espaciales para objetos no permitidos
    for det in detecciones:
        if det["clase"] != "persona" and personas_cajas:
            mejor_prop = 0.0
            menor_dist = 1.0
            for pcaja in personas_cajas:
                prop, dist = calcular_interseccion_y_distancia(det["caja"], pcaja, ancho, alto)
                if prop > mejor_prop:
                    mejor_prop = prop
                if dist < menor_dist:
                    menor_dist = dist
            det["proporcion_dentro_persona"] = mejor_prop
            det["distancia_a_persona"] = menor_dist
        else:
            det["proporcion_dentro_persona"] = 0.0
            det["distancia_a_persona"] = None

    # Motor de PREDICCIÓN de Fraude
    prediccion = predecir_riesgo_fraude(detecciones, calidad)

    # Dibujar imagen anotada
    img_anotada = img_bgr.copy()
    for det in detecciones:
        x1, y1, x2, y2 = [int(v) for v in det["caja"]]
        clase = det["clase"].lower()
        es_infraccion = clase in OBJETOS_NO_PERMITIDOS
        color = (36, 36, 235) if es_infraccion else ((235, 140, 36) if clase == "persona" else (76, 175, 80))

        # Cuadro delimitador
        cv2.rectangle(img_anotada, (x1, y1), (x2, y2), color, 2)

        # Etiqueta
        texto = f"{det['clase']} {int(det['confianza']*100)}%"
        (w_txt, h_txt), _ = cv2.getTextSize(texto, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        cv2.rectangle(img_anotada, (x1, y1 - 20), (x1 + w_txt + 6, y1), color, -1)
        cv2.putText(img_anotada, texto, (x1 + 3, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)

    # Codificar imagen anotada en Base64 JPEG
    _, buffer = cv2.imencode(".jpg", img_anotada, [int(cv2.IMWRITE_JPEG_QUALITY), 85])
    imagen_base64 = "data:image/jpeg;base64," + base64.b64encode(buffer).decode("utf-8")

    return {
        "exito": True,
        "total_detecciones": len(detecciones),
        "detecciones": detecciones,
        "calidad_imagen": calidad,
        "prediccion": prediccion,
        "imagen_anotada": imagen_base64,
    }


def predecir_desde_datos_lote(registros_lote: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Toma un conjunto de filas (de resultados_alertas.csv o JSON exportado por IMAGEN)
    y genera una matriz predictiva consolidada para el evaluador.
    """
    resumen_lote = []
    total_criticos = 0
    total_moderados = 0
    total_normales = 0

    for i, fila in enumerate(registros_lote):
        clase = str(fila.get("clase") or fila.get("alerta_objeto") or fila.get("objeto") or "desconocido")
        conf = float(fila.get("confianza_procesada") or fila.get("confianza_video") or fila.get("confianza") or 0.5)
        prop = float(fila.get("proporcion_dentro_persona") or 0.0)
        dist = float(fila.get("distancia_a_persona") or 0.5)

        obj_mock = [{
            "clase": clase,
            "confianza": conf,
            "proporcion_dentro_persona": prop,
            "distancia_a_persona": dist,
        }]

        calidad_mock = {
            "brillo": float(fila.get("brillo_region") or 120.0),
            "contraste": float(fila.get("contraste_region") or 40.0),
            "nitidez": float(fila.get("nitidez_region") or 150.0),
        }

        pred = predecir_riesgo_fraude(obj_mock, calidad_mock)

        if pred["nivel_riesgo"] == "critico":
            total_criticos += 1
        elif pred["nivel_riesgo"] == "moderado":
            total_moderados += 1
        else:
            total_normales += 1

        resumen_lote.append({
            "id": i + 1,
            "objeto": clase,
            "confianza": conf,
            "distancia": dist,
            "nivel_riesgo": pred["nivel_riesgo"],
            "probabilidad_fraude": pred["probabilidad_fraude"],
            "dictamen": pred["dictamen"],
            "archivo": fila.get("archivo") or fila.get("frame_id") or f"Alerta_{i+1}"
        })

    return {
        "total_alertas": len(registros_lote),
        "total_criticos": total_criticos,
        "total_moderados": total_moderados,
        "total_normales": total_normales,
        "tasa_riesgo_global": round((total_criticos / max(1, len(registros_lote))) * 100, 1),
        "resultados": resumen_lote
    }
