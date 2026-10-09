# AulaPredict - Módulo Imagen

> Sistema web inteligente de visión artificial para la detección de rostros, captura automática de una fotografía en memoria y predicción del estado emocional académico mediante DeepFace y Redes Neuronales.

---

## 📋 Descripción del Proyecto

**AulaPredict - Módulo Imagen** es una solución ligera, moderna y enfocada exclusivamente en el análisis de imágenes estáticas para entornos educativos:
- **Cero Video / Audio:** La cámara se utiliza únicamente como visor en el navegador para que el usuario encuadre su rostro. No se transmite video en streaming, no se graban archivos de video ni se captura audio.
- **Detección Frontal en el Cliente:** Mediante **MediaPipe Face Detection**, el navegador analiza que el usuario esté de frente, con buen tamaño en el cuadro (~20% del visor) y una estabilidad continua de ~1 segundo.
- **Captura Automática en Memoria:** Al cumplirse las condiciones, se toma **UNA SOLA foto** (JPEG) en un canvas en memoria y se envía al servidor mediante `fetch` (multipart/form-data).
- **Procesamiento Volátil:** El backend procesa los bytes de la imagen directamente en memoria RAM sin guardarla nunca en disco ni en bases de datos.
- **Mapeo Académico:** Clasifica el estado en:
  - ⚠️ **Posible riesgo:** Triste, Miedo, Enojado, Disgusto
  - ⚖️ **Estable:** Neutral
  - ✨ **Positivo:** Feliz, Sorprendido

---

## 📁 Estructura del Proyecto

```text
prediccion/
├── backend/
│   ├── main.py              # API FastAPI, endpoints /salud y /analizar, CORS y ciclo de vida
│   ├── emotion_model.py     # Lógica DeepFace, CLAHE, TTA, alineación, calibración y carga de modelo propio
│   └── requirements.txt     # Dependencias fijadas para compatibilidad con Python 3.10/3.11
├── frontend/
│   ├── index.html           # Estructura semántica, visor de cámara, métricas y aviso de privacidad
│   ├── style.css            # Diseño moderno oscuro, glassmorphism, animaciones y diseño responsivo
│   └── app.js               # MediaPipe Face Detection, captura en alta resolución, cooldown e historial
├── notebooks/
│   └── entrenamiento_aulapredict_colab.ipynb  # Cuaderno de Google Colab para entrenar modelo de alta precisión
└── README.md                # Documentación completa y guía de extensibilidad
```

---

## ⚙️ Requisitos de Entorno y Compatibilidad

> [!IMPORTANT]
> **Versión de Python recomendada: Python 3.10 o Python 3.11**  
> Las librerías de DeepFace y TensorFlow tienen problemas de compatibilidad en Python 3.12 y 3.14 debido a la compilación de extensiones en C y soporte de wheels.
> 
> En este equipo se ha configurado un entorno virtual en `backend/.venv` usando **Python 3.11**.

---

## 🚀 Puesta en Marcha (Paso a Paso)

### 1. Iniciar el Backend (FastAPI)

Abre una terminal en la carpeta raíz del proyecto y ejecuta:

```powershell
# 1. Navegar a la carpeta backend
cd backend

# 2. Activar el entorno virtual (si ya fue creado)
# En Windows (PowerShell):
.\.venv\Scripts\Activate.ps1
# O en CMD:
# .\.venv\Scripts\activate.bat

# NOTA: Si necesitas crear el entorno virtual desde cero con Python 3.11:
# Con uv (recomendado, rápido y automático):
# uv venv .venv --python 3.11
# uv pip install -r requirements.txt --python .venv\Scripts\python.exe
# O con Python estándar:
# py -3.11 -m venv .venv
# .\.venv\Scripts\activate
# pip install -r requirements.txt

# 3. Iniciar el servidor FastAPI con Uvicorn
uvicorn main:app --reload --port 8000
```

> **Primera Ejecución:**  
> Al iniciar el servidor por primera vez, verás el mensaje:  
> `Cargando modelo, la primera vez puede tardar...`  
> DeepFace descargará los pesos preentrenados del modelo de emociones (`facial_expression_model_weights.h5`) y ejecutará un análisis de warm-up. El endpoint `/salud` notificará al frontend cuando el modelo esté 100% listo.

Puedes verificar el estado en tu navegador:
- Salud del servicio: [http://localhost:8000/salud](http://localhost:8000/salud)
- Documentación Swagger interactiva: [http://localhost:8000/docs](http://localhost:8000/docs)

---

### 2. Iniciar el Frontend (HTML + CSS + JS Puro)

Abre una segunda terminal y sirve la carpeta `frontend`:

```powershell
# Navegar a la carpeta frontend
cd frontend

# Iniciar servidor estático con Python
python -m http.server 3000
```

Luego abre en tu navegador moderno (Chrome, Edge, Firefox, Brave):
👉 **[http://localhost:3000](http://localhost:3000)** (o abre directamente el archivo `frontend/index.html` con Live Server en VS Code / tu editor preferido).

---

## 🔄 Flujo de Uso de la Aplicación

1. **Permiso de Cámara:** Al cargar la página, se solicitará permiso para acceder a la cámara web.
2. **Encuadre:** Colócate frente a la cámara dentro de la guía visual.
3. **Detección y Estabilidad:** MediaPipe detectará tu rostro. Si el tamaño es adecuado (>= 20% del cuadro) y te mantienes quieto por **1 segundo**, la barra de estabilidad se completará.
4. **Flash y Captura:** Se producirá un destello visual ("flash"), se tomará **UNA fotografía** y se enviará en memoria al backend.
5. **Resultado Inmediato:** El servidor retornará en milisegundos:
   - Emoción dominante ("Feliz", "Triste", "Neutral", etc.)
   - Porcentaje de certeza / confianza
   - Estado académico ("positivo", "estable", "posible riesgo")
   - Barras animadas con las probabilidades de las 7 emociones.
6. **Cooldown y Prevención de Duplicados:** Se activa un temporizador de 5 segundos para evitar capturas repetidas continuas. También puedes hacer clic en **"Volver a Analizar"** en cualquier momento.
7. **Historial:** Las últimas 5 capturas se guardan en la memoria volátil de la sesión.

---

## 🔌 Cómo Reemplazar DeepFace por un Modelo Propio de Google Colab

El backend fue diseñado con **alta extensibilidad** y desacoplamiento total: toda la lógica del modelo está aislada en [emotion_model.py](file:///d:/Proyectos/PREDICCION/backend/emotion_model.py). Ni `main.py` ni el frontend necesitan ser modificados.

### Pasos para integrar un modelo propio entrenado en Colab (Keras / TensorFlow / PyTorch):

#### 1. Exporta tu modelo desde Google Colab
En tu notebook de Colab, guarda el modelo entrenado:
```python
# Ejemplo con Keras / TensorFlow
model.save("mi_modelo_emociones.keras")
# o formato H5
model.save("mi_modelo_emociones.h5")
```

#### 2. Coloca el archivo en la carpeta `backend/`
Copia `mi_modelo_emociones.keras` dentro de `backend/`.

#### 3. Modifica [backend/emotion_model.py](file:///d:/Proyectos/PREDICCION/backend/emotion_model.py)

En la función `precargar_modelo()`:
```python
import tensorflow as tf

global mi_modelo_propio
mi_modelo_propio = None

def precargar_modelo():
    global mi_modelo_propio, _modelo_listo
    logger.info("Cargando modelo personalizado entrenado en Colab...")
    mi_modelo_propio = tf.keras.models.load_model("mi_modelo_emociones.keras")
    _modelo_listo = True
    logger.info("Modelo personalizado cargado exitosamente.")
```

En la función `detectar_emocion(imagen_bytes: bytes)`:
```python
def detectar_emocion(imagen_bytes: bytes) -> dict:
    # 1. Decodificar la imagen con OpenCV
    np_arr = np.frombuffer(imagen_bytes, np.uint8)
    img_bgr = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
    
    # 2. Preprocesar según lo requerido por tu modelo (ejemplo: 48x48 escala de grises)
    gris = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    # Detectar cara y recortar
    # ...
    img_input = cv2.resize(gris, (48, 48)) / 255.0
    img_input = np.expand_dims(img_input, axis=(0, -1)) # Forma (1, 48, 48, 1)

    # 3. Predicción
    predicciones = mi_modelo_propio.predict(img_input)[0] # Array de 7 valores
    
    # 4. Asignar al diccionario estandarizado de salida
    etiquetas = ["enojado", "disgusto", "miedo", "feliz", "triste", "sorprendido", "neutral"]
    probabilidades = {etiquetas[i]: round(float(predicciones[i]), 4) for i in range(len(etiquetas))}
    
    emocion_dominante = max(probabilidades, key=probabilidades.get)
    confianza = probabilidades[emocion_dominante]
    estado_academico = MAPEO_ESTADO_ACADEMICO.get(emocion_dominante, "estable")

    # Retornar EXACTAMENTE la misma estructura
    return {
        "emocion": emocion_dominante,
        "confianza": confianza,
        "probabilidades": probabilidades,
        "rostro_detectado": True,
        "estado_academico": estado_academico,
        "mensaje": "Análisis completado exitosamente",
    }
```

---

## 🔒 Privacidad y Cumplimiento Normativo

- **Sin Persistencia en Disco:** No se crea ninguna carpeta ni archivo con las fotos tomadas en el servidor.
- **Sin Transmisión de Video:** La etiqueta `<video>` del frontend no envía stream al servidor; únicamente se extrae un fotograma puntual en canvas local.
- **Sin Audio:** Ningún permiso ni API de audio es invocado en ningún momento.
- **Historial Volátil:** El historial de las últimas 5 fotos reside exclusivamente en la memoria JavaScript del navegador y se borra al cerrar o recargar la pestaña.

---

## 🛠️ Tecnologías Empleadas

- **Backend:** Python 3.11, FastAPI, Uvicorn, DeepFace, TensorFlow / tf-keras, OpenCV Headless, Pillow, NumPy.
- **Frontend:** HTML5 Semántico, CSS3 Puro (Variables, Glassmorphism, CSS Grid), JavaScript ES6+ Puro (sin dependencias de frameworks).
- **Visión en Navegador:** MediaPipe Face Detection de Google (acelerado por WebAssembly/WebGL vía CDN).
