# 📊 Sistema Inteligente de Supervisión de Exámenes & Evaluación Predictiva
## Guía Completa de Diapositivas para Exposición (Módulo IMAGEN & Módulo PREDICCIÓN)

> **Estructura oficial para la defensa académica:** Integra el procesamiento digital y modelo de visión de **IMAGEN** con el motor de inferencia y scoring de **PREDICCIÓN** (sin tocar el módulo de VIDEO).

---

## 📑 Índice de Diapositivas

1. **Carátula:** Título del Proyecto y Presentación
2. **Contexto del Sistema:** La cadena de valor entre IMAGEN y PREDICCIÓN
3. **Módulo IMAGEN:** Procesamiento digital y modelo entrenado YOLOv8 (`best.pt`)
4. **Módulo IMAGEN:** Extracción de características espaciales y métricas de calidad
5. **Módulo PREDICCIÓN:** El núcleo de Machine Learning para estimación de riesgo de fraude
6. **Algoritmo Predictivo:** Fusión probabilística y matriz de clasificación (Crítico / Moderado / Normal)
7. **Explicabilidad Predictiva:** Cómo la IA justifica sus decisiones ante el docente
8. **Doble Modalidad Operativa:** Análisis de fotograma individual y procesamiento en lote (CSV/JSON)
9. **Módulo Complementario:** Estimación de atención y estado afectivo tutorial
10. **Privacidad por Diseño:** Procesamiento volátil en memoria RAM (Zero Disk Footprint)
11. **Arquitectura Tecnológica:** Stack FastAPI, Ultralytics YOLOv8, MediaPipe y Despliegue en Render
12. **Demostración Práctica:** Caso de prueba con elementos no permitidos y situación regular
13. **Conclusiones & Impacto:** Cumplimiento de objetivos y articulación intermodular

---

---

### 🖥️ Diapositiva 1: Carátula / Portada

#### 📌 Contenido de la Diapositiva:
- **Título:** Sistema de Supervisión Inteligente de Evaluaciones
- **Subtítulo:** Integración de Procesamiento Digital (Módulo IMAGEN) y Modelado de Riesgo de Fraude (Módulo PREDICCIÓN)
- **Área temática:** Visión por Computadora & Machine Learning Aplicado
- **Presentado por:** [Tu Nombre / Equipo de Predicción]
- **Tecnologías clave:** YOLOv8 (`best.pt`), Ultralytics, FastAPI, MediaPipe, OpenCV (CLAHE), DeepFace

#### 🗣️ Guion del Expositor (Qué decir):
> *"Buenas tardes profesor y compañeros. Hoy les presento la integración de nuestro **Módulo de PREDICCIÓN** con el **Módulo de IMAGEN** dentro del Sistema de Supervisión de Exámenes. En esta entrega no abordamos la capa de streaming de video; nos enfocamos rigurosamente en lo que demanda la visión computacional y el modelado analítico: tomar las capturas del examen, extraer características geométricas de alta fidelidad con el modelo YOLO entrenado por el equipo y, a través de nuestro motor de inferencia predictiva, determinar con base matemática si existe una sospecha fundada de fraude o una situación académica regular."*

---

### 🖥️ Diapositiva 2: Contexto y Articulación IMAGEN ➔ PREDICCIÓN

#### 📌 Contenido de la Diapositiva:
- **La división de responsabilidades técnicas:**
  - **Módulo IMAGEN:** Captura, limpia con filtros, ejecuta la red neuronal (`best.pt`) y extrae variables cuantitativas del entorno del estudiante.
  - **Módulo PREDICCIÓN:** Es el cerebro decisional. Recibe las variables de imagen, calcula probabilidades condicionadas y emite un dictamen pedagógico accionable.
- **Flujo de datos:**
  ```
  [Fotograma / Alerta] ──► [Filtros CLAHE & YOLO best.pt] ──► [Vector de Características] ──► [Modelo PREDICCIÓN] ──► [Diagnóstico & Riesgo %]
  ```

#### 🗣️ Guion del Expositor (Qué decir):
> *"Es fundamental entender cómo se conectan ambos módulos. El equipo de IMAGEN no toma decisiones sobre si hubo copia o no; su labor es limpiar la señal visual, segmentar y medir variables como distancia y solapamiento. Nuestro grupo de PREDICCIÓN es quien toma esa matriz de datos y calcula la probabilidad de riesgo. De esta forma, desacoplamos la visión pura de la lógica analítica, permitiendo un sistema robusto, explicable y modular."*

---

### 🖥️ Diapositiva 3: Módulo IMAGEN - Detección con Modelo Entrenado (`best.pt`)

#### 📌 Contenido de la Diapositiva:
- **Modelo de Visión Artificial Utilizado:**
  - Arquitectura: **YOLOv8 nano** optimizada para inferencia rápida.
  - Pesos: `best.pt` (6.25 MB), entrenado con 50 epochs en dataset de supervisión.
- **8 Clases reconocidas en la escena:**
  - **Objetos Restringidos (Infracciones):** `telefono`, `audifonos`, `cuaderno`, `libro`, `reloj`.
  - **Objetos Permitidos / Contextuales:** `persona`, `laptop`, `mochila`.
- **Preprocesamiento Óptimo:**
  - Reducción de ruido y ecualización adaptativa CLAHE en espacio de color LAB para compensar sombras de cámara web.

#### 🗣️ Guion del Expositor (Qué decir):
> *"En el módulo de IMAGEN integramos el modelo YOLOv8 entrenado por nuestros compañeros, guardado en el archivo `best.pt`. Este modelo fue entrenado específicamente para supervisión de exámenes sobre 8 clases clave. Cuando una imagen entra al sistema, primero se normaliza con un pipeline de filtrado bilateral y CLAHE para que la iluminación pobre no degrade la detección, y luego YOLO localiza con precisión si hay celulares, libros o audífonos presentes."*

---

### 🖥️ Diapositiva 4: Extracción de Características Espaciales de la Imagen

#### 📌 Contenido de la Diapositiva:
- **Variables extraídas por IMAGEN para alimentar la PREDICCIÓN:**
  1. **Solapamiento Corporal (`proporcion_dentro_persona`):** Porcentaje del área del objeto que coincide con el cuerpo del alumno (0% a 100%).
  2. **Distancia Euclidiana (`distancia_a_persona`):** Proximidad física normalizada entre el centro del objeto y el estudiante.
  3. **Confianza Visual de Detección:** Probabilidad directa de la red neuronal.
  4. **Calidad del Fotograma:** Brillo medio, contraste y nitidez (varianza del Laplaciano) para evitar falsos positivos por desenfoque.

#### 🗣️ Guion del Expositor (Qué decir):
> *"Detectar un celular en una foto no es suficiente para acusar a un alumno de fraude; el celular podría estar sobre una repisa al fondo de la habitación. Por eso, el módulo de imagen extrae relaciones espaciales: calcula matemáticamente la intersección entre la caja del objeto y la caja de la persona, y la distancia euclidiana entre ambos. Un celular a 5 centímetros de la mano tiene un significado radicalmente distinto a uno ubicado a dos metros de distancia."*

---

### 🖥️ Diapositiva 5: Módulo PREDICCIÓN - Motor de Inferencia de Riesgo de Fraude

#### 📌 Contenido de la Diapositiva:
- **El rol de la PREDICCIÓN:**
  - Transformar mediciones visuales ambiguas en una evaluación de riesgo certera.
- **Ponderaciones de Severidad del Objeto:**
  - **Teléfono Móvil / Celular:** Severidad 1.0 (Infracción de máxima prioridad).
  - **Audífonos / Dispositivos de Audio:** Severidad 0.95 (Comunicación externa no autorizada).
  - **Libro / Cuaderno:** Severidad 0.85 (Material de apoyo prohibido).
  - **Reloj / Smartwatch:** Severidad 0.65 (Potencial dispositivo digital).
- **Moduladores Espaciales Dinámicos:**
  - Objeto en área corporal (`> 35% solapamiento`): Factor de amplificación **+35%**.
  - Objeto al alcance inmediato (`distancia < 0.25`): Factor **+15%**.
  - Objeto distante (`distancia > 0.60`): Atenuación preventiva.

#### 🗣️ Guion del Expositor (Qué decir):
> *"Aquí entra nuestro desarrollo principal: el motor de PREDICCIÓN. Asignamos pesos de severidad diferenciados según el tipo de elemento no permitido. Un teléfono o auricular tiene mayor peso crítico que un reloj. A ese puntaje base le aplicamos los moduladores espaciales extraídos de la imagen: si el objeto está dentro de las coordenadas de las manos o el pecho del estudiante, el riesgo se amplifica drásticamente."*

---

### 🖥️ Diapositiva 6: Fusión Probabilística y Matriz de Clasificación

#### 📌 Contenido de la Diapositiva:
- **Modelo de Fusión Probabilística (Ensemble Soft-OR):**
  $$P_{\text{fraude}} = 1 - \prod_{i=1}^{n} (1 - S_i)$$
  Donde $S_i$ es el score individual de cada objeto detectado.
- **Niveles Predictivos Estandarizados:**
  - 🚨 **Fraude Inminente / Riesgo Crítico ($P \ge 75\%$):** Elemento prohibido en posesión directa activa.
  - ⚠️ **Conducta Sospechosa / Riesgo Moderado ($40\% \le P < 75\%$):** Elemento prohibido visible o cercano.
  - 🟢 **Situación Normal / Sin Infracción ($P < 40\%$):** Entorno despejado o elementos de contexto permitidos (laptop/mochila).

#### 🗣️ Guion del Expositor (Qué decir):
> *"Para consolidar múltiples detecciones simultáneas (por ejemplo, si hay un celular y un cuaderno a la vez), implementamos un ensamble probabilístico Soft-OR. Si se detectan varios objetos infractores, sus probabilidades se combinan de forma no lineal. El resultado final se clasifica en tres niveles estandarizados: Crítico, Moderado o Normal, con umbrales rigurosos para evitar acusaciones falsas."*

---

### 🖥️ Diapositiva 7: Explicabilidad y Justificación de la IA

#### 📌 Contenido de la Diapositiva:
- **IA Explicable (XAI):**
  - La predicción nunca es una 'caja negra'.
- **Elementos entregados al docente/supervisor:**
  - **Dictamen pedagógico formal:** Resumen en lenguaje natural del hallazgo.
  - **Lista de factores determinantes:** Evidencia cuantitativa puntual (*"Teléfono móvil sostenido en área corporal con 92% de confianza visual"*).
  - **Recomendación de intervención:** Guía sobre si llamar la atención, solicitar encuadre o dar continuidad a la prueba.

#### 🗣️ Guion del Expositor (Qué decir):
> *"Un aporte ético y técnico fundamental de nuestro módulo es la Explicabilidad. Ningún docente aceptaría un sistema que solo marque 'Copia: Sí' sin dar razones. Nuestro algoritmo desglosa exactamente qué variables de la imagen motivaron la alerta, indicando la clase del objeto, su distancia física y la confianza del detector, permitiendo al evaluador tomar una decisión informada y justa."*

---

### 🖥️ Diapositiva 8: Doble Modalidad Operativa del Sistema

#### 📌 Contenido de la Diapositiva:
- **1. Modo Fotograma Individual / Cámara en Vivo:**
  - Toma una fotografía automática (estabilidad de 1 s con MediaPipe) o sube un fotograma JPEG/PNG.
  - Devuelve la foto anotada con cuadros de colores en tiempo real y el panel predictivo de riesgo.
- **2. Modo Reporte por Lote (Ingesta directa de IMAGEN):**
  - Carga los archivos `resultados_alertas.csv` o `resultados_alertas.json` exportados por el módulo IMAGEN.
  - Genera una tabla analítica consolidada con métricas globales: total de alertas, fraudes confirmados y tasa de riesgo.

#### 🗣️ Guion del Expositor (Qué decir):
> *"Para demostrar la interoperabilidad entre grupos, dotamos al sistema de dos modos. En el primero, probamos en tiempo real con la cámara o subiendo una foto para ver la predicción instantánea. En el segundo, cargamos directamente el archivo CSV o JSON que exporta el grupo de IMAGEN con sus alertas procesadas; nuestro backend predice automáticamente el riesgo de cada alerta y presenta un tablero de mando con KPIs para el evaluador."*

---

### 🖥️ Diapositiva 9: Módulo Complementario - Estado Tutorial y Emocional

#### 📌 Contenido de la Diapositiva:
- **Valor agregado para el acompañamiento tutorial:**
  - Inferencia facial complementaria mediante DeepFace sobre el rostro del evaluado.
  - Identificación de estados afectivos: Concentración/Neutral, Ansiedad/Miedo, Frustración/Enojo.
- **Mapeo pedagógico:**
  - Permite distinguir si un comportamiento errático se debe a deshonestidad académica o a bloqueo por estrés/ansiedad ante la prueba.

#### 🗣️ Guion del Expositor (Qué decir):
> *"Como valor agregado, mantuvimos integrado el análisis facial que desarrollamos previamente. Esto le da una dimensión humana a la herramienta: ante un estudiante inquieto, el sistema no solo busca celulares, sino que evalúa su estado emocional. Esto ayuda al tutor a identificar si el alumno está intentando copiar o si está sufriendo un bloqueo emocional por estrés o frustración ante el examen."*

---

### 🖥️ Diapositiva 10: Privacidad por Diseño (Privacy by Design)

#### 📌 Contenido de la Diapositiva:
- **Cero almacenamiento en disco:**
  - Las fotografías y fotogramas se procesan estrictamente en la memoria RAM volátil del servidor.
- **Sin bases de datos de imágenes:**
  - La imagen se decodifica con OpenCV, se analiza con YOLO y se descarta inmediatamente tras responder la petición HTTP.
- **Conformidad con normativas de protección de datos personales.**

#### 🗣️ Guion del Expositor (Qué decir):
> *"Un punto crítico en la evaluación institucional es la privacidad. Siguiendo el principio de 'Privacy by Design', nuestro sistema no guarda ninguna foto en disco duro ni almacena imágenes en bases de datos. Los bytes viajan por memoria RAM, se infieren y se liberan de inmediato. Lo único que se conserva son las métricas matemáticas resultantes."*

---

### 🖥️ Diapositiva 11: Arquitectura de Software y Despliegue

#### 📌 Contenido de la Diapositiva:
- **Backend:** FastAPI (Python 3.11), Uvicorn, Ultralytics YOLOv8, PyTorch, DeepFace, OpenCV.
- **Frontend:** HTML5 semántico, CSS moderno sobrio e institucional, JavaScript Vanilla (MediaPipe Face Detection CDN).
- **Despliegue e Infraestructura:**
  - Repositorios separados en GitHub: `Prediccion_Backend` y `Prediccion_Frontend`.
  - Despliegue en la nube con Render (Web Service y Static Site) optimizado para operar en 512 MB de RAM con recolección de basura activa (`gc.collect()`).

#### 🗣️ Guion del Expositor (Qué decir):
> *"A nivel de ingeniería, la solución utiliza FastAPI en el backend por su alta velocidad asíncrona y tipado estricto. Separamos el proyecto en dos repositorios limpios en GitHub para facilitar su despliegue en Render. Además, optimizamos el consumo de memoria para que la red neuronal y los filtros operen fluidamente sin saturar los recursos de servidores en la nube."*

---

### 🖥️ Diapositiva 12: Demostración Práctica del Sistema

#### 📌 Contenido de la Diapositiva:
- **Caso 1: Evaluación Regular:**
  - Estudiante frente a la pantalla. Detección: `persona` (91%).
  - Resultado: **Probabilidad 5% - Riesgo Normal 🟢**.
- **Caso 2: Infracción con Teléfono Móvil:**
  - Estudiante con teléfono en mano. Detección: `persona` (89%), `telefono` (92%).
  - Resultado: **Probabilidad 94% - Fraude Crítico 🚨**.
  - Evidencia: *"Teléfono en área corporal del estudiante"*.
- **Caso 3: Carga de Reporte CSV/JSON de IMAGEN:**
  - Carga masiva de alertas con cálculo automático de KPIs.

#### 🗣️ Guion del Expositor (Qué decir):
> *"En la demostración práctica mostramos los tres escenarios. Primero, un alumno realizando su prueba con la mesa despejada: el modelo valida la presencia de la persona y marca riesgo normal. Segundo, al colocar un celular o audífonos en el cuadro, el modelo `best.pt` lo enmarca en rojo y el medidor predictivo salta a Riesgo Crítico explicando el motivo exacto. Y tercero, mostramos la ingesta por lotes del archivo de alertas generado por el grupo de IMAGEN."*

---

### 🖥️ Diapositiva 13: Conclusiones y Logros Clave

#### 📌 Contenido de la Diapositiva:
- **Logros Alcanzados:**
  1. Integración exitosa del modelo YOLO `best.pt` de IMAGEN en el pipeline de PREDICCIÓN.
  2. Implementación de un modelo predictivo probabilístico con clasificación en 3 niveles de riesgo.
  3. Explicabilidad transparente para respaldo tutorial y pedagógico.
  4. Interfaz web institucional con soporte de cámara, carga de imagen y auditoría por lotes.
  5. Despliegue funcional en producción en la nube.

#### 🗣️ Guion del Expositor (Qué decir):
> *"Para concluir: logramos articular de forma impecable el trabajo de visión artificial de IMAGEN con la inteligencia predictiva de PREDICCIÓN. El sistema no solo detecta objetos, sino que entiende el contexto espacial y traduce los datos en decisiones confiables, éticas y fundamentadas para la supervisión académica. Muchas gracias por su atención, quedo a disposición para sus preguntas."*
