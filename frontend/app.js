/**
 * AulaPredict - Módulo IMAGEN & PREDICCIÓN
 * Lógica Frontend (app.js)
 * 
 * Funcionalidades:
 * - Integración de la cadena IMAGEN (YOLOv8 best.pt) y PREDICCIÓN (Evaluación de Fraude).
 * - Modos de entrada: Cámara en vivo con MediaPipe, Subida de Imagen y Carga de Reportes (CSV/JSON de IMAGEN).
 * - Extracción y despliegue de características visuales y espaciales.
 * - Tarjeta de Diagnóstico Predictivo de Fraude con termómetro y factores determinantes.
 * - Matriz de Predicción en Lote para auditoría de evaluaciones completas.
 * - Auto-fallback inteligente entre servidor local (127.0.0.1:8000) y nube Render.
 */

// ============================================================================
// CONFIGURACIÓN Y CONSTANTES
// ============================================================================
const CONFIG = {
  LOCAL_BACKEND_URL: 'http://127.0.0.1:8000',
  RENDER_BACKEND_URL: 'https://prediccion-backend-dwq9.onrender.com',
  BACKEND_URL: (window.location.hostname === 'localhost' || window.location.hostname === '127.0.0.1' || !window.location.hostname)
    ? 'http://127.0.0.1:8000'
    : 'https://prediccion-backend-dwq9.onrender.com',
  STABILITY_REQUIRED_MS: 1000,
  COOLDOWN_MS: 5000,
  MIN_FACE_SIZE_RATIO: 0.15,
  POLL_HEALTH_INTERVAL_MS: 8000,
  MAX_HISTORY_ITEMS: 10,
};

// Mapeo de emociones (compatibilidad tutorial)
const EMOCION_METADATA = {
  feliz: { emoji: '😄', colorClass: 'bar-feliz' },
  triste: { emoji: '😢', colorClass: 'bar-triste' },
  neutral: { emoji: '😐', colorClass: 'bar-neutral' },
  enojado: { emoji: '😠', colorClass: 'bar-enojado' },
  sorprendido: { emoji: '😲', colorClass: 'bar-sorprendido' },
  miedo: { emoji: '😨', colorClass: 'bar-miedo' },
  disgusto: { emoji: '🤢', colorClass: 'bar-disgusto' },
};

const ESTADO_ACADEMICO_METADATA = {
  'posible riesgo': { icon: '⚠️', class: 'academic-riesgo' },
  'estable': { icon: '⚖️', class: 'academic-estable' },
  'positivo': { icon: '✨', class: 'academic-positivo' },
  'no determinado': { icon: '❓', class: 'academic-estable' }
};

// ============================================================================
// VARIABLES DE ESTADO
// ============================================================================
const state = {
  backendOnline: false,
  modelReady: false,
  cameraActive: false,
  faceDetected: false,
  currentMode: 'camera', // 'camera' | 'upload' | 'batch'
  
  stabilityStartTime: null,
  isAnalyzing: false,
  isInCooldown: false,
  cooldownTimer: null,
  faceDisappearedDuringCooldown: false,
  
  lastResult: null,
  lastSnapshotDataUrl: null,
  history: [],
};

// ============================================================================
// REFERENCIAS AL DOM
// ============================================================================
const DOM = {
  // Pestañas de modo
  tabCamera: document.getElementById('tab-camera'),
  tabUpload: document.getElementById('tab-upload'),
  tabBatch: document.getElementById('tab-batch'),
  fileUploadImage: document.getElementById('file-upload-image'),
  fileUploadBatch: document.getElementById('file-upload-batch'),
  uploadedPreviewImg: document.getElementById('uploaded-preview-img'),
  stabilityContainer: document.getElementById('stability-container'),

  // Video y Canvas
  video: document.getElementById('webcam'),
  overlayCanvas: document.getElementById('overlay-canvas'),
  hiddenCanvas: document.getElementById('hidden-capture-canvas'),
  faceGuide: document.getElementById('face-guide'),
  guideText: document.getElementById('guide-text'),
  cameraFlash: document.getElementById('camera-flash'),
  
  // Overlays y banners
  analyzingOverlay: document.getElementById('analyzing-overlay'),
  cameraOfflineOverlay: document.getElementById('camera-offline-overlay'),
  cameraOfflineTitle: document.getElementById('camera-offline-title'),
  cameraOfflineDesc: document.getElementById('camera-offline-desc'),
  btnRetryCamera: document.getElementById('btn-retry-camera'),
  warmupBanner: document.getElementById('warmup-banner'),
  errorBanner: document.getElementById('error-banner'),
  errorBannerText: document.getElementById('error-banner-text'),
  btnCloseError: document.getElementById('btn-close-error'),
  
  // Badges de estado
  backendStatusBadge: document.getElementById('backend-status-badge'),
  backendStatusText: document.getElementById('backend-status-text'),
  detectionStateBadge: document.getElementById('detection-state-badge'),
  detectionStateText: document.getElementById('detection-state-text'),
  stabilityPercent: document.getElementById('stability-percent'),
  stabilityFill: document.getElementById('stability-fill'),
  
  // Acciones
  btnManualCapture: document.getElementById('btn-manual-capture'),
  btnReanalyze: document.getElementById('btn-reanalyze'),
  
  // Características de IMAGEN
  imageFeaturesBox: document.getElementById('image-features-box'),
  detectedObjectsTags: document.getElementById('detected-objects-tags'),
  metricBrightness: document.getElementById('metric-brightness'),
  metricContrast: document.getElementById('metric-contrast'),
  metricSharpness: document.getElementById('metric-sharpness'),

  // Panel de resultados (PREDICCIÓN)
  resultsEmpty: document.getElementById('results-empty'),
  resultsContent: document.getElementById('results-content'),
  predictionCard: document.getElementById('prediction-card'),
  predictionRiskBadge: document.getElementById('prediction-risk-badge'),
  predictionProbText: document.getElementById('prediction-prob-text'),
  predictionMeterFill: document.getElementById('prediction-meter-fill'),
  predictionDictamenText: document.getElementById('prediction-dictamen-text'),
  predictionFactorsList: document.getElementById('prediction-factors-list'),
  predictionRecommendationText: document.getElementById('prediction-recommendation-text'),

  // Elementos fotográficos y emocionales
  snapshotImg: document.getElementById('snapshot-img'),
  resultEmotionTitle: document.getElementById('result-emotion-title'),
  resultConfidenceVal: document.getElementById('result-confidence-val'),
  academicStatusPill: document.getElementById('academic-status-pill'),
  academicStatusText: document.getElementById('academic-status-text'),
  probBarsList: document.getElementById('prob-bars-list'),

  // Panel por Lote (CSV/JSON de IMAGEN)
  batchResultsPanel: document.getElementById('batch-results-panel'),
  batchCounter: document.getElementById('batch-counter'),
  kpiTotal: document.getElementById('kpi-total'),
  kpiCriticos: document.getElementById('kpi-criticos'),
  kpiModerados: document.getElementById('kpi-moderados'),
  kpiNormales: document.getElementById('kpi-normales'),
  batchTbody: document.getElementById('batch-tbody'),
  
  // Historial
  historyGrid: document.getElementById('history-grid'),
  historyEmpty: document.getElementById('history-empty'),
  historyCounter: document.getElementById('history-counter'),
};

let faceDetectionInstance = null;
let cameraInstance = null;

// ============================================================================
// INICIALIZACIÓN
// ============================================================================
document.addEventListener('DOMContentLoaded', () => {
  registrarEventos();
  verificarSaludBackend();
  setInterval(verificarSaludBackend, CONFIG.POLL_HEALTH_INTERVAL_MS);
  iniciarCamara();
});

function registrarEventos() {
  DOM.btnRetryCamera.addEventListener('click', () => {
    DOM.cameraOfflineOverlay.classList.add('hidden');
    iniciarCamara();
  });

  DOM.btnCloseError.addEventListener('click', () => {
    DOM.errorBanner.classList.add('hidden');
  });

  DOM.btnManualCapture.addEventListener('click', () => {
    if (state.isAnalyzing) return;
    ejecutarCaptura("manual");
  });

  DOM.btnReanalyze.addEventListener('click', () => {
    reiniciarAnalisis(true);
  });

  // Selector de pestañas de modo
  DOM.tabCamera.addEventListener('click', () => cambiarModo('camera'));
  DOM.tabUpload.addEventListener('click', () => {
    cambiarModo('upload');
    DOM.fileUploadImage.click();
  });
  DOM.tabBatch.addEventListener('click', () => {
    cambiarModo('batch');
    DOM.fileUploadBatch.click();
  });

  // Entrada de archivo de imagen
  DOM.fileUploadImage.addEventListener('change', (e) => {
    const file = e.target.files[0];
    if (file) procesarArchivoImagenManual(file);
  });

  // Entrada de archivo de lote (CSV/JSON)
  DOM.fileUploadBatch.addEventListener('change', (e) => {
    const file = e.target.files[0];
    if (file) procesarArchivoLote(file);
  });
}

function cambiarModo(nuevoModo) {
  state.currentMode = nuevoModo;
  DOM.tabCamera.classList.toggle('active', nuevoModo === 'camera');
  DOM.tabUpload.classList.toggle('active', nuevoModo === 'upload');
  DOM.tabBatch.classList.toggle('active', nuevoModo === 'batch');

  if (nuevoModo === 'camera') {
    DOM.video.classList.remove('hidden');
    DOM.overlayCanvas.classList.remove('hidden');
    DOM.uploadedPreviewImg.classList.add('hidden');
    DOM.faceGuide.classList.remove('hidden');
    DOM.stabilityContainer.classList.remove('hidden');
    DOM.batchResultsPanel.classList.add('hidden');
  } else if (nuevoModo === 'upload') {
    DOM.video.classList.add('hidden');
    DOM.overlayCanvas.classList.add('hidden');
    DOM.uploadedPreviewImg.classList.remove('hidden');
    DOM.faceGuide.classList.add('hidden');
    DOM.stabilityContainer.classList.add('hidden');
    DOM.batchResultsPanel.classList.add('hidden');
  } else if (nuevoModo === 'batch') {
    DOM.batchResultsPanel.classList.remove('hidden');
  }
}

// ============================================================================
// VERIFICACIÓN DE SALUD DEL BACKEND (/salud)
// ============================================================================
async function verificarSaludBackend() {
  let url = CONFIG.BACKEND_URL;
  let data = null;

  try {
    const controller = new AbortController();
    const timeoutId = setTimeout(() => controller.abort(), 3500);
    const res = await fetch(`${url}/salud`, { method: 'GET', signal: controller.signal });
    clearTimeout(timeoutId);
    if (res.ok) data = await res.json();
  } catch (_) {
    const isLocalEnv = window.location.hostname === 'localhost' || window.location.hostname === '127.0.0.1' || !window.location.hostname;
    if (isLocalEnv && url !== CONFIG.RENDER_BACKEND_URL) {
      try {
        const controller = new AbortController();
        const timeoutId = setTimeout(() => controller.abort(), 4000);
        const resFallback = await fetch(`${CONFIG.RENDER_BACKEND_URL}/salud`, { method: 'GET', signal: controller.signal });
        clearTimeout(timeoutId);
        if (resFallback.ok) {
          data = await resFallback.json();
          CONFIG.BACKEND_URL = CONFIG.RENDER_BACKEND_URL;
          url = CONFIG.RENDER_BACKEND_URL;
        }
      } catch (_) {}
    }
  }

  if (data) {
    state.backendOnline = true;
    state.modelReady = data.modelo_listo === true;

    const esRender = url.includes('onrender.com');
    const etiquetaOrigen = esRender ? ' (Nube Render)' : ' (Local)';

    if (state.modelReady) {
      DOM.backendStatusBadge.className = 'connection-indicator status-ready';
      DOM.backendStatusText.textContent = `Backend Conectado${etiquetaOrigen}`;
      DOM.warmupBanner.classList.add('hidden');
    } else {
      DOM.backendStatusBadge.className = 'connection-indicator status-warming';
      DOM.backendStatusText.textContent = `Modelos Preparándose...${etiquetaOrigen}`;
      DOM.warmupBanner.classList.remove('hidden');
    }
  } else {
    state.backendOnline = false;
    state.modelReady = false;
    DOM.backendStatusBadge.className = 'connection-indicator status-offline';
    DOM.backendStatusText.textContent = 'Backend Desconectado';
    DOM.warmupBanner.classList.add('hidden');
  }
}

// ============================================================================
// INICIALIZACIÓN DE CÁMARA (getUserMedia)
// ============================================================================
async function iniciarCamara() {
  actualizarBadgeDeteccion('state-waiting', 'Iniciando cámara...');

  if (!navigator.mediaDevices || !navigator.mediaDevices.getUserMedia) {
    mostrarErrorCamara('Navegador no compatible', 'Tu navegador no soporta el acceso a la cámara mediante MediaDevices.');
    return;
  }

  try {
    const stream = await navigator.mediaDevices.getUserMedia({
      video: {
        width: { ideal: 1280, min: 640 },
        height: { ideal: 720, min: 480 },
        facingMode: 'user',
      },
      audio: false,
    });

    DOM.video.srcObject = stream;
    DOM.video.onloadedmetadata = () => {
      DOM.video.play();
      state.cameraActive = true;
      ajustarDimensionesCanvas();
      iniciarMediaPipe();
    };

  } catch (error) {
    console.error('Error al solicitar cámara:', error);
    mostrarErrorCamara('Cámara en espera', 'Puedes usar la cámara o subir una imagen de examen con el botón de la barra superior.');
  }
}

function mostrarErrorCamara(titulo, descripcion) {
  state.cameraActive = false;
  DOM.cameraOfflineTitle.textContent = titulo;
  DOM.cameraOfflineDesc.textContent = descripcion;
  DOM.cameraOfflineOverlay.classList.remove('hidden');
  actualizarBadgeDeteccion('state-waiting', 'Cámara inactiva');
}

function ajustarDimensionesCanvas() {
  const w = DOM.video.videoWidth || 640;
  const h = DOM.video.videoHeight || 480;
  DOM.overlayCanvas.width = w;
  DOM.overlayCanvas.height = h;
  DOM.hiddenCanvas.width = w;
  DOM.hiddenCanvas.height = h;
}

// ============================================================================
// DETECCIÓN FACIAL EN EL NAVEGADOR (MediaPipe Face Detection)
// ============================================================================
function iniciarMediaPipe() {
  if (typeof FaceDetection === 'undefined') {
    console.warn('MediaPipe FaceDetection no cargó desde CDN.');
    iniciarBucleFallback();
    return;
  }

  try {
    faceDetectionInstance = new FaceDetection({
      locateFile: (file) => `https://cdn.jsdelivr.net/npm/@mediapipe/face_detection/${file}`
    });

    faceDetectionInstance.setOptions({
      model: 'short',
      minDetectionConfidence: 0.65,
    });

    faceDetectionInstance.onResults(onResultadosMediaPipe);

    cameraInstance = new Camera(DOM.video, {
      onFrame: async () => {
        if (state.cameraActive && faceDetectionInstance && state.currentMode === 'camera') {
          await faceDetectionInstance.send({ image: DOM.video });
        }
      },
      width: 1280,
      height: 720,
    });

    cameraInstance.start();
  } catch (err) {
    console.error('Error al inicializar MediaPipe:', err);
    iniciarBucleFallback();
  }
}

function iniciarBucleFallback() {
  actualizarBadgeDeteccion('state-waiting', 'Modo captura listo');
  DOM.guideText.textContent = 'Encuadra el examen y pulsa capturar';
}

function onResultadosMediaPipe(results) {
  if (state.currentMode !== 'camera') return;
  const ctx = DOM.overlayCanvas.getContext('2d');
  ctx.clearRect(0, 0, DOM.overlayCanvas.width, DOM.overlayCanvas.height);

  if (state.isAnalyzing) return;

  const detections = results.detections || [];
  if (detections.length === 0) {
    manejarSinRostro();
    return;
  }

  const face = detections[0];
  const box = face.boundingBox;
  const videoW = DOM.overlayCanvas.width;
  const videoH = DOM.overlayCanvas.height;

  const boxW = box.width * videoW;
  const boxH = box.height * videoH;
  const boxX = box.xCenter * videoW - boxW / 2;
  const boxY = box.yCenter * videoH - boxH / 2;

  const faceRatio = (boxW * boxH) / (videoW * videoH);
  const faceAdecuado = faceRatio >= CONFIG.MIN_FACE_SIZE_RATIO;

  const centerX = boxX + boxW / 2;
  const centerY = boxY + boxH / 2;
  const centrado = (
    centerX > videoW * 0.25 && centerX < videoW * 0.75 &&
    centerY > videoH * 0.20 && centerY < videoH * 0.80
  );

  if (state.isInCooldown) {
    dibujarBordeRostro(ctx, boxX, boxY, boxW, boxH, '#94a3b8');
    return;
  }

  if (faceAdecuado && centrado) {
    state.faceDetected = true;
    dibujarBordeRostro(ctx, boxX, boxY, boxW, boxH, '#166534');
    gestionarEstabilidad(true);
  } else {
    state.faceDetected = false;
    dibujarBordeRostro(ctx, boxX, boxY, boxW, boxH, '#d97706');
    gestionarEstabilidad(false);
  }
}

function dibujarBordeRostro(ctx, x, y, w, h, color) {
  ctx.save();
  ctx.strokeStyle = color;
  ctx.lineWidth = 3;
  ctx.setLineDash([8, 6]);
  ctx.strokeRect(x, y, w, h);
  ctx.restore();
}

function manejarSinRostro() {
  state.faceDetected = false;
  gestionarEstabilidad(false);
  if (state.isInCooldown) state.faceDisappearedDuringCooldown = true;
  if (!state.isInCooldown && !state.isAnalyzing) {
    actualizarBadgeDeteccion('state-waiting', 'Esperando encuadre');
    DOM.guideText.textContent = 'Centra al estudiante en el cuadro';
  }
}

function gestionarEstabilidad(cumpleCondiciones) {
  if (state.isInCooldown || state.isAnalyzing) return;

  const now = Date.now();
  if (cumpleCondiciones) {
    if (!state.stabilityStartTime) state.stabilityStartTime = now;
    const tiempoEstable = now - state.stabilityStartTime;
    const progreso = Math.min(1.0, tiempoEstable / CONFIG.STABILITY_REQUIRED_MS);
    const porcentaje = Math.round(progreso * 100);

    DOM.stabilityPercent.textContent = `${porcentaje}%`;
    DOM.stabilityFill.style.width = `${porcentaje}%`;
    actualizarBadgeDeteccion('state-tracking', `Estabilidad: ${porcentaje}%`);

    if (progreso >= 1.0) {
      ejecutarCaptura("automatica");
    }
  } else {
    reiniciarEstabilidad();
  }
}

function reiniciarEstabilidad() {
  state.stabilityStartTime = null;
  DOM.stabilityPercent.textContent = '0%';
  DOM.stabilityFill.style.width = '0%';
}

// ============================================================================
// CAPTURA Y TRANSMISIÓN DE IMAGEN
// ============================================================================
async function ejecutarCaptura(tipoCaptura = "automatica") {
  if (state.isAnalyzing) return;

  state.isAnalyzing = true;
  reiniciarEstabilidad();

  DOM.cameraFlash.classList.remove('flash-active');
  void DOM.cameraFlash.offsetWidth;
  DOM.cameraFlash.classList.add('flash-active');

  const c = DOM.hiddenCanvas;
  const ctx = c.getContext('2d');
  c.width = DOM.video.videoWidth || 1280;
  c.height = DOM.video.videoHeight || 720;
  ctx.drawImage(DOM.video, 0, 0, c.width, c.height);

  state.lastSnapshotDataUrl = c.toDataURL('image/jpeg', 0.95);
  DOM.snapshotImg.src = state.lastSnapshotDataUrl;

  DOM.analyzingOverlay.classList.remove('hidden');
  actualizarBadgeDeteccion('state-captured', 'Ejecutando detección (best.pt) y predicción...');

  c.toBlob(async (blob) => {
    if (!blob) {
      mostrarError('No se pudo generar el archivo de imagen para análisis.');
      finalizarAnalisis();
      return;
    }
    await enviarImagenABackend(blob);
  }, 'image/jpeg', 0.95);
}

async function procesarArchivoImagenManual(file) {
  if (state.isAnalyzing) return;
  state.isAnalyzing = true;

  const reader = new FileReader();
  reader.onload = (e) => {
    state.lastSnapshotDataUrl = e.target.result;
    DOM.uploadedPreviewImg.src = e.target.result;
    DOM.snapshotImg.src = e.target.result;
  };
  reader.readAsDataURL(file);

  DOM.analyzingOverlay.classList.remove('hidden');
  actualizarBadgeDeteccion('state-captured', 'Analizando archivo cargado con YOLO...');
  await enviarImagenABackend(file);
}

async function enviarImagenABackend(blobOFile) {
  const formData = new FormData();
  formData.append('archivo', blobOFile, 'captura.jpg');

  try {
    const response = await fetch(`${CONFIG.BACKEND_URL}/analizar`, {
      method: 'POST',
      body: formData,
    });

    if (response.status === 503) {
      const errData = await response.json().catch(() => ({}));
      mostrarError(errData.mensaje || 'Los modelos de IA se están preparando, espera unos segundos...');
      DOM.warmupBanner.classList.remove('hidden');
      return;
    }

    if (!response.ok) {
      const errorData = await response.json().catch(() => ({}));
      throw new Error(errorData.detail || `Error en el servidor: HTTP ${response.status}`);
    }

    const resultado = await response.json();
    procesarRespuestaBackend(resultado);

  } catch (err) {
    console.error('Error al enviar imagen al backend:', err);
    mostrarError(
      !state.backendOnline
        ? `No se puede conectar con el servidor backend (${CONFIG.BACKEND_URL}). Asegúrate de que FastAPI esté activo.`
        : `Error en análisis: ${err.message}`
    );
  } finally {
    finalizarAnalisis();
    if (state.currentMode === 'camera') iniciarCooldown();
  }
}

function finalizarAnalisis() {
  state.isAnalyzing = false;
  DOM.analyzingOverlay.classList.add('hidden');
}

function iniciarCooldown() {
  state.isInCooldown = true;
  state.faceDisappearedDuringCooldown = false;
  let tiempoRestante = Math.round(CONFIG.COOLDOWN_MS / 1000);

  actualizarBadgeDeteccion('state-ready', `Resultado listo (cooldown: ${tiempoRestante}s)`);

  state.cooldownTimer = setInterval(() => {
    tiempoRestante--;
    if (tiempoRestante > 0) {
      actualizarBadgeDeteccion('state-ready', `Resultado listo (cooldown: ${tiempoRestante}s)`);
    } else {
      clearInterval(state.cooldownTimer);
      state.cooldownTimer = null;
      state.isInCooldown = false;
      actualizarBadgeDeteccion('state-waiting', 'Listo para siguiente análisis');
    }
  }, 1000);
}

function reiniciarAnalisis(forzar = false) {
  if (state.cooldownTimer) {
    clearInterval(state.cooldownTimer);
    state.cooldownTimer = null;
  }
  state.isInCooldown = false;
  state.isAnalyzing = false;
  reiniciarEstabilidad();
  actualizarBadgeDeteccion('state-waiting', 'Buscando encuadre...');
  if (forzar) DOM.errorBanner.classList.add('hidden');
}

// ============================================================================
// PROCESAMIENTO Y RENDERIZADO DEL RESULTADO
// ============================================================================
function procesarRespuestaBackend(data) {
  state.lastResult = data;

  DOM.resultsEmpty.classList.add('hidden');
  DOM.resultsContent.classList.remove('hidden');

  // 1. RENDERIZAR MÓDULO PREDICCIÓN (Evaluación de Fraude en Examen)
  const sup = data.supervision || {};
  const pred = sup.prediccion || {
    nivel_riesgo: 'normal',
    etiqueta_riesgo: 'Normal - Sin Infracciones',
    probabilidad_fraude: 0.0,
    dictamen: 'Sin elementos sospechosos detectados.',
    factores_clave: ['Evaluación despejada.'],
    recomendacion: 'Supervisión habitual.',
  };

  DOM.predictionCard.className = `prediction-card risk-card-${pred.nivel_riesgo}`;
  DOM.predictionRiskBadge.className = `risk-pill pill-${pred.nivel_riesgo}`;
  DOM.predictionRiskBadge.textContent = pred.etiqueta_riesgo;
  DOM.predictionProbText.textContent = `${pred.probabilidad_fraude}%`;
  DOM.predictionMeterFill.style.width = `${Math.max(5, pred.probabilidad_fraude)}%`;
  DOM.predictionDictamenText.textContent = pred.dictamen;

  DOM.predictionFactorsList.innerHTML = (pred.factores_clave || [])
    .map(f => `<li>${f}</li>`)
    .join('');
  DOM.predictionRecommendationText.textContent = pred.recomendacion;

  // 2. RENDERIZAR CARACTERÍSTICAS EXTRAÍDAS POR MÓDULO IMAGEN (best.pt)
  if (sup.exito && sup.detecciones) {
    DOM.imageFeaturesBox.classList.remove('hidden');
    DOM.detectedObjectsTags.innerHTML = sup.detecciones.map(d => {
      const esInfractor = ['telefono', 'celular', 'audifonos', 'airpods', 'cuaderno', 'libro', 'reloj'].includes(d.clase.toLowerCase());
      const claseCss = esInfractor ? 'obj-tag-infractor' : 'obj-tag-permitido';
      return `<span class="obj-tag ${claseCss}">${d.clase.toUpperCase()} (${Math.round(d.confianza * 100)}%)</span>`;
    }).join('');

    if (sup.calidad_imagen) {
      DOM.metricBrightness.textContent = `Brillo: ${sup.calidad_imagen.brillo}`;
      DOM.metricContrast.textContent = `Contraste: ${sup.calidad_imagen.contraste}`;
      DOM.metricSharpness.textContent = `Nitidez: ${sup.calidad_imagen.nitidez}`;
    }

    // Si viene la imagen anotada con los cuadros de colores de YOLO
    if (sup.imagen_anotada) {
      DOM.snapshotImg.src = sup.imagen_anotada;
    }
  }

  // 3. RENDERIZAR ESTADO EMOCIONAL COMPLEMENTARIO (DeepFace)
  const emocion = (data.emocion || 'neutral').toLowerCase();
  const emocionFormateada = emocion.charAt(0).toUpperCase() + emocion.slice(1);
  DOM.resultEmotionTitle.textContent = emocionFormateada;
  DOM.resultConfidenceVal.textContent = `${Math.round((data.confianza || 0) * 100)}%`;

  const estadoAcad = (data.estado_academico || 'estable').toLowerCase();
  const metaAcad = ESTADO_ACADEMICO_METADATA[estadoAcad] || ESTADO_ACADEMICO_METADATA['estable'];
  DOM.academicStatusPill.className = `academic-badge ${metaAcad.class}`;
  DOM.academicStatusText.textContent = estadoAcad.charAt(0).toUpperCase() + estadoAcad.slice(1);

  renderizarBarrasProbabilidades(data.probabilidades || {});

  // 4. REGISTRAR EN HISTORIAL
  agregarAlHistorial({
    timestamp: new Date(),
    snapshotDataUrl: sup.imagen_anotada || state.lastSnapshotDataUrl,
    nivel_riesgo: pred.nivel_riesgo,
    etiqueta_riesgo: pred.etiqueta_riesgo,
    probabilidad_fraude: pred.probabilidad_fraude,
    objetos: (sup.detecciones || []).map(d => d.clase).join(', ') || 'Ninguno',
    estado_academico: estadoAcad,
  });

  actualizarBadgeDeteccion('state-ready', 'Evaluación completada');
}

function renderizarBarrasProbabilidades(probabilidades) {
  DOM.probBarsList.innerHTML = '';
  const listaEmociones = [
    { key: 'feliz', label: 'Feliz' },
    { key: 'triste', label: 'Triste' },
    { key: 'neutral', label: 'Neutral' },
    { key: 'enojado', label: 'Enojado' },
    { key: 'sorprendido', label: 'Sorprendido' },
    { key: 'miedo', label: 'Miedo' },
    { key: 'disgusto', label: 'Disgusto' },
  ];

  listaEmociones.forEach(em => {
    const rawVal = probabilidades[em.key] !== undefined ? probabilidades[em.key] : 0;
    const pct = Math.round(rawVal * 100);

    const item = document.createElement('div');
    item.className = 'prob-item';
    item.innerHTML = `
      <div class="prob-item-header">
        <span class="prob-emotion-name">${em.label}</span>
        <span class="prob-percent-val">${pct}%</span>
      </div>
      <div class="prob-track">
        <div class="prob-bar-fill" style="width: 0%;"></div>
      </div>
    `;
    DOM.probBarsList.appendChild(item);
    setTimeout(() => {
      const fill = item.querySelector('.prob-bar-fill');
      if (fill) fill.style.width = `${pct}%`;
    }, 40);
  });
}

// ============================================================================
// PROCESAMIENTO EN LOTE (CSV / JSON DE IMAGEN)
// ============================================================================
async function procesarArchivoLote(file) {
  DOM.batchResultsPanel.classList.remove('hidden');
  DOM.batchCounter.textContent = 'Procesando archivo...';

  const formData = new FormData();
  formData.append('archivo', file);

  try {
    const res = await fetch(`${CONFIG.BACKEND_URL}/supervision/predecir_lote`, {
      method: 'POST',
      body: formData,
    });

    if (!res.ok) {
      const err = await res.json().catch(() => ({}));
      throw new Error(err.detail || `Error HTTP ${res.status}`);
    }

    const data = await res.json();
    renderizarResultadosLote(data);

  } catch (err) {
    console.error('Error al procesar lote:', err);
    mostrarError(`Error al procesar reporte de IMAGEN: ${err.message}`);
  }
}

function renderizarResultadosLote(data) {
  DOM.batchCounter.textContent = `${data.total_alertas} alertas procesadas`;
  DOM.kpiTotal.textContent = data.total_alertas;
  DOM.kpiCriticos.textContent = data.total_criticos;
  DOM.kpiModerados.textContent = data.total_moderados;
  DOM.kpiNormales.textContent = data.total_normales;

  DOM.batchTbody.innerHTML = '';
  (data.resultados || []).forEach(r => {
    const tr = document.createElement('tr');
    tr.innerHTML = `
      <td>${r.id}</td>
      <td><strong>${r.objeto.toUpperCase()}</strong></td>
      <td>${Math.round(r.confianza * 100)}%</td>
      <td><span class="risk-pill pill-${r.nivel_riesgo}">${r.nivel_riesgo.toUpperCase()}</span></td>
      <td><strong>${r.probabilidad_fraude}%</strong></td>
      <td>${r.dictamen}</td>
    `;
    DOM.batchTbody.appendChild(tr);
  });
}

// ============================================================================
// HISTORIAL DE OBSERVACIONES
// ============================================================================
function agregarAlHistorial(registro) {
  state.history.unshift(registro);
  if (state.history.length > CONFIG.MAX_HISTORY_ITEMS) state.history.pop();
  renderizarHistorial();
}

function renderizarHistorial() {
  if (state.history.length === 0) {
    DOM.historyEmpty.classList.remove('hidden');
    DOM.historyCounter.textContent = '0 registros';
    DOM.historyGrid.innerHTML = '';
    return;
  }

  DOM.historyEmpty.classList.add('hidden');
  DOM.historyCounter.textContent = `${state.history.length} registros`;
  DOM.historyGrid.innerHTML = '';

  state.history.forEach(item => {
    const tr = document.createElement('tr');
    const hora = item.timestamp.toLocaleTimeString([], { hour: '2-digit', minute: '2-digit', second: '2-digit' });
    tr.innerHTML = `
      <td>${hora}</td>
      <td><img src="${item.snapshotDataUrl}" alt="Miniatura" class="history-thumb"></td>
      <td>
        <span class="risk-pill pill-${item.nivel_riesgo}">${item.etiqueta_riesgo}</span>
        <br><small>Probabilidad: ${item.probabilidad_fraude}%</small>
      </td>
      <td>${item.objetos}</td>
      <td><span class="academic-badge academic-${item.estado_academico}">${item.estado_academico}</span></td>
    `;
    DOM.historyGrid.appendChild(tr);
  });
}

// ============================================================================
// UTILIDADES DE INTERFAZ
// ============================================================================
function actualizarBadgeDeteccion(claseEstado, texto) {
  DOM.detectionStateBadge.className = `state-pill ${claseEstado}`;
  DOM.detectionStateText.textContent = texto;
}

function mostrarError(mensaje) {
  DOM.errorBannerText.textContent = mensaje;
  DOM.errorBanner.classList.remove('hidden');
}
