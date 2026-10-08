/**
 * AulaPredict - Módulo Imagen
 * Lógica Frontend (app.js)
 * 
 * Funcionalidades:
 * - Acceso a cámara web con getUserMedia (solo vista previa para encuadre).
 * - Detección facial en tiempo real en el navegador usando MediaPipe Face Detection.
 * - Captura automática de UNA sola foto tras 1 segundo de estabilidad frontal y tamaño adecuado.
 * - Cooldown de 5 segundos y prevención de capturas repetidas.
 * - Envío en memoria (FormData) al backend FastAPI (/analizar).
 * - Renderizado de emoción dominante, confianza, barras de probabilidad y estado académico.
 * - Historial en memoria (últimas 5 capturas).
 * - Manejo robusto de errores y verificación de salud del servidor (/salud).
 */

// ============================================================================
// CONFIGURACIÓN Y CONSTANTES
// ============================================================================
const CONFIG = {
  // En local apunta al puerto 8000; en Render se adapta al mismo dominio o al URL configurado
  BACKEND_URL: (window.location.hostname === 'localhost' || window.location.hostname === '127.0.0.1')
    ? 'http://localhost:8000'
    : (window.AULAPREDICT_BACKEND_URL || window.location.origin),
  STABILITY_REQUIRED_MS: 1000,    // 1 segundo de estabilidad continua
  COOLDOWN_MS: 5000,              // 5 segundos de espera tras captura
  MIN_FACE_SIZE_RATIO: 0.18,       // Mínimo ~18-20% del cuadro para garantizar nitidez
  POLL_HEALTH_INTERVAL_MS: 8000,  // Intervalo de sondeo de salud del backend
  MAX_HISTORY_ITEMS: 5,           // Máximo de registros en historial en memoria
};

// Mapeo de emojis y clases para cada emoción
const EMOCION_METADATA = {
  feliz: { emoji: '😄', colorClass: 'bar-feliz' },
  triste: { emoji: '😢', colorClass: 'bar-triste' },
  neutral: { emoji: '😐', colorClass: 'bar-neutral' },
  enojado: { emoji: '😠', colorClass: 'bar-enojado' },
  sorprendido: { emoji: '😲', colorClass: 'bar-sorprendido' },
  miedo: { emoji: '😨', colorClass: 'bar-miedo' },
  disgusto: { emoji: '🤢', colorClass: 'bar-disgusto' },
};

// Mapeo de estado académico (icono y clase CSS)
const ESTADO_ACADEMICO_METADATA = {
  'posible riesgo': { icon: '⚠️', class: 'academic-riesgo' },
  'estable': { icon: '⚖️', class: 'academic-estable' },
  'positivo': { icon: '✨', class: 'academic-positivo' },
  'no determinado': { icon: '❓', class: 'academic-estable' }
};

// ============================================================================
// VARIABLES DE ESTADO DE LA APLICACIÓN
// ============================================================================
const state = {
  backendOnline: false,
  modelReady: false,
  cameraActive: false,
  faceDetected: false,
  
  // Control de estabilidad para captura automática
  stabilityStartTime: null,
  isAnalyzing: false,
  isInCooldown: false,
  cooldownTimer: null,
  faceDisappearedDuringCooldown: false,
  
  // Último resultado y fotos
  lastResult: null,
  lastSnapshotDataUrl: null,
  
  // Historial en memoria (máximo 5)
  history: [],
};

// ============================================================================
// REFERENCIAS AL DOM
// ============================================================================
const DOM = {
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
  
  // Badges y barras de progreso
  backendStatusBadge: document.getElementById('backend-status-badge'),
  backendStatusText: document.getElementById('backend-status-text'),
  detectionStateBadge: document.getElementById('detection-state-badge'),
  detectionStateText: document.getElementById('detection-state-text'),
  stabilityPercent: document.getElementById('stability-percent'),
  stabilityFill: document.getElementById('stability-fill'),
  
  // Acciones
  btnManualCapture: document.getElementById('btn-manual-capture'),
  btnReanalyze: document.getElementById('btn-reanalyze'),
  
  // Panel de resultados
  resultsEmpty: document.getElementById('results-empty'),
  resultsContent: document.getElementById('results-content'),
  snapshotImg: document.getElementById('snapshot-img'),
  resultEmoji: document.getElementById('result-emoji'),
  resultEmotionTitle: document.getElementById('result-emotion-title'),
  resultConfidenceVal: document.getElementById('result-confidence-val'),
  academicStatusPill: document.getElementById('academic-status-pill'),
  academicStatusIcon: document.getElementById('academic-status-icon'),
  academicStatusText: document.getElementById('academic-status-text'),
  probBarsList: document.getElementById('prob-bars-list'),
  
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
}

// ============================================================================
// VERIFICACIÓN DE SALUD DEL BACKEND (/salud)
// ============================================================================
async function verificarSaludBackend() {
  try {
    const res = await fetch(`${CONFIG.BACKEND_URL}/salud`, { method: 'GET' });
    if (!res.ok) throw new Error(`HTTP ${res.status}`);
    const data = await res.json();
    
    state.backendOnline = true;
    state.modelReady = data.modelo_listo === true;

    if (state.modelReady) {
      DOM.backendStatusBadge.className = 'status-pill status-ready';
      DOM.backendStatusText.textContent = 'Backend Conectado';
      DOM.warmupBanner.classList.add('hidden');
    } else {
      DOM.backendStatusBadge.className = 'status-pill status-warming';
      DOM.backendStatusText.textContent = 'Modelo Preparándose...';
      DOM.warmupBanner.classList.remove('hidden');
    }
  } catch (err) {
    state.backendOnline = false;
    state.modelReady = false;
    DOM.backendStatusBadge.className = 'status-pill status-offline';
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
    mostrarErrorCamara(
      'Navegador no compatible',
      'Tu navegador no soporta el acceso a la cámara mediante MediaDevices.'
    );
    return;
  }

  try {
    const stream = await navigator.mediaDevices.getUserMedia({
      video: {
        width: { ideal: 1920, min: 1280 },
        height: { ideal: 1080, min: 720 },
        facingMode: 'user',
        frameRate: { ideal: 30 }
      },
      audio: false, // SOLO IMAGEN: nada de audio
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
    if (error.name === 'NotAllowedError' || error.name === 'PermissionDeniedError') {
      mostrarErrorCamara(
        'Permiso de cámara denegado',
        'Por favor, concede permiso de cámara en la barra de tu navegador para que AulaPredict pueda detectar tu rostro.'
      );
    } else if (error.name === 'NotFoundError' || error.name === 'DevicesNotFoundError') {
      mostrarErrorCamara(
        'Cámara no encontrada',
        'No se detectó ninguna cámara conectada en tu equipo.'
      );
    } else {
      mostrarErrorCamara(
        'Error al iniciar cámara',
        `Detalles: ${error.message || error.name}`
      );
    }
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
  const ancho = DOM.video.videoWidth || 640;
  const alto = DOM.video.videoHeight || 480;
  DOM.overlayCanvas.width = ancho;
  DOM.overlayCanvas.height = alto;
}

window.addEventListener('resize', () => {
  if (state.cameraActive) {
    ajustarDimensionesCanvas();
  }
});

// ============================================================================
// DETECCIÓN FACIAL CON MEDIAPIPE FACE DETECTION
// ============================================================================
function iniciarMediaPipe() {
  if (typeof FaceDetection === 'undefined') {
    console.warn('MediaPipe CDN no cargó correctamente, esperando o reintentando...');
    setTimeout(iniciarMediaPipe, 1000);
    return;
  }

  try {
    faceDetectionInstance = new FaceDetection({
      locateFile: (file) => `https://cdn.jsdelivr.net/npm/@mediapipe/face_detection/${file}`
    });

    faceDetectionInstance.setOptions({
      model: 'short', // Modelo rápido optimizado para distancias cortas (selfies)
      minDetectionConfidence: 0.6,
    });

    faceDetectionInstance.onResults(procesarResultadosMediaPipe);

    // Usar requestAnimationFrame para enviar fotogramas de forma suave
    let procesandoFrame = false;
    async function bucleDeteccion() {
      if (state.cameraActive && DOM.video.readyState >= 2) {
        if (!procesandoFrame && !state.isAnalyzing) {
          procesandoFrame = true;
          try {
            await faceDetectionInstance.send({ image: DOM.video });
          } catch (e) {
            console.error('Error enviando frame a MediaPipe:', e);
          }
          procesandoFrame = false;
        }
      }
      requestAnimationFrame(bucleDeteccion);
    }

    requestAnimationFrame(bucleDeteccion);
    actualizarBadgeDeteccion('state-waiting', 'Esperando rostro...');

  } catch (err) {
    console.error('Fallo al inicializar MediaPipe:', err);
    actualizarBadgeDeteccion('state-waiting', 'Detección en modo básico');
  }
}

// ============================================================================
// PROCESAMIENTO DE DETECCIÓN Y ESTABILIDAD
// ============================================================================
function procesarResultadosMediaPipe(results) {
  const ctx = DOM.overlayCanvas.getContext('2d');
  ctx.clearRect(0, 0, DOM.overlayCanvas.width, DOM.overlayCanvas.height);

  const hayRostros = results.detections && results.detections.length > 0;

  // Si estamos en cooldown, verificar si el rostro desapareció
  if (state.isInCooldown) {
    if (!hayRostros) {
      state.faceDisappearedDuringCooldown = true;
    }
    dibujarEncuadre(ctx, results.detections, false);
    return;
  }

  // Si estamos en proceso de envío/análisis, no acumular estabilidad
  if (state.isAnalyzing) {
    return;
  }

  if (!hayRostros) {
    // Rostro perdido: reiniciar contador de estabilidad
    reiniciarEstabilidad();
    DOM.faceGuide.className = 'face-guide-silhouette';
    DOM.guideText.textContent = 'Centra tu rostro aquí';
    actualizarBadgeDeteccion('state-waiting', 'Esperando rostro...');
    return;
  }

  // Tomamos la detección principal
  const deteccion = results.detections[0];
  const box = deteccion.boundingBox;

  // Validar tamaño relativo del rostro (al menos ~18-20% de ancho y alto)
  const anchoValido = box.width >= CONFIG.MIN_FACE_SIZE_RATIO;
  const altoValido = box.height >= CONFIG.MIN_FACE_SIZE_RATIO;
  const tamanoSuficiente = anchoValido && altoValido;

  // Validar si está de frente usando landmarks (ojos y nariz)
  const esDeFrente = verificarRostroDeFrente(deteccion.landmarks);

  dibujarEncuadre(ctx, results.detections, tamanoSuficiente && esDeFrente);

  if (!tamanoSuficiente) {
    reiniciarEstabilidad();
    DOM.faceGuide.className = 'face-guide-silhouette';
    DOM.guideText.textContent = 'Acércate un poco más a la cámara';
    actualizarBadgeDeteccion('state-waiting', 'Rostro muy lejos...');
    return;
  }

  if (!esDeFrente) {
    reiniciarEstabilidad();
    DOM.faceGuide.className = 'face-guide-silhouette active';
    DOM.guideText.textContent = 'Mira de frente a la cámara';
    actualizarBadgeDeteccion('state-tracking', 'Mira de frente...');
    return;
  }

  // Rostro de frente y con tamaño adecuado -> Acumular estabilidad
  DOM.faceGuide.className = 'face-guide-silhouette active';
  DOM.guideText.textContent = '¡Perfecto! No te muevas...';

  const ahora = performance.now();
  if (state.stabilityStartTime === null) {
    state.stabilityStartTime = ahora;
  }

  const transcurrido = ahora - state.stabilityStartTime;
  const progreso = Math.min(100, (transcurrido / CONFIG.STABILITY_REQUIRED_MS) * 100);

  actualizarBarraEstabilidad(progreso);

  const segundosRestantes = Math.max(0.1, ((CONFIG.STABILITY_REQUIRED_MS - transcurrido) / 1000)).toFixed(1);
  actualizarBadgeDeteccion('state-tracking', `Rostro detectado (${segundosRestantes}s)...`);

  // Cuando se alcanza el segundo de estabilidad requerido -> CAPTURAR UNA FOTO
  if (transcurrido >= CONFIG.STABILITY_REQUIRED_MS) {
    DOM.faceGuide.className = 'face-guide-silhouette locked';
    ejecutarCaptura("automatica");
  }
}

/**
 * Verifica si el rostro está mirando de frente comparando
 * la posición de los ojos con la punta de la nariz.
 */
function verificarRostroDeFrente(landmarks) {
  if (!landmarks || landmarks.length < 3) return true; // Si no hay landmarks, permitimos por defecto
  
  // Landmarks de MediaPipe Face Detection:
  // 0: Ojo derecho (desde la perspectiva de la persona)
  // 1: Ojo izquierdo
  // 2: Punta de la nariz
  const ojoDer = landmarks[0];
  const ojoIzq = landmarks[1];
  const nariz = landmarks[2];

  if (!ojoDer || !ojoIzq || !nariz) return true;

  const distanciaOjos = Math.abs(ojoDer.x - ojoIzq.x);
  if (distanciaOjos < 0.05) return false;

  // La nariz debe estar aproximadamente centrada entre ambos ojos
  const centroX = (ojoDer.x + ojoIzq.x) / 2;
  const desviacionNariz = Math.abs(nariz.x - centroX) / distanciaOjos;

  // Si la nariz se desvía más del 38% respecto al centro entre ojos, está de perfil
  return desviacionNariz < 0.38;
}

function reiniciarEstabilidad() {
  state.stabilityStartTime = null;
  actualizarBarraEstabilidad(0);
}

function actualizarBarraEstabilidad(porcentaje) {
  const p = Math.round(porcentaje);
  DOM.stabilityFill.style.width = `${p}%`;
  DOM.stabilityPercent.textContent = `${p}%`;
}

function actualizarBadgeDeteccion(claseEstado, texto) {
  DOM.detectionStateBadge.className = `state-badge ${claseEstado}`;
  DOM.detectionStateText.textContent = texto;
}

/**
 * Dibuja un marco estético de seguimiento sobre el canvas de overlay.
 */
function dibujarEncuadre(ctx, detections, esOptimo) {
  if (!detections || detections.length === 0) return;

  const anchoC = DOM.overlayCanvas.width;
  const altoC = DOM.overlayCanvas.height;

  detections.forEach(det => {
    const b = det.boundingBox;
    const x = b.xCenter * anchoC - (b.width * anchoC) / 2;
    const y = b.yCenter * altoC - (b.height * altoC) / 2;
    const w = b.width * anchoC;
    const h = b.height * altoC;

    ctx.save();
    ctx.lineWidth = esOptimo ? 3 : 1.5;
    ctx.strokeStyle = esOptimo ? 'rgba(16, 185, 129, 0.85)' : 'rgba(56, 189, 248, 0.4)';
    
    // Marco suave redondeado
    const r = 12;
    ctx.beginPath();
    ctx.moveTo(x + r, y);
    ctx.lineTo(x + w - r, y);
    ctx.quadraticCurveTo(x + w, y, x + w, y + r);
    ctx.lineTo(x + w, y + h - r);
    ctx.quadraticCurveTo(x + w, y + h, x + w - r, y + h);
    ctx.lineTo(x + r, y + h);
    ctx.quadraticCurveTo(x, y + h, x, y + h - r);
    ctx.lineTo(x, y + r);
    ctx.quadraticCurveTo(x, y, x + r, y);
    ctx.closePath();
    ctx.stroke();

    ctx.restore();
  });
}

// ============================================================================
// CAPTURA DE UNA SOLA FOTO Y ENVÍO AL BACKEND
// ============================================================================
async function ejecutarCaptura(tipoCaptura = "automatica") {
  if (state.isAnalyzing) return;

  state.isAnalyzing = true;
  reiniciarEstabilidad();

  // 1. Efecto visual de flash
  DOM.cameraFlash.classList.remove('flash-active');
  void DOM.cameraFlash.offsetWidth; // Forzar reflujo CSS
  DOM.cameraFlash.classList.add('flash-active');

  // 2. Extraer el fotograma al canvas oculto (SOLO IMAGEN en memoria)
  const c = DOM.hiddenCanvas;
  c.width = DOM.video.videoWidth || 640;
  c.height = DOM.video.videoHeight || 480;
  const ctx = c.getContext('2d');
  
  // Dibujamos la imagen exactamente como se ve (vista espejo invertida para que coincida)
  ctx.save();
  ctx.translate(c.width, 0);
  ctx.scale(-1, 1);
  ctx.drawImage(DOM.video, 0, 0, c.width, c.height);
  ctx.restore();

  // 3. Obtener DataURL para mostrar la foto inmediatamente en el resultado
  const dataUrl = c.toDataURL('image/jpeg', 0.95);
  state.lastSnapshotDataUrl = dataUrl;
  DOM.snapshotImg.src = dataUrl;

  // 4. Mostrar overlay de análisis
  DOM.analyzingOverlay.classList.remove('hidden');
  actualizarBadgeDeteccion('state-captured', 'Analizando emoción con CLAHE y TTA...');

  // 5. Convertir a Blob y enviar en FormData
  c.toBlob(async (blob) => {
    if (!blob) {
      mostrarError('No se pudo generar el archivo de imagen para análisis.');
      finalizarAnalisis();
      return;
    }

    const formData = new FormData();
    formData.append('archivo', blob, 'captura.jpg');

    try {
      const response = await fetch(`${CONFIG.BACKEND_URL}/analizar`, {
        method: 'POST',
        body: formData,
      });

      if (response.status === 503) {
        // Modelo aún en calentamiento
        const errData = await response.json().catch(() => ({}));
        mostrarError(errData.mensaje || 'El modelo se está preparando, espera unos segundos...');
        DOM.warmupBanner.classList.remove('hidden');
        finalizarAnalisis();
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
          ? 'No se puede conectar con el servidor backend (puerto 8000). Asegúrate de que FastAPI esté corriendo.'
          : `Error en análisis: ${err.message}`
      );
    } finally {
      finalizarAnalisis();
      iniciarCooldown();
    }
  }, 'image/jpeg', 0.95);
}

function finalizarAnalisis() {
  state.isAnalyzing = false;
  DOM.analyzingOverlay.classList.add('hidden');
}

// ============================================================================
// COOLDOWN Y CONTROL DE CAPTURAS REPETIDAS
// ============================================================================
function iniciarCooldown() {
  state.isInCooldown = true;
  state.faceDisappearedDuringCooldown = false;
  
  let tiempoRestante = CONFIG.COOLDOWN_MS / 1000;
  actualizarBadgeDeteccion('state-ready', `Resultado listo (cooldown: ${tiempoRestante}s)`);

  if (state.cooldownTimer) clearInterval(state.cooldownTimer);

  state.cooldownTimer = setInterval(() => {
    tiempoRestante -= 1;
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
  actualizarBadgeDeteccion('state-waiting', 'Buscando rostro...');
  
  if (forzar) {
    DOM.errorBanner.classList.add('hidden');
  }
}

// ============================================================================
// PROCESAMIENTO Y RENDERIZADO DEL RESULTADO
// ============================================================================
function procesarRespuestaBackend(data) {
  state.lastResult = data;

  if (!data.rostro_detectado) {
    mostrarError(data.mensaje || 'No se detectó ningún rostro en la imagen.');
    actualizarBadgeDeteccion('state-waiting', 'Sin rostro detectado');
    return;
  }

  // Ocultar estado vacío y mostrar panel de resultados
  DOM.resultsEmpty.classList.add('hidden');
  DOM.resultsContent.classList.remove('hidden');

  const emocion = (data.emocion || 'neutral').toLowerCase();
  const meta = EMOCION_METADATA[emocion] || { emoji: '😐', colorClass: 'bar-neutral' };

  // 1. Emoción dominante
  const emocionFormateada = emocion.charAt(0).toUpperCase() + emocion.slice(1);
  DOM.resultEmotionTitle.textContent = emocionFormateada;
  
  // 2. Porcentaje de confianza
  const confianzaPct = Math.round((data.confianza || 0) * 100);
  DOM.resultConfidenceVal.textContent = `${confianzaPct}%`;

  // 3. Estado académico
  const estadoAcad = (data.estado_academico || 'estable').toLowerCase();
  const metaAcad = ESTADO_ACADEMICO_METADATA[estadoAcad] || ESTADO_ACADEMICO_METADATA['estable'];
  const estadoFormateado = estadoAcad.charAt(0).toUpperCase() + estadoAcad.slice(1);
  
  DOM.academicStatusPill.className = `academic-badge ${metaAcad.class}`;
  DOM.academicStatusText.textContent = estadoFormateado;

  // 4. Barras de probabilidades para las 7 emociones
  renderizarBarrasProbabilidades(data.probabilidades || {});

  // 5. Registrar en historial en memoria
  agregarAlHistorial({
    timestamp: new Date(),
    snapshotDataUrl: state.lastSnapshotDataUrl,
    emocion: emocionFormateada,
    confianza: confianzaPct,
    estado_academico: estadoFormateado,
    estado_class: metaAcad.class,
  });

  actualizarBadgeDeteccion('state-ready', 'Resultado listo');
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

    // Animación funcional de llenado
    setTimeout(() => {
      const fill = item.querySelector('.prob-bar-fill');
      if (fill) fill.style.width = `${pct}%`;
    }, 40);
  });
}

// ============================================================================
// HISTORIAL DE LAS ÚLTIMAS 5 PREDICCIONES (TABLA EN MEMORIA)
// ============================================================================
function agregarAlHistorial(registro) {
  state.history.unshift(registro);

  if (state.history.length > CONFIG.MAX_HISTORY_ITEMS) {
    state.history.pop();
  }

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
  DOM.historyCounter.textContent = `${state.history.length} de ${CONFIG.MAX_HISTORY_ITEMS} registros`;
  DOM.historyGrid.innerHTML = '';

  state.history.forEach((item) => {
    const hora = item.timestamp.toLocaleTimeString([], { hour: '2-digit', minute: '2-digit', second: '2-digit' });

    const row = document.createElement('tr');
    row.innerHTML = `
      <td>${hora}</td>
      <td><img src="${item.snapshotDataUrl}" alt="${item.emocion}" class="history-thumb"></td>
      <td><strong>${item.emocion}</strong></td>
      <td>${item.confianza}%</td>
      <td><span class="academic-badge ${item.estado_class}">${item.estado_academico}</span></td>
    `;

    DOM.historyGrid.appendChild(row);
  });
}

// ============================================================================
// MANEJO DE ERRORES Y ALERTAS VISUALES
// ============================================================================
function mostrarError(mensaje) {
  DOM.errorBannerText.textContent = mensaje;
  DOM.errorBanner.classList.remove('hidden');
}
