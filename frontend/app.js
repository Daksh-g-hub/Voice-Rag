// Initialize Lucide icons
lucide.createIcons();

// DOM Elements
const micBtn = document.getElementById('micBtn');
const micBtnText = document.getElementById('micBtnText');
const textQueryInput = document.getElementById('textQueryInput');
const sendTextBtn = document.getElementById('sendTextBtn');
const recordingTimer = document.getElementById('recordingTimer');
const visualizerCanvas = document.getElementById('visualizerCanvas');
const visualizerPlaceholder = document.getElementById('visualizerPlaceholder');
const statusBadge = document.getElementById('statusBadge');
const statusText = document.getElementById('statusText');
const answerContent = document.getElementById('answerContent');
const transcriptBox = document.getElementById('transcriptBox');
const transcriptText = document.getElementById('transcriptText');
const groundingBadge = document.getElementById('groundingBadge');
const responseFooter = document.getElementById('responseFooter');
const confidenceVal = document.getElementById('confidenceVal');
const groundingVal = document.getElementById('groundingVal');
const totalLatencyVal = document.getElementById('totalLatencyVal');
const coreRetrievalMetric = document.getElementById('coreRetrievalMetric');
const totalPipelineMetric = document.getElementById('totalPipelineMetric');
const contextList = document.getElementById('contextList');
const contextCount = document.getElementById('contextCount');
const copyBtn = document.getElementById('copyBtn');
const copyBtnText = document.getElementById('copyBtnText');
const ttsPlayBtn = document.getElementById('ttsPlayBtn');

// State
let isRecording = false;
let mediaRecorder = null;
let audioChunks = [];
let audioContext = null;
let analyser = null;
let animationFrameId = null;
let recordStartTime = null;
let recordTimerInterval = null;
let latestRawAnswer = "";

const API_BASE = window.location.origin;

// Canvas Visualizer Setup
const canvasCtx = visualizerCanvas.getContext('2d');

function resizeCanvas() {
    visualizerCanvas.width = visualizerCanvas.parentElement.clientWidth;
    visualizerCanvas.height = visualizerCanvas.parentElement.clientHeight;
}
window.addEventListener('resize', resizeCanvas);
resizeCanvas();

function setupAudioVisualizer(stream) {
    try {
        audioContext = new (window.AudioContext || window.webkitAudioContext)();
        analyser = audioContext.createAnalyser();
        analyser.fftSize = 128;
        analyser.smoothingTimeConstant = 0.8;
        
        const source = audioContext.createMediaStreamSource(stream);
        source.connect(analyser);

        visualizerPlaceholder.classList.add('hidden');
        drawVisualizer();
    } catch (e) {
        console.warn("Visualizer init error:", e);
    }
}

// Theme Management
const THEMES = {
    emerald: { primary: '#10b981', secondary: '#06b6d4' },
    nebula: { primary: '#8b5cf6', secondary: '#ec4899' },
    solar: { primary: '#f59e0b', secondary: '#f43f5e' },
    cyan: { primary: '#06b6d4', secondary: '#3b82f6' }
};

let currentTheme = localStorage.getItem('voice_rag_theme') || 'emerald';
setTheme(currentTheme);

function setTheme(themeName) {
    if (!THEMES[themeName]) themeName = 'emerald';
    currentTheme = themeName;
    document.documentElement.setAttribute('data-theme', themeName);
    localStorage.setItem('voice_rag_theme', themeName);
    
    // Update active ring on buttons
    document.querySelectorAll('.theme-btn').forEach(btn => {
        if (btn.getAttribute('data-theme') === themeName) {
            btn.classList.add('ring-2', 'ring-white', 'scale-110');
        } else {
            btn.classList.remove('ring-2', 'ring-white', 'scale-110');
        }
    });
}

document.querySelectorAll('.theme-btn').forEach(btn => {
    btn.addEventListener('click', () => {
        const theme = btn.getAttribute('data-theme');
        setTheme(theme);
    });
});

function drawVisualizer() {
    if (!analyser) return;
    animationFrameId = requestAnimationFrame(drawVisualizer);

    const bufferLength = analyser.frequencyBinCount;
    const dataArray = new Uint8Array(bufferLength);
    analyser.getByteFrequencyData(dataArray);

    canvasCtx.clearRect(0, 0, visualizerCanvas.width, visualizerCanvas.height);

    const totalBars = 32;
    const barWidth = (visualizerCanvas.width / totalBars) * 0.7;
    const gap = (visualizerCanvas.width - (barWidth * totalBars)) / (totalBars + 1);
    let x = gap;

    const themeColors = THEMES[currentTheme] || THEMES.emerald;

    for (let i = 0; i < totalBars; i++) {
        const sampleIndex = Math.floor(i * (bufferLength / totalBars));
        const val = dataArray[sampleIndex] || 0;
        const barHeight = Math.max(3, (val / 255) * (visualizerCanvas.height * 0.85));

        const gradient = canvasCtx.createLinearGradient(0, visualizerCanvas.height, 0, visualizerCanvas.height - barHeight);
        gradient.addColorStop(0, themeColors.primary);
        gradient.addColorStop(1, themeColors.secondary);

        canvasCtx.fillStyle = gradient;
        
        const y = (visualizerCanvas.height - barHeight) / 2;
        const radius = barWidth / 2;
        
        canvasCtx.beginPath();
        canvasCtx.roundRect(x, y, barWidth, barHeight, [radius]);
        canvasCtx.fill();

        x += barWidth + gap;
    }
}

function stopVisualizer() {
    if (animationFrameId) {
        cancelAnimationFrame(animationFrameId);
    }
    if (audioContext && audioContext.state !== 'closed') {
        audioContext.close();
    }
    canvasCtx.clearRect(0, 0, visualizerCanvas.width, visualizerCanvas.height);
    visualizerPlaceholder.classList.remove('hidden');
}

// Microphone Audio Capture Handler
async function startRecording() {
    try {
        textQueryInput.value = "";
        transcriptBox.classList.add('hidden');
        transcriptText.textContent = "";

        const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
        setupAudioVisualizer(stream);

        mediaRecorder = new MediaRecorder(stream);
        audioChunks = [];

        mediaRecorder.ondataavailable = (e) => {
            if (e.data.size > 0) {
                audioChunks.push(e.data);
            }
        };

        mediaRecorder.onstop = async () => {
            const audioBlob = new Blob(audioChunks, { type: mediaRecorder.mimeType || 'audio/webm' });
            stream.getTracks().forEach(track => track.stop());
            stopVisualizer();
            await processVoiceQuery(audioBlob);
        };

        mediaRecorder.start(250);
        isRecording = true;

        micBtn.className = "w-full sm:w-auto px-6 py-3.5 rounded-xl bg-red-600 hover:bg-red-500 active:scale-[0.98] text-white font-semibold text-sm flex items-center justify-center gap-2.5 transition-all glow-recording cursor-pointer";
        micBtnText.textContent = "Stop Recording";
        
        recordStartTime = Date.now();
        recordTimerInterval = setInterval(() => {
            const elapsedSec = ((Date.now() - recordStartTime) / 1000).toFixed(1);
            recordingTimer.textContent = `Recording • ${elapsedSec}s`;
        }, 100);

    } catch (err) {
        console.error("Microphone access denied:", err);
        alert("Microphone permission is required for voice queries. Please allow mic access in your browser.");
    }
}

function stopRecording() {
    if (mediaRecorder && isRecording) {
        mediaRecorder.stop();
        isRecording = false;
        clearInterval(recordTimerInterval);
        recordingTimer.textContent = "Processing audio...";
        micBtn.className = "w-full sm:w-auto px-6 py-3.5 rounded-xl bg-gradient-to-r from-brand-600 to-indigo-600 hover:from-brand-500 hover:to-indigo-500 active:scale-[0.98] text-white font-semibold text-sm flex items-center justify-center gap-2.5 transition-all shadow-lg shadow-brand-500/25 cursor-pointer";
        micBtnText.textContent = "Record Voice";
    }
}

micBtn.addEventListener('click', () => {
    if (!isRecording) {
        startRecording();
    } else {
        stopRecording();
    }
});

// Text Query Handler
async function processTextQuery(queryText) {
    if (!queryText.trim()) return;

    setLoadingState(true);
    transcriptBox.classList.add('hidden');

    try {
        const response = await fetch(`${API_BASE}/api/query/text`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ query: queryText, language_code: "en-IN", use_cache: false })
        });

        if (!response.ok) {
            const errData = await response.json().catch(() => ({}));
            throw new Error(errData.detail || `Server error ${response.status}`);
        }

        const data = await response.json();
        renderPipelineResponse(data);
    } catch (err) {
        renderError(err.message);
    } finally {
        setLoadingState(false);
    }
}

// Voice Query Handler
async function processVoiceQuery(audioBlob) {
    setLoadingState(true);
    recordingTimer.textContent = "Transcribing & Synthesizing...";

    try {
        const formData = new FormData();
        formData.append('audio', audioBlob, 'speech.webm');
        formData.append('language_code', 'en-IN');
        formData.append('use_cache', 'false');

        const response = await fetch(`${API_BASE}/api/query/voice`, {
            method: 'POST',
            body: formData
        });

        if (!response.ok) {
            const errData = await response.json().catch(() => ({}));
            throw new Error(errData.detail || `Voice server error ${response.status}`);
        }

        const data = await response.json();
        renderPipelineResponse(data);
    } catch (err) {
        renderError(err.message);
    } finally {
        setLoadingState(false);
        recordingTimer.textContent = "Ready to listen";
    }
}

sendTextBtn.addEventListener('click', () => {
    const q = textQueryInput.value;
    if (q) processTextQuery(q);
});

textQueryInput.addEventListener('keydown', (e) => {
    if (e.key === 'Enter') {
        const q = textQueryInput.value;
        if (q) processTextQuery(q);
    }
});

// Quick Sample Pills
document.querySelectorAll('.sample-pill').forEach(btn => {
    btn.addEventListener('click', () => {
        const q = btn.getAttribute('data-query');
        textQueryInput.value = q;
        processTextQuery(q);
    });
});

// UI State & Response Rendering
function setLoadingState(loading) {
    if (loading) {
        statusBadge.className = "flex items-center gap-2 bg-amber-500/10 border border-amber-500/20 text-amber-400 px-3 py-1.5 rounded-xl text-xs font-mono";
        statusText.textContent = "Executing Pipeline...";
        copyBtn.classList.add('hidden');
        ttsPlayBtn.classList.add('hidden');
        answerContent.innerHTML = `
            <div class="flex flex-col items-center justify-center py-6 space-y-3">
                <div class="w-8 h-8 border-2 border-brand-500 border-t-transparent rounded-full animate-spin"></div>
                <span class="text-xs font-mono text-slate-400">Embedding & Searching MSMARCO-XI Hierarchical Index...</span>
            </div>
        `;
    } else {
        statusBadge.className = "flex items-center gap-2 bg-emerald-500/10 border border-emerald-500/20 text-emerald-400 px-3 py-1.5 rounded-xl text-xs font-mono";
        statusText.textContent = "System Ready";
    }
}

function renderPipelineResponse(data) {
    latestRawAnswer = data.answer || "";

    // 1. Transcript (if voice)
    if (data.transcript) {
        transcriptBox.classList.remove('hidden');
        transcriptText.textContent = data.transcript;
    } else {
        transcriptBox.classList.add('hidden');
    }

    // 2. Formatted Answer with interactive citation tags
    let formattedAnswer = data.answer || "";
    formattedAnswer = formattedAnswer.replace(/\[Source\s+(\d+)\]/gi, (match, p1) => {
        return `<span class="citation-tag inline-flex items-center gap-1 bg-brand-500/15 text-brand-300 border border-brand-500/30 px-2 py-0.5 rounded-md text-xs font-mono font-semibold mx-1 cursor-pointer hover:bg-brand-500/30 transition-all" data-source="${p1}"><i data-lucide="bookmark" class="w-3 h-3"></i>Source ${p1}</span>`;
    });

    answerContent.innerHTML = `<div class="text-slate-100 text-sm leading-relaxed">${formattedAnswer}</div>`;
    lucide.createIcons();

    // 3. Grounding status badge
    groundingBadge.classList.remove('hidden');
    copyBtn.classList.remove('hidden');
    ttsPlayBtn.classList.remove('hidden');

    if (data.status === "security_blocked") {
        groundingBadge.className = "text-xs font-mono px-2.5 py-1 rounded-lg border bg-red-500/10 border-red-500/30 text-red-400";
        groundingBadge.textContent = "Security Blocked";
    } else if (data.is_grounded && data.retrieved_contexts && data.retrieved_contexts.length > 0) {
        groundingBadge.className = "text-xs font-mono px-2.5 py-1 rounded-lg border bg-emerald-500/10 border-emerald-500/30 text-emerald-400 flex items-center gap-1";
        groundingBadge.innerHTML = `<i data-lucide="check-check" class="w-3.5 h-3.5"></i> Grounded (MSMARCO-XI)`;
    } else {
        groundingBadge.className = "text-xs font-mono px-2.5 py-1 rounded-lg border bg-cyan-500/10 border-cyan-500/30 text-cyan-400 flex items-center gap-1";
        groundingBadge.innerHTML = `<i data-lucide="sparkles" class="w-3.5 h-3.5"></i> General Knowledge`;
    }
    lucide.createIcons();

    // 4. Telemetry Strip
    responseFooter.classList.remove('hidden');
    confidenceVal.textContent = `${(data.confidence_score * 100).toFixed(1)}%`;
    groundingVal.textContent = data.is_grounded ? "Verified ✓" : "Synthesized";
    totalLatencyVal.textContent = `${data.total_latency_ms.toFixed(1)} ms`;

    // 5. Latency Waterfall Bars
    updateLatencyWaterfall(data.latency_breakdown, data.total_latency_ms);

    // 6. Context Inspector
    renderContextInspector(data.retrieved_contexts);
}

function updateLatencyWaterfall(lat, totalMs) {
    const maxScale = Math.max(totalMs || 100, 60);

    const stages = [
        { key: 'stt_ms', textId: 'time_stt', barId: 'bar_stt' },
        { key: 'input_guard_ms', textId: 'time_guard', barId: 'bar_guard' },
        { key: 'embedding_ms', textId: 'time_embed', barId: 'bar_embed' },
        { key: 'vector_search_ms', textId: 'time_vector', barId: 'bar_vector' },
        { key: 'parent_resolution_ms', textId: 'time_rerank', barId: 'bar_rerank' },
        { key: 'llm_generation_ms', textId: 'time_llm', barId: 'bar_llm' },
        { key: 'grounding_guard_ms', textId: 'time_grounding', barId: 'bar_grounding' }
    ];

    let coreMs = (lat.embedding_ms || 0) + (lat.vector_search_ms || 0) + (lat.parent_resolution_ms || 0) + (lat.rerank_ms || 0);

    coreRetrievalMetric.textContent = `${coreMs.toFixed(1)} ms`;
    totalPipelineMetric.textContent = `${(totalMs || 0).toFixed(1)} ms`;

    stages.forEach(s => {
        const val = lat[s.key] || 0;
        document.getElementById(s.textId).textContent = `${val.toFixed(1)} ms`;
        const pct = Math.min(100, Math.max(3, (val / maxScale) * 100));
        document.getElementById(s.barId).style.width = `${pct}%`;
    });
}

function renderContextInspector(contexts) {
    if (!contexts || contexts.length === 0) {
        contextList.innerHTML = `<div class="text-slate-500 text-center py-8 italic font-mono">No MSMARCO-XI passages matched (General AI mode).</div>`;
        contextCount.textContent = "0 Sources";
        return;
    }

    contextCount.textContent = `${contexts.length} Sources`;
    let html = "";

    contexts.forEach((ctx, idx) => {
        const sourceNum = idx + 1;
        const score = ctx.max_child_score || ctx.rerank_score || 0;
        const parentId = ctx.parent_id || ctx.doc_id || `source_${sourceNum}`;
        const text = ctx.parent_text || ctx.text || "";
        const lang = ctx.language || "en";

        html += `
            <div id="source-card-${sourceNum}" class="glass-panel-interactive rounded-xl p-4 space-y-2">
                <div class="flex items-center justify-between">
                    <div class="flex items-center gap-2">
                        <span class="font-mono text-brand-400 font-bold text-xs">[Source ${sourceNum}]</span>
                        <span class="text-slate-400 font-mono text-[11px]">${parentId}</span>
                    </div>
                    <div class="flex items-center gap-1.5">
                        <span class="bg-brand-500/10 text-brand-300 border border-brand-500/20 px-2 py-0.5 rounded font-mono text-[10px]">
                            ${lang.toUpperCase()}
                        </span>
                        <span class="bg-emerald-500/10 text-emerald-400 border border-emerald-500/20 px-2 py-0.5 rounded font-mono text-[10px] font-semibold">
                            ${(score * 100).toFixed(1)}% match
                        </span>
                    </div>
                </div>
                <p class="text-slate-300 text-xs leading-relaxed font-sans">${text}</p>
            </div>
        `;
    });

    contextList.innerHTML = html;

    // Attach click highlight events on citation tags
    document.querySelectorAll('.citation-tag').forEach(tag => {
        tag.addEventListener('click', () => {
            const sNum = tag.getAttribute('data-source');
            const targetCard = document.getElementById(`source-card-${sNum}`);
            if (targetCard) {
                targetCard.scrollIntoView({ behavior: 'smooth', block: 'nearest' });
                targetCard.classList.add('ring-2', 'ring-brand-500');
                setTimeout(() => targetCard.classList.remove('ring-2', 'ring-brand-500'), 1500);
            }
        });
    });
}

// Copy Answer Feature
copyBtn.addEventListener('click', () => {
    if (!latestRawAnswer) return;
    navigator.clipboard.writeText(latestRawAnswer);
    copyBtnText.textContent = "Copied!";
    setTimeout(() => { copyBtnText.textContent = "Copy"; }, 1800);
});

// Text-to-Speech (TTS) Read Aloud Feature
ttsPlayBtn.addEventListener('click', () => {
    if (!latestRawAnswer) return;
    if ('speechSynthesis' in window) {
        window.speechSynthesis.cancel();
        // Strip [Source X] for clean voice output
        const cleanSpeech = latestRawAnswer.replace(/\[Source\s+\d+\]/gi, '').trim();
        const utterance = new SpeechSynthesisUtterance(cleanSpeech);
        utterance.rate = 1.05;
        utterance.pitch = 1.0;
        window.speechSynthesis.speak(utterance);
    }
});

function renderError(errMsg) {
    answerContent.innerHTML = `
        <div class="text-red-400 bg-red-500/10 border border-red-500/20 rounded-xl p-4 text-xs font-mono space-y-1">
            <strong>Error executing pipeline:</strong>
            <p>${errMsg}</p>
        </div>
    `;
    statusBadge.className = "flex items-center gap-2 bg-red-500/10 border border-red-500/20 text-red-400 px-3 py-1.5 rounded-xl text-xs font-mono";
    statusText.textContent = "Pipeline Error";
}
