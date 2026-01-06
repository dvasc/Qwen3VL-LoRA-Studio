const els = {
    // DOM Elements
    dropZoneTrain: document.getElementById('dropZoneTrain'),
    fileInputTrain: document.getElementById('fileInputTrain'),
    fileListTrain: document.getElementById('fileListTrain'),
    dropZoneVal: document.getElementById('dropZoneVal'),
    fileInputVal: document.getElementById('fileInputVal'),
    fileListVal: document.getElementById('fileListVal'),
    startBtn: document.getElementById('startBtn'),
    configView: document.getElementById('configView'),
    trainingView: document.getElementById('trainingView'),
    useThinking: document.getElementById('useThinking'),
    progressBar: document.getElementById('progressBar'),
    terminal: document.getElementById('terminalLog'),
    monitorStatus: document.getElementById('monitorStatus'),
    monitorTimer: document.getElementById('monitorTimer'),
    monitorETR: document.getElementById('monitorETR'),
    monitorPercent: document.getElementById('monitorPercent'),
    gpuName: document.getElementById('gpuName'),
    gpuUtil: document.getElementById('gpuUtil'),
    gpuUtilBar: document.getElementById('gpuUtilBar'),
    vramUsage: document.getElementById('vramUsage'),
    vramBar: document.getElementById('vramBar'),
    gpuTemp: document.getElementById('gpuTemp'),
    activeControls: document.getElementById('activeControls'),
    stopBtn: document.getElementById('stopBtn'),
    completionActions: document.getElementById('completionActions'),
    downloadBtn: document.getElementById('downloadBtn'),
    resetBtn: document.getElementById('resetBtn'),
    validationResults: document.getElementById('validationResults'),
    metricsBody: document.getElementById('metricsBody')
};

let pollInterval = null;
let trainFileCount = 0;
let trainingStartTime = null; // Stores Unix timestamp for live timer

// --- Helper to format time ---
function formatTime(seconds) {
    if (seconds === null || isNaN(seconds)) return "00:00";
    const h = Math.floor(seconds / 3600);
    const m = Math.floor((seconds % 3600) / 60);
    const s = Math.floor(seconds % 60);
    const pad = (num) => String(num).padStart(2, '0');
    if (h > 0) {
        return `${pad(h)}:${pad(m)}:${pad(s)}`;
    }
    return `${pad(m)}:${pad(s)}`;
}

// --- Setup Helper for Drop Zones ---
function setupDropZone(zone, input, listElement, type) {
    zone.onclick = () => input.click();
    zone.ondragover = (e) => { e.preventDefault(); zone.classList.add('dragover'); };
    zone.ondragleave = () => zone.classList.remove('dragover');
    zone.ondrop = (e) => {
        e.preventDefault();
        zone.classList.remove('dragover');
        handleUpload(e.dataTransfer.files, type, listElement);
    };
    input.onchange = (e) => handleUpload(e.target.files, type, listElement);
}
setupDropZone(els.dropZoneTrain, els.fileInputTrain, els.fileListTrain, 'train');
setupDropZone(els.dropZoneVal, els.fileInputVal, els.fileListVal, 'val');

async function handleUpload(files, type, listElement) {
    const formData = new FormData();
    formData.append('type', type);
    for (let i = 0; i < files.length; i++) {
        formData.append('files', files[i]);
    }
    try {
        const res = await fetch('/upload', { method: 'POST', body: formData });
        const data = await res.json();
        listElement.innerHTML = data.paths.map(f => `<div><i class="fa-solid fa-file-code"></i> ${f}</div>`).join('');
        if (type === 'train') {
            trainFileCount = data.count;
            els.startBtn.disabled = trainFileCount === 0;
        }
    } catch (e) {
        alert(`Upload failed for ${type} data.`);
    }
}

// --- Training Logic ---
els.startBtn.onclick = async () => {
    const config = {
        base_model: document.getElementById('baseModel').value,
        epochs: document.getElementById('epochs').value,
        batch_size: document.getElementById('batchSize').value,
        lr: document.getElementById('lr').value,
        use_thinking: els.useThinking.checked
    };
    const res = await fetch('/train', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(config)
    });
    if (res.ok) {
        els.configView.classList.remove('active');
        els.trainingView.classList.add('active');
        els.stopBtn.disabled = false;
        els.activeControls.classList.remove('hidden');
        els.completionActions.classList.add('hidden');
        els.validationResults.classList.add('hidden');
        trainingStartTime = null; // Reset timer state
        startPolling();
    }
};

els.stopBtn.onclick = async () => {
    if (!confirm("Are you sure you want to stop? Partial progress will be saved.")) return;
    els.stopBtn.disabled = true;
    await fetch('/stop', { method: 'POST' });
};

function startPolling() {
    pollInterval = setInterval(async () => {
        try {
            const res = await fetch('/status');
            const state = await res.json();

            // Live Timer Logic
            if (state.status === 'TRAINING' && state.start_time) {
                if (!trainingStartTime) {
                    trainingStartTime = state.start_time;
                }
                const now = new Date().getTime() / 1000;
                const elapsed = now - trainingStartTime;
                els.monitorTimer.textContent = formatTime(elapsed);
            } else {
                els.monitorTimer.textContent = state.duration;
            }

            // Update UI from state
            els.monitorStatus.textContent = state.status;
            els.monitorETR.textContent = state.etr || "--:--";
            els.monitorPercent.textContent = state.progress + '%';
            els.progressBar.style.width = state.progress + '%';
            els.terminal.innerHTML = state.logs.map(l => `<div class="log-line">${l}</div>`).join('');
            els.terminal.scrollTop = els.terminal.scrollHeight;

            // Handle end-of-training states
            if (['FINISHED', 'INTERRUPTED', 'ERROR'].includes(state.status)) {
                clearInterval(pollInterval);
                trainingStartTime = null;
                els.monitorStatus.style.color = state.status === 'FINISHED' ? 'var(--success)' : (state.status === 'ERROR' ? '#ef4444' : '#f59e0b');
                els.activeControls.classList.add('hidden');
                if (state.status !== 'ERROR') {
                    els.completionActions.classList.remove('hidden');
                    if (state.val_metrics) renderMetrics(state.val_metrics);
                } else {
                    alert("Error: " + state.error_msg);
                }
            }

            const hwRes = await fetch('/hardware-status');
            const hw = await hwRes.json();
            updateHardwareUI(hw);

        } catch (err) {
            console.error("Polling error:", err);
        }
    }, 1000);
}

function updateHardwareUI(hw) {
    if (hw.available) {
        els.gpuName.textContent = hw.gpu_name;
        els.gpuUtil.textContent = hw.utilization + '%';
        els.gpuUtilBar.style.width = hw.utilization + '%';
        els.vramUsage.textContent = `${hw.vram_used} / ${hw.vram_total} GB`;
        if (hw.vram_total > 0) {
            els.vramBar.style.width = (hw.vram_used / hw.vram_total) * 100 + '%';
        }
        els.gpuTemp.textContent = hw.temp + '°C';
    } else {
        els.gpuName.textContent = "HARDWARE NOT DETECTED";
        els.gpuUtil.textContent = "ERR";
        els.vramUsage.textContent = "N/A";
        els.gpuTemp.textContent = "ERR";
        
        // Style error states
        [els.gpuName, els.gpuUtil, els.gpuTemp].forEach(el => el.style.color = '#ef4444');
        els.gpuUtilBar.style.width = '0%';
        els.vramBar.style.width = '0%';
    }
}

function renderMetrics(metrics) {
    els.validationResults.classList.remove('hidden');
    const keys = Object.keys(metrics).sort((a, b) => (a === 'eval_loss' ? -1 : b === 'eval_loss' ? 1 : 0));
    els.metricsBody.innerHTML = keys.map(key => {
        let label = key.replace('eval_', '').replace(/_/g, ' ').toUpperCase();
        let value = typeof metrics[key] === 'number' ? metrics[key].toFixed(4) : metrics[key];
        return `<div class="metric-row"><span class="metric-label">${label}</span><span class="metric-val">${value}</span></div>`;
    }).join('');
}

els.downloadBtn.onclick = () => window.location.href = '/download';
els.resetBtn.onclick = async () => {
    await fetch('/reset', { method: 'POST' });
    location.reload();
};