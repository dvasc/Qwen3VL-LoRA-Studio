const els = {
    // Training Data Zone
    dropZoneTrain: document.getElementById('dropZoneTrain'),
    fileInputTrain: document.getElementById('fileInputTrain'),
    fileListTrain: document.getElementById('fileListTrain'),
    
    // Validation Data Zone
    dropZoneVal: document.getElementById('dropZoneVal'),
    fileInputVal: document.getElementById('fileInputVal'),
    fileListVal: document.getElementById('fileListVal'),

    // Config Controls
    startBtn: document.getElementById('startBtn'),
    configView: document.getElementById('configView'),
    trainingView: document.getElementById('trainingView'),
    useThinking: document.getElementById('useThinking'), // Checkbox
    
    // Monitor Elements
    progressBar: document.getElementById('progressBar'),
    terminal: document.getElementById('terminalLog'),
    monitorStatus: document.getElementById('monitorStatus'),
    monitorTimer: document.getElementById('monitorTimer'),
    monitorETR: document.getElementById('monitorETR'),
    monitorPercent: document.getElementById('monitorPercent'),
    
    // Hardware Elements
    gpuName: document.getElementById('gpuName'),
    gpuUtil: document.getElementById('gpuUtil'),
    vramUsage: document.getElementById('vramUsage'),
    gpuTemp: document.getElementById('gpuTemp'),
    
    // Active Controls
    activeControls: document.getElementById('activeControls'),
    stopBtn: document.getElementById('stopBtn'),
    
    // Completion & Results
    completionActions: document.getElementById('completionActions'),
    downloadBtn: document.getElementById('downloadBtn'),
    resetBtn: document.getElementById('resetBtn'),
    validationResults: document.getElementById('validationResults'),
    metricsBody: document.getElementById('metricsBody')
};

let pollInterval = null;
let trainFileCount = 0;

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

// Initialize Zones
setupDropZone(els.dropZoneTrain, els.fileInputTrain, els.fileListTrain, 'train');
setupDropZone(els.dropZoneVal, els.fileInputVal, els.fileListVal, 'val');

async function handleUpload(files, type, listElement) {
    const formData = new FormData();
    formData.append('type', type); // Identify 'train' or 'val'
    for (let i = 0; i < files.length; i++) {
        formData.append('files', files[i]);
    }

    try {
        const res = await fetch('/upload', { method: 'POST', body: formData });
        const data = await res.json();
        
        // Render File List
        listElement.innerHTML = '';
        data.paths.forEach(filename => {
            const div = document.createElement('div');
            div.style.padding = '8px';
            div.style.borderBottom = '1px solid var(--border)';
            div.style.fontSize = '0.8rem';
            div.innerHTML = `<i class="fa-solid fa-file-code" style="margin-right:8px"></i> ${filename}`;
            listElement.appendChild(div);
        });

        // Update State Logic
        if (type === 'train') {
            trainFileCount = data.count;
            if (trainFileCount > 0) {
                els.startBtn.disabled = false;
                els.startBtn.innerHTML = `<i class="fa-solid fa-bolt"></i> Start Training`;
            } else {
                els.startBtn.disabled = true;
                els.startBtn.innerHTML = `<i class="fa-solid fa-bolt"></i> Start Training Run`;
            }
        }
    } catch (e) {
        console.error(e);
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
        // Send the thinking flag state
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
        // Reset UI State
        els.stopBtn.disabled = false;
        els.stopBtn.innerHTML = '<i class="fa-solid fa-hand"></i> STOP TRAINING';
        els.activeControls.classList.remove('hidden');
        els.completionActions.classList.add('hidden');
        els.validationResults.classList.add('hidden');
        startPolling();
    }
};

// --- Stop Logic ---
els.stopBtn.onclick = async () => {
    if (!confirm("Are you sure you want to stop? Partial progress will be saved.")) return;
    
    els.stopBtn.disabled = true;
    els.stopBtn.innerHTML = '<i class="fa-solid fa-spinner fa-spin"></i> STOPPING...';
    
    try {
        await fetch('/stop', { method: 'POST' });
    } catch (e) {
        els.stopBtn.disabled = false;
    }
};

function startPolling() {
    pollInterval = setInterval(async () => {
        try {
            // 1. Poll Status
            const res = await fetch('/status');
            const state = await res.json();

            // Basic Metrics
            els.monitorStatus.textContent = state.status;
            els.monitorTimer.textContent = state.duration;
            els.monitorETR.textContent = state.etr || "--:--";
            els.monitorPercent.textContent = state.progress + '%';
            els.progressBar.style.width = state.progress + '%';

            // Logs
            els.terminal.innerHTML = state.logs.map(l => `<div class="log-line">${l}</div>`).join('');
            els.terminal.scrollTop = els.terminal.scrollHeight;

            // Completion Handling
            if (state.status === 'FINISHED' || state.status === 'INTERRUPTED') {
                clearInterval(pollInterval);
                els.monitorStatus.style.color = state.status === 'FINISHED' ? 'var(--success)' : '#f59e0b';
                els.activeControls.classList.add('hidden');
                els.completionActions.classList.remove('hidden');

                // Render Validation Metrics if available
                if (state.val_metrics) {
                    renderMetrics(state.val_metrics);
                }
            } else if (state.status === 'ERROR') {
                clearInterval(pollInterval);
                els.monitorStatus.style.color = '#ef4444';
                els.activeControls.classList.add('hidden');
                alert("Error: " + state.error_msg);
            }

            // 2. Poll Hardware
            const hwRes = await fetch('/hardware-status');
            const hw = await hwRes.json();
            
            if (hw.available) {
                els.gpuName.textContent = hw.gpu_name;
                els.gpuUtil.textContent = hw.utilization + '%';
                els.vramUsage.textContent = `${hw.vram_used} / ${hw.vram_total} GB`;
                els.gpuTemp.textContent = hw.temp + '°C';
                els.gpuUtil.style.color = hw.utilization > 90 ? '#ef4444' : 'var(--accent)';
            }
        } catch (err) {
            console.error("Polling error:", err);
        }
    }, 1000);
}

function renderMetrics(metrics) {
    els.validationResults.classList.remove('hidden');
    let html = '';
    
    // Sort keys to put Loss first
    const keys = Object.keys(metrics).sort((a, b) => {
        if (a === 'eval_loss') return -1;
        if (b === 'eval_loss') return 1;
        return 0;
    });

    for (const key of keys) {
        // Clean up key names (e.g., eval_loss -> EVAL LOSS)
        let label = key.replace('eval_', '').replace(/_/g, ' ').toUpperCase();
        let value = metrics[key];
        
        // Format numbers
        if (typeof value === 'number') {
            value = value % 1 !== 0 ? value.toFixed(4) : value;
        }
        
        html += `
        <div class="metric-row">
            <span class="metric-label">${label}</span>
            <span class="metric-val">${value}</span>
        </div>`;
    }
    els.metricsBody.innerHTML = html;
}

// --- Footer Actions ---
els.downloadBtn.onclick = () => window.location.href = '/download';
els.resetBtn.onclick = async () => {
    await fetch('/reset', { method: 'POST' });
    location.reload();
};