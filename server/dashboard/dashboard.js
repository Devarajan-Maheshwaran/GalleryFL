let ws;
let accChart, lossChart, comparisonChart, f1Chart;
let accData = [], lossData = [], roundLabels = [];
let clients = {};
let isTraining = false;

const COLORS = {
    primary: '#007aff', /* Apple Blue */
    gold: '#ff9500', /* Apple Orange */
    text: '#86868b',
    grid: 'rgba(0, 0, 0, 0.05)',
    fill: 'rgba(0, 122, 255, 0.1)',
    fillLoss: 'rgba(255, 149, 0, 0.1)',
};

function initCharts() {
    const baseOptions = {
        responsive: true,
        maintainAspectRatio: false,
        animation: { duration: 400, easing: 'easeOutQuart' },
        scales: {
            x: { grid: { color: COLORS.grid }, ticks: { color: COLORS.text, font: { family: 'JetBrains Mono', size: 11 } } },
            y: { grid: { color: COLORS.grid }, ticks: { color: COLORS.text, font: { family: 'JetBrains Mono', size: 11 } } }
        },
        plugins: {
            legend: { display: false }
        }
    };

    accChart = new Chart(document.getElementById('accChart').getContext('2d'), {
        type: 'line',
        data: {
            labels: roundLabels,
            datasets: [{
                label: 'Accuracy',
                data: accData,
                borderColor: COLORS.primary,
                backgroundColor: COLORS.fill,
                fill: true,
                tension: 0.3,
                pointRadius: 3,
                pointBackgroundColor: COLORS.primary,
                borderWidth: 2,
            }]
        },
        options: { ...baseOptions, scales: { ...baseOptions.scales, y: { ...baseOptions.scales.y, min: 0, max: 1 } } }
    });

    lossChart = new Chart(document.getElementById('lossChart').getContext('2d'), {
        type: 'line',
        data: {
            labels: roundLabels,
            datasets: [{
                label: 'Loss',
                data: lossData,
                borderColor: COLORS.gold,
                backgroundColor: COLORS.fillLoss,
                fill: true,
                tension: 0.3,
                pointRadius: 3,
                pointBackgroundColor: COLORS.gold,
                borderWidth: 2,
            }]
        },
        options: baseOptions
    });

    comparisonChart = new Chart(document.getElementById('comparisonChart').getContext('2d'), {
        type: 'bar',
        data: {
            labels: [],
            datasets: [
                {
                    label: 'Baseline (Pre-trained)',
                    data: [],
                    backgroundColor: 'rgba(0, 0, 0, 0.05)',
                    borderColor: 'rgba(0, 0, 0, 0.2)',
                    borderWidth: 1
                },
                {
                    label: 'Federated Model',
                    data: [],
                    backgroundColor: 'rgba(0, 122, 255, 0.2)',
                    borderColor: '#007aff',
                    borderWidth: 1
                }
            ]
        },
        options: {
            ...baseOptions,
            plugins: {
                legend: { display: true, labels: { font: { family: 'Outfit', size: 11 } } }
            },
            scales: {
                x: { grid: { color: COLORS.grid } },
                y: { min: 0, max: 1, grid: { color: COLORS.grid } }
            }
        }
    });

    f1Chart = new Chart(document.getElementById('f1Chart').getContext('2d'), {
        type: 'bar',
        data: {
            labels: [],
            datasets: [{
                label: 'F1 Score',
                data: [],
                backgroundColor: 'rgba(88, 86, 214, 0.2)', /* Apple Indigo */
                borderColor: '#5856d6',
                borderWidth: 1
            }]
        },
        options: {
            ...baseOptions,
            indexAxis: 'y',
            scales: {
                x: { min: 0, max: 1, grid: { color: COLORS.grid } },
                y: { grid: { display: false } }
            }
        }
    });
}

function connectWS() {
    const protocol = location.protocol === 'https:' ? 'wss:' : 'ws:';
    ws = new WebSocket(`${protocol}//${location.host}/ws/feed?client_id=dashboard`);

    ws.onopen = () => {
        document.getElementById('ws-status-dot').className = 'nm-dot nm-dot--online';
        document.getElementById('ws-status-text').innerText = 'Connected';
    };

    ws.onclose = () => {
        document.getElementById('ws-status-dot').className = 'nm-dot nm-dot--offline';
        document.getElementById('ws-status-text').innerText = 'Disconnected';
        setTimeout(connectWS, 3000);
    };

    ws.onmessage = (event) => {
        const msg = JSON.parse(event.data);
        routeEvent(msg.type, msg.data);
    };
}

function routeEvent(type, data) {
    switch (type) {
        case 'round_started':
            onRoundStart(data);
            break;
        case 'round_completed':
            onRoundComplete(data);
            break;
        case 'training_complete':
            onTrainingComplete(data);
            break;
        case 'client_connected':
            clients[data.client_id] = data;
            renderClients();
            break;
        case 'client_disconnected':
            delete clients[data.client_id];
            renderClients();
            break;
        case 'update_received':
            highlightClient(data.client_id);
            break;
        case 'clients_cleared':
            clients = {};
            renderClients();
            refreshStatus();
            refreshLeaderboard();
            break;
    }
}

function onRoundStart(data) {
    isTraining = true;
    document.getElementById('phase-badge').innerText = `Round ${data.round}`;
    document.getElementById('phase-badge').style.color = 'var(--accent-primary)';
    document.getElementById('round-progress-label').innerText = `Round ${data.round} / ${data.total_rounds}`;
    document.getElementById('round-progress-fill').style.width = `${(data.round / data.total_rounds) * 100}%`;
    document.getElementById('btn-start').style.display = 'none';
    document.getElementById('btn-stop').style.display = '';
    setInputsDisabled(true);
}

function onRoundComplete(data) {
    roundLabels.push(`R${data.round}`);
    accData.push(data.global_accuracy);
    lossData.push(data.global_loss);

    if (roundLabels.length > 50) {
        roundLabels.shift();
        accData.shift();
        lossData.shift();
    }

    accChart.update();
    lossChart.update();
    refreshLeaderboard();
    refreshStatus();
    refreshComparison();
}

function onTrainingComplete(data) {
    isTraining = false;
    document.getElementById('phase-badge').innerText = 'Complete';
    document.getElementById('phase-badge').style.color = 'var(--success)';
    document.getElementById('round-progress-fill').style.width = '100%';
    document.getElementById('btn-start').style.display = '';
    document.getElementById('btn-stop').style.display = 'none';
    document.getElementById('btn-export-model').disabled = false;
    document.getElementById('btn-export-report').disabled = false;
    setInputsDisabled(false);
}

function renderClients() {
    const list = document.getElementById('client-list');
    const ids = Object.keys(clients);

    if (ids.length === 0) {
        list.innerHTML = '<p class="empty-state">No clients connected</p>';
        return;
    }

    list.innerHTML = '';
    ids.forEach(id => {
        const c = clients[id];
        const card = document.createElement('div');
        card.className = 'client-card';
        card.id = `client-${id}`;
        card.innerHTML = `
            <div class="client-info">
                <h4>${c.nickname || id.substring(0, 8)}</h4>
                <p>${c.device_model || 'Unknown device'}</p>
            </div>
            <span class="nm-dot nm-dot--online"></span>
        `;
        list.appendChild(card);
    });
}

function highlightClient(clientId) {
    const el = document.getElementById(`client-${clientId}`);
    if (el) {
        el.style.boxShadow = '0 0 0 2px var(--accent-gold)';
        setTimeout(() => { el.style.boxShadow = 'var(--nm-raised)'; }, 800);
    }
}

function setInputsDisabled(disabled) {
    document.querySelectorAll('.control-inputs .nm-input').forEach(inp => inp.disabled = disabled);
}

async function refreshStatus() {
    try {
        const res = await fetch('/api/training/status');
        const data = await res.json();
        document.getElementById('model-version').innerText = `v${data.model_version}`;
        document.getElementById('model-ver-display').innerText = data.model_version;
        document.getElementById('client-count').innerText = data.connected_clients;
        document.getElementById('registered-count').innerText = data.registered_clients;
        document.getElementById('param-count').innerText = formatNumber(data.total_parameters);
        document.getElementById('access-code').innerText = data.access_code || '-';

        // Extract the true LAN IP from the access code (e.g. 192.168.x.x:8000)
        if (data.access_code && data.access_code.includes('@')) {
            document.getElementById('lan-ip').innerText = data.access_code.split('@')[0];
        } else {
            document.getElementById('lan-ip').innerText = window.location.host;
        }

        // Sync online clients list from endpoint status response
        if (data.online_clients) {
            const newClients = {};
            data.online_clients.forEach(c => {
                newClients[c.client_id] = c;
            });
            clients = newClients;
            renderClients();
        }
    } catch (e) {}
}

async function refreshLeaderboard() {
    try {
        const res = await fetch('/api/metrics/leaderboard');
        const data = await res.json();
        const tbody = document.querySelector('#leaderboard-table tbody');
        tbody.innerHTML = '';

        data.leaderboard.forEach((c, i) => {
            const tr = document.createElement('tr');
            if (i === 0) tr.className = 'rank-1';
            tr.innerHTML = `
                <td>${i + 1}</td>
                <td>${c.nickname || c.client_id.substring(0, 8)}</td>
                <td class="mono">${c.images}</td>
                <td class="mono">${c.rounds}</td>
                <td class="mono">${c.score}</td>
                <td class="mono">${c.epsilon}</td>
            `;
            tbody.appendChild(tr);
        });
    } catch (e) {}
}

async function refreshComparison() {
    try {
        const res = await fetch('/api/metrics/comparison');
        const data = await res.json();
        if (data.categories && data.categories.length > 0) {
            comparisonChart.data.labels = data.categories;
            comparisonChart.data.datasets[0].data = data.baseline;
            comparisonChart.data.datasets[1].data = data.federated;
            comparisonChart.update();

            f1Chart.data.labels = data.categories;
            f1Chart.data.datasets[0].data = data.federated;
            f1Chart.update();
        }
    } catch (e) {}
}

async function startTraining() {
    const minClients = parseInt(document.getElementById('inp-min-clients').value) || 2;
    const maxRounds = parseInt(document.getElementById('inp-max-rounds').value) || 10;
    const localEpochs = parseInt(document.getElementById('inp-local-epochs').value) || 3;

    try {
        const res = await fetch('/api/training/start', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                min_clients: minClients,
                max_rounds: maxRounds,
                local_epochs: localEpochs
            })
        });
        const data = await res.json();
        if (data.status !== 'started') {
            document.getElementById('phase-badge').innerText = data.status;
        }
    } catch (e) {}
}

async function stopTraining() {
    try {
        await fetch('/api/training/stop', { method: 'POST' });
    } catch (e) {}
}

async function loadHistory() {
    try {
        const res = await fetch('/api/metrics/history');
        const data = await res.json();
        if (data.history && data.history.length > 0) {
            data.history.forEach(h => {
                roundLabels.push(`R${h.round}`);
                accData.push(h.global_accuracy);
                lossData.push(h.global_loss);
            });
            accChart.update();
            lossChart.update();
            document.getElementById('btn-export-model').disabled = false;
            document.getElementById('btn-export-report').disabled = false;
            refreshComparison();
        }
    } catch (e) {}
}

function formatNumber(n) {
    if (!n) return '-';
    if (n >= 1000000) return (n / 1000000).toFixed(1) + 'M';
    if (n >= 1000) return (n / 1000).toFixed(1) + 'K';
    return n.toString();
}

let startTime = Date.now();
setInterval(() => {
    let diff = Math.floor((Date.now() - startTime) / 1000);
    let h = Math.floor(diff / 3600).toString().padStart(2, '0');
    let m = Math.floor((diff % 3600) / 60).toString().padStart(2, '0');
    let s = (diff % 60).toString().padStart(2, '0');
    document.getElementById('uptime').innerText = `${h}:${m}:${s}`;
}, 1000);

function initNavigation() {
    document.querySelectorAll('.sidebar__nav a').forEach(link => {
        link.addEventListener('click', (e) => {
            e.preventDefault();
            const target = document.getElementById(link.getAttribute('href').substring(1));
            if (target) target.scrollIntoView({ behavior: 'smooth' });
        });
    });

    const observer = new IntersectionObserver((entries) => {
        entries.forEach(entry => {
            if (entry.isIntersecting) {
                document.querySelectorAll('.sidebar__nav a').forEach(a => a.classList.remove('active'));
                const active = document.querySelector(`.sidebar__nav a[href="#${entry.target.id}"]`);
                if (active) active.classList.add('active');
            }
        });
    }, { rootMargin: '-10% 0px -80% 0px' });

    document.querySelectorAll('.main-content > section').forEach(s => observer.observe(s));
}

let pendingExportUrl = "";

function showExportModal(url, text) {
    pendingExportUrl = url;
    document.getElementById('modal-export-text').innerText = text;
    const modal = document.getElementById('modal-export');
    modal.style.display = 'flex';
    setTimeout(() => modal.classList.add('active'), 10);
}

function hideExportModal() {
    const modal = document.getElementById('modal-export');
    modal.classList.remove('active');
    setTimeout(() => modal.style.display = 'none', 200);
}

document.addEventListener('DOMContentLoaded', () => {
    initCharts();
    connectWS();
    refreshStatus();
    refreshLeaderboard();
    loadHistory();
    refreshComparison();

    document.getElementById('btn-start').addEventListener('click', startTraining);
    document.getElementById('btn-stop').addEventListener('click', stopTraining);
    
    document.getElementById('btn-export-model').addEventListener('click', () => {
        showExportModal('/api/export/model', 'Are you sure you want to export the latest aggregated global model weights in TFLite format? This file can be deployed to clients.');
    });
    document.getElementById('btn-export-report').addEventListener('click', () => {
        showExportModal('/api/export/report', 'Are you sure you want to download the training session metrics report in JSON format? This contains cumulative participant leaderboard data.');
    });

    document.getElementById('modal-cancel-btn').addEventListener('click', hideExportModal);
    document.getElementById('modal-confirm-btn').addEventListener('click', () => {
        if (pendingExportUrl) {
            window.open(pendingExportUrl, '_blank');
        }
        hideExportModal();
    });

    document.getElementById('btn-copy-code').addEventListener('click', () => {
        const code = document.getElementById('access-code').innerText;
        if (code && code !== '-') {
            navigator.clipboard.writeText(code);
            const btn = document.getElementById('btn-copy-code');
            btn.innerText = 'Copied!';
            setTimeout(() => btn.innerText = 'Copy', 2000);
        }
    });

    document.getElementById('btn-regenerate-code').addEventListener('click', async () => {
        try {
            const res = await fetch('/api/training/regenerate-token', { method: 'POST' });
            const data = await res.json();
            if (data.access_code) {
                document.getElementById('access-code').innerText = data.access_code;
                const btn = document.getElementById('btn-regenerate-code');
                btn.innerText = 'Regenerated!';
                setTimeout(() => btn.innerText = 'Regenerate', 2000);
                
                // Clear UI clients lists & stats instantly
                clients = {};
                renderClients();
                refreshStatus();
                refreshLeaderboard();
            }
        } catch (e) {}
    });

    document.getElementById('lan-ip').innerText = window.location.host;

    initNavigation();
    setInterval(refreshStatus, 5000);
});
