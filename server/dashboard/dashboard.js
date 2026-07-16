let ws;
let accChart, lossChart;
let accData = [], lossData = [], labels = [];
let clients = {};

const colorPrimary = '#D4845A';
const colorGold = '#E8B87A';
const colorText = '#3D2B1F';
const colorGrid = 'rgba(122, 99, 85, 0.1)';

function initCharts() {
    const commonOptions = {
        responsive: true,
        maintainAspectRatio: false,
        animation: { duration: 500, easing: 'easeOutQuart' },
        scales: {
            x: { grid: { color: colorGrid }, ticks: { color: colorText } },
            y: { grid: { color: colorGrid }, ticks: { color: colorText } }
        },
        plugins: {
            legend: { labels: { color: colorText, font: { family: 'Outfit' } } }
        }
    };

    const ctxAcc = document.getElementById('accChart').getContext('2d');
    accChart = new Chart(ctxAcc, {
        type: 'line',
        data: {
            labels: labels,
            datasets: [{
                label: 'Accuracy',
                data: accData,
                borderColor: colorPrimary,
                backgroundColor: 'rgba(212, 132, 90, 0.1)',
                fill: true,
                tension: 0.4
            }]
        },
        options: { ...commonOptions, scales: { ...commonOptions.scales, y: { ...commonOptions.scales.y, min: 0, max: 1 } } }
    });

    const ctxLoss = document.getElementById('lossChart').getContext('2d');
    lossChart = new Chart(ctxLoss, {
        type: 'line',
        data: {
            labels: labels,
            datasets: [{
                label: 'Loss',
                data: lossData,
                borderColor: colorGold,
                backgroundColor: 'rgba(232, 184, 122, 0.1)',
                fill: true,
                tension: 0.4
            }]
        },
        options: commonOptions
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
        setTimeout(connectWS, 2000);
    };
    
    ws.onmessage = (event) => {
        const msg = JSON.parse(event.data);
        handleWSEvent(msg.type, msg.data);
    };
}

function handleWSEvent(type, data) {
    if (type === 'round_started') {
        document.getElementById('phase-badge').innerText = `Training Round ${data.round}`;
        document.getElementById('phase-badge').style.color = 'var(--accent-primary)';
        document.getElementById('round-progress-label').innerText = `Round ${data.round} / ${data.total_rounds}`;
        document.getElementById('round-progress-fill').style.width = `${(data.round / data.total_rounds) * 100}%`;
    } 
    else if (type === 'round_completed') {
        // Update charts
        labels.push(data.round);
        accData.push(data.global_accuracy);
        lossData.push(data.global_loss);
        if (labels.length > 50) { labels.shift(); accData.shift(); lossData.shift(); }
        accChart.update();
        lossChart.update();
        refreshLeaderboard();
    }
    else if (type === 'training_complete') {
        document.getElementById('phase-badge').innerText = 'Complete';
        document.getElementById('phase-badge').style.color = 'var(--success)';
        document.getElementById('round-progress-fill').style.width = '100%';
    }
}

async function refreshStatus() {
    const res = await fetch('/api/training/status');
    const data = await res.json();
    document.getElementById('model-version').innerText = `Model v${data.model_version}`;
    document.getElementById('client-count').innerText = data.connected_clients;
}

async function refreshLeaderboard() {
    const res = await fetch('/api/metrics/leaderboard');
    const data = await res.json();
    const tbody = document.querySelector('#leaderboard-table tbody');
    tbody.innerHTML = '';
    
    data.leaderboard.forEach((client, index) => {
        const tr = document.createElement('tr');
        if(index === 0) tr.className = 'rank-1';
        tr.innerHTML = `
            <td>#${index + 1}</td>
            <td>${client.nickname || client.client_id.substring(0,8)}</td>
            <td>${client.images}</td>
            <td class="mono">${client.score}</td>
        `;
        tbody.appendChild(tr);
    });
}

function startTraining() {
    // In a real app this would call an API endpoint to trigger training start on server
    // For now, since FLCoordinator auto-starts or is triggered via API, we just alert.
    alert("Training initiation would happen here via REST call to server.");
}

// Uptime timer
let startTime = Date.now();
setInterval(() => {
    let diff = Math.floor((Date.now() - startTime) / 1000);
    let h = Math.floor(diff / 3600).toString().padStart(2, '0');
    let m = Math.floor((diff % 3600) / 60).toString().padStart(2, '0');
    let s = (diff % 60).toString().padStart(2, '0');
    document.getElementById('uptime').innerText = `${h}:${m}:${s}`;
}, 1000);

document.addEventListener('DOMContentLoaded', () => {
    initCharts();
    connectWS();
    refreshStatus();
    refreshLeaderboard();
    setInterval(refreshStatus, 5000);
});
