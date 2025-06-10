document.addEventListener('DOMContentLoaded', () => {
    // --- CONFIGURATION ---
    const API_BASE_URL = 'http://YOUR_RASPBERRY_PI_IP:8000'; // Replace with your Pi's IP

    // --- DOM ELEMENTS ---
    const titleEl = document.getElementById('scenario-title');
    const priceEl = document.getElementById('price');
    const analystRatioEl = document.getElementById('analyst-ratio');
    const newsEl = document.getElementById('news');
    const statusMessageEl = document.getElementById('status-message');
    const webcamEl = document.getElementById('webcam');
    const canvasEl = document.getElementById('canvas');
    const buyBtn = document.getElementById('buy-btn');
    const holdBtn = document.getElementById('hold-btn');
    const sellBtn = document.getElementById('sell-btn');

    let currentScenario = null;

    // --- WEBCAM INITIALIZATION ---
    async function setupWebcam() {
        try {
            const stream = await navigator.mediaDevices.getUserMedia({ video: true });
            webcamEl.srcObject = stream;
        } catch (error) {
            statusMessageEl.textContent = "Webcam access is required for facial analysis.";
        }
    }

    // --- CORE FUNCTIONS ---
    async function fetchNextScenario() {
        statusMessageEl.textContent = 'Loading next scenario...';
        try {
            const response = await fetch(`${API_BASE_URL}/get_next_scenario`);
            if (!response.ok) {
                document.querySelector('.simulator-container').innerHTML = "<h1>All Done!</h1><p>Thank you. The training dataset has been created.</p>";
                return;
            }
            currentScenario = await response.json();
            updateUI(currentScenario);
        } catch (error) {
            statusMessageEl.textContent = 'Error connecting to the backend server.';
        }
    }

    function updateUI(scenario) {
        titleEl.textContent = `${scenario.Symbol} on ${new Date(scenario.Date).toLocaleDateString()}`;
        priceEl.textContent = parseFloat(scenario.Close_Price).toFixed(2);
        analystRatioEl.textContent = (parseFloat(scenario.Analyst_Buy_Ratio) * 100).toFixed(1);
        newsEl.textContent = scenario.News_Headline;
        statusMessageEl.textContent = 'Please make a decision.';
    }

    async function logDecision(action) {
        if (!currentScenario) return;

        canvasEl.width = webcamEl.videoWidth;
        canvasEl.height = webcamEl.videoHeight;
        canvasEl.getContext('2d').drawImage(webcamEl, 0, 0, canvasEl.width, canvasEl.height);

        canvasEl.toBlob(async (blob) => {
            const formData = new FormData();
            formData.append('image', blob, 'capture.jpg');
            formData.append('Investor_Action', action);
            for (const key in currentScenario) {
                formData.append(key, currentScenario[key]);
            }

            statusMessageEl.textContent = `Logging decision: ${action}...`;
            try {
                await fetch(`${API_BASE_URL}/log_decision`, { method: 'POST', body: formData });
                fetchNextScenario();
            } catch (error) {
                statusMessageEl.textContent = 'Failed to log decision.';
            }
        }, 'image/jpeg');
    }

    // --- EVENT LISTENERS & INITIALIZATION ---
    buyBtn.addEventListener('click', () => logDecision('BUY'));
    holdBtn.addEventListener('click', () => logDecision('HOLD'));
    sellBtn.addEventListener('click', () => logDecision('SELL'));

    setupWebcam();
    fetchNextScenario();
});