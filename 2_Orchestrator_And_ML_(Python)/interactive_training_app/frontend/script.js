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

        // Capture a single image from the primary webcam
        canvasEl.width = webcamEl.videoWidth;
        canvasEl.height = webcamEl.videoHeight;
        canvasEl.getContext('2d').drawImage(webcamEl, 0, 0, canvasEl.width, canvasEl.height);

        canvasEl.toBlob(async (blob) => {
            // --- KEY CHANGE: Use FormData to send multiple files ---
            const formData = new FormData();

            // Append the same image twice under different keys to simulate two cameras
            formData.append('image_left', blob, 'capture_left.jpg');
            formData.append('image_right', blob, 'capture_right.jpg');

            // In the future, you could add the Kinect depth map here:
            // formData.append('depth_map', depthBlob, 'depth.bin');

            // Append all other scenario data as form fields
            formData.append('Investor_Action', action);
            for (const key in currentScenario) {
                formData.append(key, currentScenario[key]);
            }

            // Send data to the new backend endpoint on the Pi
            statusMessageEl.textContent = `Logging decision: ${action}...`;
            try {
                // Note: The Pi server endpoint that receives this needs to be updated
                // to call the new /analyze_vision endpoint on the PC service.
                await fetch(`${API_BASE_URL}/log_decision`, { // Assuming you update pi_server.py
                    method: 'POST',
                    body: formData,
                });

                fetchNextScenario(); // Load next scenario on success
            } catch (error) {
                console.error('Error logging decision:', error);
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

