document.addEventListener('DOMContentLoaded', () => {
    const video = document.getElementById('webcam');
    const canvas = document.getElementById('canvas');
    const ctx = canvas.getContext('2d');
    const predictionDiv = document.getElementById('prediction');
    const startBtn = document.getElementById('startBtn');
    const stopBtn = document.getElementById('stopBtn');
    let stream = null;
    let detectionInterval = null;
    
    // Colors for different predictions
    const predictionColors = {
        'No Mask': '#FF5252',
        'Mask': '#4CAF50',
        'Incorrect Mask': '#FFC107'
    };
    
    // Start webcam
    startBtn.addEventListener('click', async () => {
        try {
            stream = await navigator.mediaDevices.getUserMedia({ video: true });
            video.srcObject = stream;
            startBtn.disabled = true;
            stopBtn.disabled = false;
            
            // Start detection every 500ms
            detectionInterval = setInterval(detectMask, 500);
        } catch (err) {
            console.error("Error accessing webcam:", err);
            predictionDiv.textContent = "Could not access webcam";
        }
    });
    
    // Stop webcam
    stopBtn.addEventListener('click', () => {
        if (stream) {
            stream.getTracks().forEach(track => track.stop());
            clearInterval(detectionInterval);
            startBtn.disabled = false;
            stopBtn.disabled = true;
            predictionDiv.textContent = "";
            ctx.clearRect(0, 0, canvas.width, canvas.height);
        }
    });
    
    // Mask detection function
    async function detectMask() {
        // Draw current frame to canvas
        ctx.drawImage(video, 0, 0, canvas.width, canvas.height);
        
        // Get image data
        const imageData = canvas.toDataURL('image/jpeg');
        
        try {
            // Send to server for prediction
            const response = await fetch('/predict_webcam', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify({ image: imageData })
            });
            
            const result = await response.json();
            
            if (result.success) {
                // Display prediction
                predictionDiv.textContent = result.label;
                predictionDiv.style.backgroundColor = predictionColors[result.label] || '#333';
                
                // Draw rectangle around face (if coordinates are returned)
                if (result.face_box) {
                    ctx.clearRect(0, 0, canvas.width, canvas.height);
                    ctx.drawImage(video, 0, 0, canvas.width, canvas.height);
                    ctx.strokeStyle = '#FF0000';
                    ctx.lineWidth = 2;
                    ctx.strokeRect(result.face_box.x * canvas.width, result.face_box.y * canvas.height, result.face_box.width * canvas.width, result.face_box.height * canvas.height);
                }
            }
        } catch (err) {
            console.error("Error predicting:", err);
        }
    }
});
