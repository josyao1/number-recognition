const canvas = document.getElementById('canvas');
const ctx = canvas.getContext('2d');
ctx.fillStyle = 'black';
ctx.fillRect(0, 0, canvas.width, canvas.height);
ctx.strokeStyle = 'white';
ctx.lineWidth = 10;

let drawing = false;

canvas.addEventListener('mousedown', () => {
    drawing = true;
});

canvas.addEventListener('mouseup', () => {
    drawing = false;
    ctx.beginPath();
});

canvas.addEventListener('mousemove', draw);

function draw(event) {
    if (!drawing) return;

    ctx.lineCap = 'round';
    ctx.lineTo(event.clientX - canvas.offsetLeft, event.clientY - canvas.offsetTop);
    ctx.stroke();
    ctx.beginPath();
    ctx.moveTo(event.clientX - canvas.offsetLeft, event.clientY - canvas.offsetTop);
}

function clearCanvas() {
    ctx.fillRect(0, 0, canvas.width, canvas.height);
}

async function predictDigit() {
    console.log("Predict Digit button clicked");

    const image = canvas.toDataURL();
    console.log("Image data URL: ", image);

    try {
        const response = await fetch('/predict', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify({ image: image }),
        });
        const data = await response.json();
        document.getElementById('result').innerText = `Predicted Digit: ${data.prediction}`;
    } catch (error) {
        console.error('Error:', error);
    }
}
