document.getElementById('uploadForm').addEventListener('submit', async (e) => {
    e.preventDefault();

    const imageInput = document.getElementById('imageInput');
    const resultDiv = document.getElementById('result');

    if (imageInput.files.length === 0) {
        resultDiv.textContent = 'Please select an image first.';
        return;
    }

    const formData = new FormData();
    formData.append('image', imageInput.files[0]);

    try {
        resultDiv.textContent = 'Processing...';

        const response = await fetch('http://localhost:5000/predict', {
            method: 'POST',
            body: formData
        });

        if (!response.ok) {
            throw new Error('Prediction failed.');
        }

        const data = await response.json();
        resultDiv.textContent = `Prediction: ${data.prediction}`;
    } catch (error) {
        resultDiv.textContent = `Error: ${error.message}`;
    }
});
