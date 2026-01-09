document.getElementById("predictForm").addEventListener("submit", async function(e) {
    e.preventDefault();

    const age = document.getElementById("age").value;
    const bmi = document.getElementById("bmi").value;
    const smoker = document.getElementById("smoker").value;

    document.getElementById("result").innerText = "⏳ Mengirim data...";

    try {
        const response = await fetch("http://127.0.0.1:5000/api/predict", {
            method: "POST",
            headers: {
                "Content-Type": "application/json"
            },
            body: JSON.stringify({ age, bmi, smoker })
        });

        const data = await response.json();
        document.getElementById("result").innerText = `💡 Prediksi Biaya: ${data.prediction}`;
    } catch (error) {
        document.getElementById("result").innerText = "❌ Error memproses prediksi";
        console.error(error);
    }
});
