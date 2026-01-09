console.log("History JS Loaded");

fetch("/api/predict/history")
  .then(res => res.json())
  .then(data => {
    const table = document.getElementById("history-table");
    table.innerHTML = "";

    if (data.length === 0) {
      table.innerHTML = `<tr><td colspan="5" style="text-align:center;">Belum ada data riwayat.</td></tr>`;
      return;
    }

    function formatRupiah(value) {
    if (!value || isNaN(value)) return value;
    return new Intl.NumberFormat("id-ID", {
        style: "currency",
        currency: "IDR",
        minimumFractionDigits: 0
    }).format(value);
}


    data.forEach(row => {
      table.innerHTML += `
        <tr>
          <td>${row.timestamp}</td>
          <td>${row.age}</td>
          <td>${row.bmi}</td>
          <td>${row.smoker}</td>
          <td><b>${row.prediction}</b></td>
        </tr>
      `;
    });
  })
  .catch(err => {
    console.error(err);
    document.getElementById("history-table").innerHTML = 
        `<tr><td colspan="5" style="text-align:center;color:red;">Gagal memuat data!</td></tr>`;
  });
