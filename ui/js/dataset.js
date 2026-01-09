let dataset = [];
let filteredData = [];
let currentPage = 1;
const rowsPerPage = 10;


// =====================
// LOAD DATA
// =====================


async function loadDataset() {
    const response = await fetch("http://127.0.0.1:5000/api/dataset");
    dataset = await response.json();

    filteredData = dataset;
    currentPage = 1;

    renderTable();
}


// =====================
// RENDER TABLE + PAGINATION
// =====================
function formatRupiah(value) {
    if (!value || isNaN(value)) return value;
    return new Intl.NumberFormat("id-ID", {
        style: "currency",
        currency: "IDR",
        minimumFractionDigits: 0
    }).format(value);
}

function renderTable() {
    const head = document.getElementById("dataset-head");
    const body = document.getElementById("dataset-body");
    const pagination = document.getElementById("pagination");

    if (filteredData.length === 0) {
        head.innerHTML = "";
        body.innerHTML = "<tr><td colspan='99'>Tidak ada data</td></tr>";
        pagination.innerHTML = "";
        return;
    }

    // Render header (sekali)
    head.innerHTML = `
        <tr>
            ${Object.keys(filteredData[0]).map(col => `<th>${col}</th>`).join("")}
            <th>Aksi</th>
        </tr>
    `;

    // Pagination logic
    const start = (currentPage - 1) * rowsPerPage;
    const paginatedData = filteredData.slice(start, start + rowsPerPage);

    // Render body rows
    body.innerHTML = paginatedData.map((row, index) => `
        <tr>
            ${Object.keys(row).map(key => {
    const value = row[key];

    // format jika kolom adalah biaya / cost / charges
    if (["charges", "premium_cost", "predicted_cost"].includes(key.toLowerCase())) {
        return `<td>${formatRupiah(value)}</td>`;
    }

    return `<td>${value}</td>`;
}).join("")}
            <td><button onclick="useForPredict(${start + index})">🔍 Gunakan</button></td>
        </tr>
    `).join("");

    renderPagination();
}


// =====================
// PAGINATION BUTTONS
// =====================
function renderPagination() {
    const pagination = document.getElementById("pagination");
    const totalPages = Math.ceil(filteredData.length / rowsPerPage);

    let html = `Page ${currentPage} of ${totalPages} &nbsp;`;

    if (currentPage > 1) {
        html += `<button onclick="changePage(${currentPage - 1})">⬅ Prev</button>`;
    }

    if (currentPage < totalPages) {
        html += ` <button onclick="changePage(${currentPage + 1})">Next ➡</button>`;
    }

    pagination.innerHTML = html;
}

function changePage(page) {
    currentPage = page;
    renderTable();
}


// =====================
// SEARCH / FILTER DATASET
// =====================
function filterData() {
    const query = document.getElementById("search").value.toLowerCase();

    filteredData = dataset.filter(row =>
        Object.values(row).some(val =>
            String(val).toLowerCase().includes(query)
        )
    );

    currentPage = 1;
    renderTable();
}


// =====================
// UPLOAD CSV
// =====================
async function uploadFile() {
    const fileInput = document.getElementById("fileInput");
    const file = fileInput.files[0];

    if (!file) return alert("Pilih file CSV dahulu!");

    const formData = new FormData();
    formData.append("file", file);

    const response = await fetch("http://127.0.0.1:5000/api/dataset/upload", {
        method: "POST",
        body: formData
    });

    const result = await response.json();
    alert("Dataset berhasil diupload!");

    loadDataset();
}


// =====================
// EXPORT DATASET
// =====================
function downloadCSV() {
    if (!dataset.length) return alert("Tidak ada data untuk diexport");

    const csvContent = [
        Object.keys(dataset[0]).join(","),
        ...dataset.map(row => Object.values(row).join(","))
    ].join("\n");

    const blob = new Blob([csvContent], { type: "text/csv" });
    const url = URL.createObjectURL(blob);

    const a = document.createElement("a");
    a.href = url;
    a.download = "dataset_export.csv";
    a.click();
}


// =====================
// USE ROW FOR PREDICTION
// =====================
function useForPredict(index) {
    localStorage.setItem("dataset_index", index);
    window.location.href = "/predict";
}


// =====================
loadDataset();