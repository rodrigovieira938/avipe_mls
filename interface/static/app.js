const dropzone = document.getElementById("dropzone");
const fileInput = document.getElementById("fileInput");
const btnChoose = document.getElementById("btnChoose");
const btnCamera = document.getElementById("btnCamera");

const previewImg = document.getElementById("previewImg");
const previewMeta = document.getElementById("previewMeta");

const statAnalises = document.getElementById("statAnalises");

let analysesCount = 0;

function isValidFile(file) {
  const okType = ["image/jpeg", "image/png"].includes(file.type);
  const okSize = file.size <= 10 * 1024 * 1024; // 10MB
  return okType && okSize;
}

function setPreview(file) {
  const url = URL.createObjectURL(file);
  previewImg.src = url;
  previewImg.style.display = "block";
  previewMeta.textContent = `${file.name} • ${(file.size / 1024 / 1024).toFixed(2)}MB`;
}

async function analyzeLeaf(file) {
  const form = new FormData();
  form.append("image", file);

  const endpoint = "/api/analyze-leaf/";

  try {
    const res = await fetch(endpoint, { method: "POST", body: form });
    const data = await res.json().catch(() => null);

    if (!res.ok) {
      const msg = data?.error || data?.detail || `Erro HTTP ${res.status}`;
      throw new Error(msg);
    }
    return data;
  } catch (err) {
    return { status: "error", error: err.message || "Falha na requisição" };
  }
}

async function handleFile(file) {
  if (!isValidFile(file)) {
    alert("Arquivo inválido. Envie PNG/JPG até 10MB.");
    return;
  }

  setPreview(file);
  previewMeta.textContent += " • a analisar...";

  const result = await analyzeLeaf(file);
  console.log("Análise concluída:", result);
  if (result.status === "error") {
    previewMeta.textContent =
      `${file.name} • ${(file.size / 1024 / 1024).toFixed(2)}MB • ERRO: ${result.error}`;
    return;
  }

  analysesCount += 1;
  statAnalises.textContent = analysesCount;

  previewMeta.textContent =
    `${file.name} • ${(file.size / 1024 / 1024).toFixed(2)}MB • Resultado: ${result.disease} • ${(result.confidence * 100).toFixed(0)}%`;
}

/* Clique no dropzone abre o seletor */
dropzone.addEventListener("click", () => fileInput.click());

btnChoose.addEventListener("click", (e) => {
  e.stopPropagation();
  fileInput.click();
});

/* Input file */
fileInput.addEventListener("change", (e) => {
  const file = e.target.files?.[0];
  if (file) handleFile(file);
});

/* Drag & Drop */
["dragenter", "dragover"].forEach(evt => {
  dropzone.addEventListener(evt, (e) => {
    e.preventDefault();
    e.stopPropagation();
    dropzone.classList.add("dragover");
  });
});

["dragleave", "drop"].forEach(evt => {
  dropzone.addEventListener(evt, (e) => {
    e.preventDefault();
    e.stopPropagation();
    dropzone.classList.remove("dragover");
  });
});

dropzone.addEventListener("drop", (e) => {
  const file = e.dataTransfer.files?.[0];
  if (file) handleFile(file);
});

/* Capturar foto (mobile) */
btnCamera.addEventListener("click", (e) => {
  e.stopPropagation();
  fileInput.setAttribute("capture", "environment");
  fileInput.click();
  setTimeout(() => fileInput.removeAttribute("capture"), 500);
});
