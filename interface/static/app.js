const UI = {
    fileInput: document.getElementById("fileInput"),
    btnCamera: document.getElementById("btnCamera"),
    btnChoose: document.getElementById("btnChoose"),
    preview: document.getElementById("previewImg"),
    placeholder: document.getElementById("uploadPlaceholder"),
    panel: document.getElementById("resultPanel"),
    disease: document.getElementById("resDisease"),
    conf: document.getElementById("resConfidence")
};

async function handleAnalysis(file) {
    UI.placeholder.innerText = "A PROCESSAR AMOSTRA...";
    UI.preview.style.display = "none";
    UI.panel.hidden = true;

    const fd = new FormData();
    fd.append("image", file);

    try {
        const resp = await fetch("/api/analyze-leaf/", { method: "POST", body: fd });
        const data = await resp.json();

        if (resp.ok) {
            UI.placeholder.style.display = "none";
            UI.preview.src = URL.createObjectURL(file);
            UI.preview.style.display = "block";
            
            UI.panel.hidden = false;
            UI.disease.innerText = data.disease;
            UI.conf.innerText = (data.confidence * 100).toFixed(1) + "%";
        }
    } catch (e) {
        UI.placeholder.innerText = "ERRO TÉCNICO";
    }
}

UI.btnChoose.onclick = () => UI.fileInput.click();
UI.btnCamera.onclick = () => {
    UI.fileInput.setAttribute("capture", "environment");
    UI.fileInput.click();
};

UI.fileInput.onchange = (e) => {
    if (e.target.files[0]) handleAnalysis(e.target.files[0]);
};