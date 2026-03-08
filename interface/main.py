from flask import Flask, request, jsonify
from flask_cors import CORS
from PIL import Image
from leaf_analyser import analyse_leaf
app = Flask(__name__)
CORS(app)

@app.route("/")
def home():
    return app.send_static_file("index.html")
@app.route("/api/analyze-leaf/", methods=["POST"])
def analyze_leaf():
    if "image" not in request.files:
        return jsonify({"error": "Campo 'image' não encontrado"}), 400

    image = request.files["image"]

    if image.filename == "":
        return jsonify({"error": "Nenhum arquivo selecionado"}), 400

    # Validate size (10MB)
    image.seek(0, 2)
    size = image.tell()
    image.seek(0)
    if size > 10 * 1024 * 1024:
        return jsonify({"error": "Imagem maior que 10MB"}), 400

    image = request.files["image"]

    img = Image.open(image.stream)

    result = analyse_leaf(img)

    return jsonify(result), 200

if __name__ == "__main__":
    app.run(debug=True)
