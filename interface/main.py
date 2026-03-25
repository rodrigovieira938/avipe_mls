from flask import Flask, request, jsonify
from flask_cors import CORS
from PIL import Image
from leaf_analyser import analyse_leaf
import io
import os
import glob

app = Flask(__name__, static_folder='static')
CORS(app)

def cleanup_diagnostics():
    """Limpa ficheiros temporários de debug."""
    for f in glob.glob("debug_*.png"):
        try:
            os.remove(f)
        except:
            pass

@app.route("/")
def index():
    return app.send_static_file("index.html")

@app.route("/api/analyze-leaf/", methods=["POST"])
def analyze():
    if "image" not in request.files:
        return jsonify({"error": "Sem imagem"}), 400

    file = request.files["image"]
    
    try:
        img = Image.open(io.BytesIO(file.read()))
        if img.mode != 'RGB':
            img = img.convert('RGB')
            
        result = analyse_leaf(img)
        
        # Limpeza obrigatória
        cleanup_diagnostics()
        
        return jsonify(result), 200

    except Exception as e:
        cleanup_diagnostics()
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(debug=True, host='0.0.0.0', port=5000)