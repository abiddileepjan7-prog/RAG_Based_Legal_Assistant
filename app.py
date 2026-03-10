import time

from flask import Flask, render_template, request, jsonify, Response
import os

from rag_pipeline import generate_answer_stream, add_documents
 # ⭐ CONNECT MODEL

app = Flask(__name__)

DATA_FOLDER = "files"
os.makedirs(DATA_FOLDER, exist_ok=True)


@app.route("/")
def home():
    return render_template("index.html")


@app.route("/chat", methods=["POST"])
def chat():

    data = request.json
    message = data.get("message", "")

    def stream():
        for token in generate_answer_stream(message):
            yield token

    return Response(stream(), mimetype="text/plain")


# =========================
# UPLOAD FILES → files/
# =========================
@app.route("/upload", methods=["POST"])
def upload():

    files = request.files.getlist("files")

    saved_paths = []

    for file in files:
        if file.filename == "":
            continue

        path = os.path.join(DATA_FOLDER, file.filename)
        file.save(path)
        saved_paths.append(path)

    # ⭐ IMPORTANT — Update RAG Vector Store
    if saved_paths:
        add_documents(saved_paths)

    return jsonify({"status": "ok"})


# =========================
# LIST FILES
# =========================
@app.route("/files")
def list_files():

    files = os.listdir(DATA_FOLDER)
    return jsonify({"files": files})


# =========================
# DELETE FILE
# =========================
@app.route("/delete", methods=["POST"])
def delete_file():

    data = request.json
    filename = data.get("filename")

    path = os.path.join(DATA_FOLDER, filename)

    if os.path.exists(path):
        os.remove(path)

    return jsonify({"status": "deleted"})


# =========================
# RUN
# =========================
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5001, debug=False, use_reloader=False)

