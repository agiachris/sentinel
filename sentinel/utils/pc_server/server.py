from flask import Flask, jsonify, render_template, send_from_directory
import os
import json

app = Flask(__name__)

UPLOAD_FOLDER = os.path.join(os.getcwd(), "uploads")


@app.route("/data", methods=["GET"])
def data():
    try:
        with open(f"{UPLOAD_FOLDER}/point_cloud.json", "r") as pc_file:
            point_cloud = json.load(pc_file)

        with open(f"{UPLOAD_FOLDER}/elapsed_time.txt", "r") as et_file:
            elapsed_time = float(et_file.read())

        with open(f"{UPLOAD_FOLDER}/process_elapsed_time.txt", "r") as et_file:
            process_elapsed_time = float(et_file.read())

        data = {
            "pointCloud": point_cloud,
            "image": "image.jpg",
            "mask": "mask.png",
            "elapsedTime": elapsed_time,
            "processElapsedTime": process_elapsed_time,
        }

        return jsonify(data)
    except Exception as e:
        return jsonify(error=str(e)), 500


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/uploads/<filename>")
def uploaded_file(filename):
    return send_from_directory(UPLOAD_FOLDER, filename)


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8000, debug=True)
