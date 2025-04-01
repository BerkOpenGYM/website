from flask import Flask, request, jsonify
from flask_cors import CORS
import os
from dotenv import load_dotenv
from google import generativeai as genai
from PIL import Image
import cv2
import tempfile

from multimodal_lm import analyze_posture  # you already have this

load_dotenv()
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

app = Flask(__name__)
CORS(app)  # Allow front-end to talk to backend

@app.route("/analyze", methods=["POST"])
def analyze():
    video = request.files.get("video")
    if not video:
        return jsonify({"error": "No video file uploaded"}), 400

    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
    tmp_path = tmp.name
    tmp.close()
    video.save(tmp_path)
    try:
        feedback = analyze_posture(tmp_path)
        return jsonify({"feedback": feedback})
    finally:
        try:
            os.remove(tmp_path)
        except PermissionError:
            print(f"Could not delete file: {tmp_path}")

    # with  as tmp:
    #     video.save(tmp.name)
    #     try:
    #         feedback = analyze_posture(tmp.name)
    #         return jsonify({"feedback": feedback})
    #     finally:
    #         os.remove(tmp.name)

if __name__ == "__main__":
    app.run(debug=True)
