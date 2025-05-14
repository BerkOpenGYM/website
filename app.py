from flask import Flask, request, jsonify
from flask_cors import CORS
import os
from dotenv import load_dotenv
from google import genai
from PIL import Image
import cv2
import tempfile
from openai import OpenAI
from pydantic import BaseModel

from multimodal_lm import analyze_posture  # you already have this

load_dotenv()
client = genai.Client(api_key=os.getenv("GOOGLE_API_KEY"))
client = OpenAI(
    api_key=os.getenv('OPENAI_API_KEY')
)

shared_memory = {"workout_plan": [], "goal": [], "suggestions": []}

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

class Workout(BaseModel):
    name: str
    reptition: int
    num_set: int
    rest_sec: int
    equipments: str

class WorkoutPlan(BaseModel):
    plan: list[Workout]

# Initialize Flask app and enable CORS
app = Flask(__name__)
CORS(app)  # allow all origins; adjust later for more restrictive policies

@app.route('/plan', methods=['POST'])
def plan_workout():
    data = request.get_json()
    height = data.get('height')
    weight = data.get('weight')
    experience = data.get('experience')
    goal = data.get('goal')

    system_msg = "You are a helpful workout planning assistant."
    user_msg = (
        f"User profile: height {height} cm, weight {weight} kg, experience level {experience}. Goal: {goal}. "
        "Generate a weekly workout plan in JSON format as an array of exercises. "
        "Each exercise object should have keys: 'name', 'sets', 'reps', 'rest_sec'."
        f"Here are the learning lession and previous workout history {shared_memory}"
    )

    # Call the OpenAI Chat Completions endpoint
    response = client.beta.chat.completions.parse(
        model="gpt-4o",
        messages=[
            {"role": "system", "content": system_msg},
            {"role": "user", "content": user_msg}
        ],
        response_format=WorkoutPlan
    )
    content = response.choices[0].message.parsed.model_dump_json()
    shared_memory['workout_plan'].append(content)
    shared_memory['goal'].append(goal)
    return content

class Suggestion(BaseModel):
    title: str
    type: str
    description: str
    link: str

class LearningResponse(BaseModel):
    suggestions: list[Suggestion]

# add this endpoint below your /plan route
@app.route("/learn", methods=["POST"])
def learn():
    payload = request.get_json() or {}
    print(payload)
    challenge = payload.get("challenge", "")
    emotion  = payload.get("emotion", "")
    system_msg = (
        "You are a fitness learning agent that identifies training plateaus "
        "and recommends human-led resources or courses to overcome them."
    )
    user_msg = f"User challenge: {challenge}."
    if emotion:
        user_msg += f" Emotion: {emotion}."
    user_msg += (
        " Provide your recommendations as JSON with key 'suggestions': "
        "an array of objects with 'title', 'type', 'description', and 'link'."
    )
    # call OpenAI (using whatever `client` you already have)
    response = client.beta.chat.completions.parse(
        model="gpt-4o",
        messages=[
            {"role": "system", "content": system_msg},
            {"role": "user", "content": user_msg}
        ],
        response_format=LearningResponse
    )
    content = response.choices[0].message.parsed.model_dump_json()
    shared_memory['suggestions'].append(content)
    return content


if __name__ == "__main__":
    app.run(debug=True)
