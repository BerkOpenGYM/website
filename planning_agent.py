from flask import Flask, request, jsonify
from flask_cors import CORS  # add CORS support
import os
import json
from openai import OpenAI  # updated client import
from dotenv import load_dotenv
from pydantic import BaseModel
from openai import OpenAI


class Workout(BaseModel):
    name: str
    repetition: int
    num_set: int
    rest_sec: int
    equipments: str

class WorkoutPlan(BaseModel):
    plan: list[Workout]

# Load environment variables
load_dotenv()

# Initialize OpenAI client
client = OpenAI(
    api_key=os.getenv('OPENAI_API_KEY')
)

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
    return content

if __name__ == '__main__':
    app.run(debug=True)