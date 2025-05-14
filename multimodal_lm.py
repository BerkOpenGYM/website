import cv2
import mediapipe as mp
from google import genai
from PIL import Image
import os
from dotenv import load_dotenv
from pose_tools import draw_pose

FPS = 30
video_path = "out/output.mp4"
model_name = "gemini-1.5-flash"

load_dotenv()
client = genai.Client(api_key='GEMINI_API_KEY') # use environment variable

def extract_key_frames(video_path, max_frames=3):
    cap = cv2.VideoCapture(video_path)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    interval = max(1, total // max_frames)
    frames = []

    for i in range(max_frames):
        cap.set(cv2.CAP_PROP_POS_FRAMES, i * interval)
        success, frame = cap.read()
        if success:
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            pil_image = Image.fromarray(rgb)
            frames.append(pil_image)
    cap.release()
    return frames

def analyze_posture(video_path):
    prompt = """
    You are given a series of frames from a gym exercise with pose detection overlaid. 
    Provide feedback on the person's gym posture and any corrections you would suggest.
    """
    
    if os.path.getsize(video_path) > 20 * 1024 * 1024:
        raise ValueError("Video size exceeds 20MB limit.")
    
    frames = extract_key_frames(video_path)
    model = client.models(model_name)

    response = model.generate_content([
        *frames,
        prompt
    ])
    return response.text


if __name__ == "__main__":
    # 🎥 Capture and annotate video
    cap = cv2.VideoCapture(0)
    width = 640
    height = 480
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(video_path, fourcc, FPS, (width, height))

    while cap.isOpened():
        success, image = cap.read()
        if not success:
            continue

        image, _ = draw_pose(image)
        out.write(image)
        cv2.imshow('Video', image)
        
        if cv2.waitKey(1) & 0xFF == 27:  # ESC to stop
            break
        
    cap.release()
    out.release()
    cv2.destroyAllWindows()

    # 🧠 Analyze posture
    print("Predicted answer:", analyze_posture(video_path))
