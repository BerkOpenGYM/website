# OpenGYM

## Project Overview
OpenGYM is an AI-powered fitness assistant platform that leverages Large Language Models (LLMs) and multimodal capabilities to provide personalized workout planning, posture analysis, and learning recommendations. By integrating external tools and APIs with advanced language models, OpenGYM extends the capabilities of AI to deliver practical fitness guidance and support.

## Motivation
Traditional fitness applications often lack personalization and real-time feedback. OpenGYM addresses this gap by utilizing LLMs' function-calling capabilities to interact with specialized tools for specific fitness tasks. This approach enables more dynamic, context-aware assistance that can adapt to individual needs and goals.

## Key Features

### 1. Feedback Agent
- Upload gym videos for real-time posture analysis
- Receive personalized feedback on exercise form
- AI-powered detection of potential injury risks

### 2. Workout Builder/Planning Agent
- Generate personalized workout plans based on:
  - Height and weight
  - Experience level
  - Fitness goals
- Detailed exercise recommendations with sets, reps, and rest periods
- Equipment-specific suggestions

### 3. Learning Agent
- Overcome fitness plateaus with targeted recommendations
- Personalized resources based on specific challenges
- Emotional context-aware suggestions
- Direct links to external learning resources

## Technologies Used

### Frontend
- HTML/CSS/JavaScript
- TailwindCSS for styling
- Font Awesome for icons

### Backend
- Python/Flask server
- Flask-CORS for cross-origin requests
- OpenAI API (GPT-4o)
- Google Gemini API
- MediaPipe for pose detection and analysis

## Setup and Installation

### Prerequisites
- Python 3.10+
- API keys for OpenAI and Google Gemini

### Environment Setup
1. Clone the repository
2. Create a Python virtual environment:
   ```bash
   conda env create -f environment.yml
   ```
3. Copy the `.env.example` file to `.env` and add your API keys:
   ```
   OPENAI_API_KEY=your_openai_key_here
   GOOGLE_API_KEY=your_google_key_here
   ```

### Running the Application
1. Start the Flask backend:
   ```bash
   python app.py
   ```
2. Open any of the HTML files in your browser or serve them using a static file server

## Usage

### Main Dashboard
- Navigate to `index.html` to access the main dashboard
- Select from available agents based on your fitness needs

### Analyzing Your Form
1. Go to the Feedback Agent (mlm.html)
2. Upload a video of your workout
3. Receive AI-powered feedback on your posture and form

### Creating a Workout Plan
1. Go to the Workout Builder (planning.html)
2. Enter your height, weight, experience level, and goals
3. Get a personalized workout plan with detailed exercise instructions

### Overcoming Plateaus
1. Go to the Learning Agent (learning.html)
2. Describe your current fitness challenge
3. Indicate how you feel about the challenge
4. Receive tailored recommendations and resources

## Project Structure
- `app.py` - Main Flask application with API endpoints
- `planning_agent.py` - Workout planning functionality
- `multimodal_lm.py` - Video analysis and posture feedback
- `pose_tools.py` - Utilities for pose detection
- HTML files for each component interface
- `.env` - Environment variables (API keys)

## Security Notes
- API keys are stored in the `.env` file which should never be committed to version control
- The `.gitignore` file is configured to exclude the `.env` file

## Future Enhancements
- User accounts and progress tracking
- Integration with wearable fitness devices
- Enhanced video analysis with more detailed feedback
- Mobile application version
