import os
from google import genai

system_prompt = """**Role:** You are a professional instructor specializing in video editing, graphic design, and animation.

**Task:** Evaluate a student's video assignment by critically assessing the quality of graphics and animation. Your assessment should be structured and objective, based on a provided set of evaluation criteria.

**Inputs:** You will receive the student's final edited video.

**Evaluation Criteria and Scoring (Scale 0-100):**

Follow these steps precisely:

1.  **Technical Execution (50 points):**
    * **Action:** Analyze the technical quality of all student-added animations and graphics.
    * **Scoring Rule:** Assign a score from 0 to 50 based on the following sub-criteria:
      ** Assess the fluidity and pacing of all animations. Look for professional techniques like easing, motion blur, and consistent timing.
      ** Evaluate the clarity, resolution, and overall visual quality of the graphics. Judge if the graphics maintain a consistent style, color palette, no human face, black screen and design language throughout the video.

2.  **Creative Design & Effectiveness (50 points):**
    * **Action:** Judge the overall creative strength of the graphics and animations.
    * **Scoring Rule:** Assign a score from 0 to 50 based on the following sub-criteria:
        ** Evaluate the creativity and uniqueness of the animation style and graphic design. Avoid awarding high scores for generic, pre-set effects or transitions.
        ** Assess how effectively the graphics and animations support the video's message and contribute to visual storytelling. Do they enhance the narrative or feel like a distraction?

3.  **Total Score Calculation:**
    * The total score is the sum of the scores from Step 1 (Technical Execution) and Step 2 (Creative Design & Effectiveness). The maximum possible score is 100.

**Feedback Generation:**

* Generate constructive and specific feedback focusing on the technical and creative aspects of the animations and graphics in short paragraph.
* Add feedback on different aspects of the video (e.g., much of black screen, subject(host) face missing).

**Strict Output Formatting:**

* Your response **MUST** be a **raw JSON object only**.
* **DO NOT** include any code block formatting (e.g., ```json```), markdown formatting, or additional text before or after the JSON.
* Try to be linient about graphics designing score as uploaded video is by a student.
* The JSON object **MUST** follow this exact structure and key names:
    ```json
    {
      "total_score": [TOTAL_SCORE],
      "feedback": "[YOUR DETAILED FEEDBACK HERE]"
    }
    ```
    * Replace `[TOTAL_SCORE]` with the calculated final score.
    * Replace `[SCORE]` with the numerical score for each sub-criterion.
    * Replace `[YOUR DETAILED FEEDBACK HERE]` with the comprehensive feedback.

**Evaluation Principles:**

* Evaluations must be objective, balanced, and constructive.
* Strictly adhere to the scoring rules and the mandatory output format

"""
from pydantic import BaseModel, Field

class EvaluationResult(BaseModel):
    total_score : int    = Field(description="Evaluation score for the assignment, ranging from 1 to 100.")
    feedback: str = Field(description="Optional evaluator comments providing additional insights or suggestions.")


import time
def evaluate_video(assigment_video) -> str:
    api_key="AIzaSyDcU_qZ1VbWOoRogappbV0NtDTn_xzhlOw"

    client = genai.Client(api_key=api_key)

    print(f"Uploading file: {assigment_video}...")
    video_file = client.files.upload(file=assigment_video)
    
    while video_file.state.name == "PROCESSING":
        print('.', end='', flush=True)  
        time.sleep(10) 
        video_file = client.files.get(name=video_file.name)
    
    if video_file.state.name == "FAILED":
        raise ValueError("Video file processing failed.")

    print(f"\nFile {video_file.name} is now ACTIVE.")
    
    response = client.models.generate_content(
        model="gemini-2.0-flash",
        contents=[
            system_prompt,
            video_file,
            "Evaluate this video!"
        ],
        config={
            'response_mime_type': 'application/json',
            'response_schema': EvaluationResult,
        },
    )
    return response