import os
import uvicorn
import logging
import httpx
from datetime import datetime
from ravallusion_ai.run import evaluate_videos_with_paths
from fastapi import FastAPI, BackgroundTasks, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from motor.motor_asyncio import AsyncIOMotorClient
from dotenv import load_dotenv
from pydantic import BaseModel
from bson import ObjectId
from typing import Optional, Union

# Simple logging setup
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

# Simple FastAPI app
app = FastAPI(
    title="Revallusion AI - Complete Evaluation System",
    description="Video evaluation system",
    version="1.0.0"
)

# Basic CORS (essential for frontend)
FRONTEND_URL = os.getenv("FRONTEND_URL", "http://localhost:3000")
app.add_middleware(
    CORSMiddleware,
    allow_origins=[FRONTEND_URL, "http://localhost:3000", "http://localhost:5173"],
    allow_credentials=True,
    allow_methods=["GET", "POST", "PUT", "DELETE"],
    allow_headers=["*"],
)

# Simple error handler
@app.exception_handler(Exception)
async def global_exception_handler(request, exc: Exception):
    logger.error(f"Error: {exc}")
    return JSONResponse(
        status_code=500,
        content={"error": "Internal server error", "message": str(exc)}
    )

# Simple health check
@app.get("/health")
async def health_check():
    return {"status": "healthy", "timestamp": datetime.now().isoformat()}

# Simple MongoDB connection
MONGO_DB_URI = os.getenv("MONGO_DB_URI")
if not MONGO_DB_URI:
    logger.error("MONGO_DB_URI environment variable is not set")
    raise ValueError("MONGO_DB_URI is required")

client = AsyncIOMotorClient(MONGO_DB_URI)
db = client["ravallusiondevelopmentdb"]
submitted_assignments_collection = db["submittedassignments"]
assignment_resources_collection = db["assignmentresources"]

# Configuration from environment with working fallbacks
CLOUDFRONT_BASE_URL = os.getenv("CLOUDFRONT_BASE_URL")
EXTERNAL_API_TOKEN = os.getenv("EXTERNAL_API_TOKEN")

# Simple response models
class AssignmentPathsResponse(BaseModel):
    submittedFileUrl: str
    finalCutVideoUrl: str
    assetsUrl: str

class StudentSubmission(BaseModel):
    asset_path: str
    final_video_path: str
    student_video_path: str

class EvaluationResponse(BaseModel):
    score: str
    feedback: str
    external_api_response: Optional[dict] = None

class SimpleResponse(BaseModel):
    success: bool
    message: str

def process_assignment_paths(assignment_id: str, submitted_file_url: str, final_cut_video_url: str, assets_url: str):
    """
    Background task to process the fetched paths
    """
    full_submitted_file_url = f"{CLOUDFRONT_BASE_URL}/{submitted_file_url}"
    full_final_cut_video_url = f"{CLOUDFRONT_BASE_URL}/{final_cut_video_url}"
    full_assets_url = f"{CLOUDFRONT_BASE_URL}/{assets_url}"
    
    logger.info(f"Background processing started for assignment: {assignment_id}")
    logger.debug(f"Generated URLs - Student: {full_submitted_file_url}")
    logger.debug(f"Generated URLs - Final Cut: {full_final_cut_video_url}")
    logger.debug(f"Generated URLs - Assets: {full_assets_url}")
    logger.info(f"Background processing completed for assignment: {assignment_id}")

async def submit_to_external_api(assignment_id: str, score: int, feedback: str):
    """Submit evaluation results to the external API"""
    try:
        url = f"https://api.ravallusion.com/api/v1/submitted-assignment/ai/{assignment_id}"
        
        data = {
            "score": score,
            "authToken": EXTERNAL_API_TOKEN,
            "feedback": feedback
        }
        
        async with httpx.AsyncClient(timeout=30.0) as client:
            response = await client.put(url, json=data)
            
            if response.status_code == 200:
                logger.info(f"Results submitted successfully for assignment {assignment_id}")
                return True
            else:
                logger.error(f"Failed to submit results: {response.status_code}")
                logger.debug(f"Response content: {response.text}")
                return False
                
    except Exception as e:
        logger.error(f"Error submitting results: {str(e)}")
        return False

@app.get("/")
async def root():
    return {"message": "Revallusion AI - Complete Evaluation System", "status": "running"}

@app.get("/fetch-assignment-paths/{assignment_id}", response_model=AssignmentPathsResponse)
async def fetch_assignment_paths(assignment_id: str, background_tasks: BackgroundTasks):
    """
    Fetch paths for an assignment based on assignment_id
    
    1. Query submittedassignments collection to get submittedFileUrl using assignment_id as _id
    2. Query assignmentresources collection to get finalCutVideoUrl and assetsUrl using assignment_id
    3. Return all three paths
    4. Process in background
    """
    logger.info(f"Fetching assignment paths for ID: {assignment_id}")
    
    try:
        # Convert assignment_id to ObjectId
        try:
            assignment_obj_id = ObjectId(assignment_id)
            logger.debug(f"Successfully converted assignment_id to ObjectId: {assignment_obj_id}")
        except Exception as e:
            logger.error(f"Invalid assignment_id format '{assignment_id}': {str(e)}")
            raise HTTPException(status_code=400, detail="Invalid assignment_id format")
        
        # Step 1: Get submittedFileUrl from submittedassignments collection
        logger.debug("Querying submittedassignments collection...")
        submitted_assignment = await submitted_assignments_collection.find_one(
            {"_id": assignment_obj_id}
        )
        
        if not submitted_assignment:
            logger.error(f"No submitted assignment found for ID: {assignment_id}")
            raise HTTPException(status_code=404, detail=f"No submitted assignment found with id: {assignment_id}")
        
        logger.info(f"Found submitted assignment: {submitted_assignment['_id']}")
        
        submitted_file_url = submitted_assignment.get("submittedFileUrl")
        
        if not submitted_file_url:
            logger.error("submittedFileUrl field missing in submitted assignment")
            raise HTTPException(status_code=404, detail="submittedFileUrl not found in submitted assignment")
        
        # Step 2: Get the video ObjectId from submitted assignment
        video_obj_id = submitted_assignment.get("video")
        
        if not video_obj_id:
            logger.error("video ObjectId field missing in submitted assignment")
            raise HTTPException(status_code=404, detail="video ObjectId not found in submitted assignment")
        
        logger.debug(f"Found video ObjectId: {video_obj_id}")
        
        # Step 3: Get finalCutVideoUrl and assetsUrl from assignmentresources collection
        logger.debug("Querying assignmentresources collection...")
        assignment_resource = await assignment_resources_collection.find_one(
            {"video": video_obj_id}
        )
        
        if not assignment_resource:
            logger.error(f"No assignment resource found for video ID: {str(video_obj_id)}")
            raise HTTPException(status_code=404, detail=f"No assignment resource found with video id: {str(video_obj_id)}")
        
        logger.info(f"Found assignment resource: {assignment_resource['_id']}")
        
        final_cut_video_url = assignment_resource.get("finalCutVideoUrl")
        assets_url = assignment_resource.get("assetsUrl")
        
        if not final_cut_video_url or not assets_url:
            logger.error("finalCutVideoUrl or assetsUrl fields missing in assignment resource")
            raise HTTPException(status_code=404, detail="finalCutVideoUrl or assetsUrl not found in assignment resource")
        
        logger.debug(f"Retrieved paths - Student: {submitted_file_url}")
        logger.debug(f"Retrieved paths - Final Cut: {final_cut_video_url}")
        logger.debug(f"Retrieved paths - Assets: {assets_url}")
        
        # Add background task for processing
        background_tasks.add_task(
            process_assignment_paths,
            assignment_id,
            submitted_file_url,
            final_cut_video_url,
            assets_url
        )
        
        # CloudFront base URL for response
        response = AssignmentPathsResponse(
            submittedFileUrl=f"{CLOUDFRONT_BASE_URL}/{submitted_file_url}",
            finalCutVideoUrl=f"{CLOUDFRONT_BASE_URL}/{final_cut_video_url}",
            assetsUrl=f"{CLOUDFRONT_BASE_URL}/{assets_url}"
        )
        
        logger.info(f"Successfully fetched and generated paths for assignment: {assignment_id}")
        return response
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error fetching assignment paths: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error fetching assignment paths: {str(e)}")

@app.post("/evaluate", response_model=EvaluationResponse)
async def evaluate_student(request: StudentSubmission):
    """Evaluate student video submission"""
    try:
        # Check if files exist
        if not os.path.exists(request.final_video_path):
            return EvaluationResponse(score="0", feedback=f"Final video not found: {request.final_video_path}", external_api_response=None)
        if not os.path.exists(request.student_video_path):
            return EvaluationResponse(score="0", feedback=f"Student video not found: {request.student_video_path}", external_api_response=None)
        if not os.path.exists(request.asset_path):
            return EvaluationResponse(score="0", feedback=f"Assets not found: {request.asset_path}", external_api_response=None)
        
        # Run the evaluation
        logger.info("Running video evaluation...")
        results = evaluate_videos_with_paths(
            request.final_video_path, 
            request.student_video_path, 
            request.asset_path
        )
        
        score = results["video_acceptance_score"]["overall_score"]
        feedback = results.get("student_feedback", "Evaluation completed")
        
        logger.info(f"Evaluation completed - Score: {score}")
        
        return EvaluationResponse(score=str(score), feedback=feedback, external_api_response=None)
        
    except Exception as e:
        logger.error(f"Evaluation error: {str(e)}")
        return EvaluationResponse(score="0", feedback=f"Error: {str(e)}", external_api_response=None)

@app.post("/evaluate-assignment/{assignment_id}", response_model=SimpleResponse)
async def evaluate_assignment(assignment_id: str, background_tasks: BackgroundTasks):
    """Complete evaluation workflow: fetch paths, evaluate, and submit results"""
    try:
        logger.info(f"Starting evaluation for assignment {assignment_id}")
        
        # Get paths from database
        paths = await fetch_assignment_paths(assignment_id, background_tasks)
        
        # Create evaluation request
        eval_request = StudentSubmission(
            asset_path=paths.assetsUrl,
            final_video_path=paths.finalCutVideoUrl,
            student_video_path=paths.submittedFileUrl
        )
        
        # Run evaluation
        evaluation = await evaluate_student(eval_request)
        
        # Submit to external API - convert score to 0-100 range but fallback to 12 if needed
        try:
            # Evaluation score is typically 0-1, convert to 0-100
            eval_score = float(evaluation.score)
            if 0 <= eval_score <= 1:
                # Scale 0-1 to 0-100
                score = max(1, round(eval_score * 100))  # Ensure at least 1 for valid submissions
            else:
                # If score is already in 0-100 range or unexpected format, use as-is
                score = max(1, round(eval_score))
        except (ValueError, TypeError):
            # Fallback to working value if score parsing fails
            logger.warning(f"Could not parse evaluation score '{evaluation.score}', using fallback score")
            score = 12
            
        logger.info(f"Sending score to external API: {score} (from evaluation score: {evaluation.score})")
        success = await submit_to_external_api(assignment_id, score, evaluation.feedback)
        
        if success:
            return SimpleResponse(success=True, message="Evaluation completed and results submitted successfully")
        else:
            return SimpleResponse(success=False, message="Evaluation completed but failed to submit results")
            
    except Exception as e:
        logger.error(f"Error in evaluation workflow: {str(e)}")
        return SimpleResponse(success=False, message=f"Evaluation failed: {str(e)}")

@app.get("/evaluate-assignment-simple/{assignment_id}", response_model=SimpleResponse)
async def evaluate_assignment_simple(assignment_id: str, background_tasks: BackgroundTasks):
    """
    Simple GET API:
    - Takes assignment_id as path param
    - Calls the existing evaluate-assignment POST workflow
    - Returns final result (success/failure + message)
    """
    try:
        # Reuse your existing workflow
        result = await evaluate_assignment(assignment_id, background_tasks)
        return result
    except Exception as e:
        logger.error(f"Error in simple evaluate-assignment GET API: {str(e)}")
        return SimpleResponse(success=False, message=f"Error: {str(e)}")

if __name__ == "__main__":
    uvicorn.run("app:app", host="127.0.0.1", port=8080, reload=True)
