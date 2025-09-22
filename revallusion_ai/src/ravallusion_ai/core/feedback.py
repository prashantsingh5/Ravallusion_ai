import os
from google import genai
from typing import Dict, Any
from pydantic import BaseModel, Field

# Use same API key as gemini_service.py
api_key = "AIzaSyDcU_qZ1VbWOoRogappbV0NtDTn_xzhlOw"

class FeedbackResult(BaseModel):
    feedback: str = Field(description="Detailed personalized feedback for student improvement")

class VideoFeedbackGenerator:
    def __init__(self):
        self.client = genai.Client(api_key=api_key)
        self.score_thresholds = {
            'excellent': 0.85,
            'good': 0.70,
            'fair': 0.50,
            'poor': 0.30
        }
    
    def analyze_performance_level(self, score):
        """Determine performance level based on score"""
        if score >= self.score_thresholds['excellent']:
            return 'excellent'
        elif score >= self.score_thresholds['good']:
            return 'good'
        elif score >= self.score_thresholds['fair']:
            return 'fair'
        elif score >= self.score_thresholds['poor']:
            return 'poor'
        else:
            return 'needs_improvement'
    
    def get_score_insights(self, scores):
        """Extract detailed insights from all scores"""
        video_score = scores["video_acceptance_score"]
        
        insights = {
            'overall': {
                'score': video_score.get('overall_score', 0),
                'level': self.analyze_performance_level(video_score.get('overall_score', 0))
            },
            'host_presence': {
                'score': video_score.get('host_score', 0),
                'level': self.analyze_performance_level(video_score.get('host_score', 0)),
                'meaning': 'Host visibility and presence compared to reference video'
            },
            'textual_accuracy': {
                'score': video_score.get('text_score', 0),
                'level': self.analyze_performance_level(video_score.get('text_score', 0)),
                'meaning': 'Accuracy of text elements and on-screen content'
            },
            'asset_utilization': {
                'score': video_score.get('asset_presence_score', 0),
                'level': self.analyze_performance_level(video_score.get('asset_presence_score', 0)),
                'meaning': 'Percentage of required assets included in the video'
            },
            'visual_similarity': {
                'score': video_score.get('visual_similarity_score', 0),
                'level': self.analyze_performance_level(video_score.get('visual_similarity_score', 0)),
                'meaning': 'Overall visual composition similarity to reference'
            },
            'asset_placement': {
                'score': video_score.get('accurate_placement_score', 0),
                'level': self.analyze_performance_level(video_score.get('accurate_placement_score', 0)),
                'meaning': 'Correctness of asset positioning and timing'
            },
            'content_quality': {
                'score': scores.get('gemini_video_score', 0),
                'level': self.analyze_performance_level(scores.get('gemini_video_score', 0)),
                'meaning': 'Overall content structure and narrative quality'
            },
            'audio_quality': {
                'score': scores.get('final_audio_score', scores.get('audio_similarity_score', 0)),
                'level': self.analyze_performance_level(scores.get('final_audio_score', scores.get('audio_similarity_score', 0))),
                'meaning': 'Combined audio similarity and asset usage score',
                'original_audio_score': scores.get('audio_similarity_score', 0),
                'audio_asset_score': scores.get('audio_asset_score', 0),
                'detected_audio_assets': scores.get('detected_audio_assets', []),
                'total_audio_assets': scores.get('total_audio_assets', 0)
            }
        }
        
        return insights
    
    def get_missing_assets_info(self, scores):
        """Get information about missing assets"""
        video_score = scores["video_acceptance_score"]
        missing_assets = video_score.get('missing_assets', set())
        total_assets = video_score.get('total_assets', [])
        
        if missing_assets:
            missing_files = [os.path.basename(asset) for asset in missing_assets]
            return f"Missing Assets: {', '.join(missing_files)}"
        else:
            return "All required assets were found in the video"
    
    def create_feedback_prompt(self, insights, missing_assets_info):
        """Generate comprehensive feedback prompt for Gemini"""
        
        prompt = f"""You are an expert video production instructor providing detailed feedback to help students improve their video creation skills.

Analyze the following video evaluation scores and provide constructive, actionable feedback:

PERFORMANCE BREAKDOWN:
- Overall Score: {insights['overall']['score']:.2f} ({insights['overall']['level']})
- Host Presence: {insights['host_presence']['score']:.2f} ({insights['host_presence']['level']})
- Text Accuracy: {insights['textual_accuracy']['score']:.2f} ({insights['textual_accuracy']['level']})
- Asset Usage: {insights['asset_utilization']['score']:.2f} ({insights['asset_utilization']['level']})
- Visual Match: {insights['visual_similarity']['score']:.2f} ({insights['visual_similarity']['level']})
- Asset Placement: {insights['asset_placement']['score']:.2f} ({insights['asset_placement']['level']})
- Content Quality: {insights['content_quality']['score']:.2f} ({insights['content_quality']['level']})
- Audio Quality: {insights['audio_quality']['score']:.2f} ({insights['audio_quality']['level']})

ASSET ANALYSIS:
{missing_assets_info}

SCORE DEFINITIONS:
- Host Presence: How well the presenter appears and is visible compared to the reference video
- Text Accuracy: Correctness and clarity of text elements, titles, and on-screen information
- Asset Usage: Percentage of required visual assets (logos, graphics, images) that were included
- Visual Match: Overall visual composition and scene similarity to the reference video
- Asset Placement: How accurately assets are positioned, timed, and integrated
- Content Quality: Narrative structure, flow, and overall production value
- Audio Quality: Voice clarity, background music, and audio synchronization

Provide feedback in this format:

**STRENGTHS:**
[Highlight what the student did well based on scores above 0.7]

**AREAS FOR IMPROVEMENT:**
[Focus on the lowest scoring areas with specific suggestions]

**MISSING ASSETS TO ADD:**
[List specific asset files that need to be included if any are missing]

**PRIORITY ACTIONS:**
[List 3-4 most important improvements in order of priority]

**TECHNICAL RECOMMENDATIONS:**
[Specific technical suggestions for video production]

**OVERALL ASSESSMENT:**
[Brief summary with encouragement and next steps]

Keep the tone supportive but honest. Focus on actionable improvements rather than just stating scores.
Provide specific file names for missing assets when applicable.
"""
        return prompt
    
    def generate_feedback(self, scores):
        """Generate personalized feedback using Gemini AI"""
        try:
            insights = self.get_score_insights(scores)
            missing_assets_info = self.get_missing_assets_info(scores)
            prompt = self.create_feedback_prompt(insights, missing_assets_info)
            
            response = self.client.models.generate_content(
                model="gemini-2.0-flash",
                contents=[prompt],
                config={
                    'response_mime_type': 'application/json',
                    'response_schema': FeedbackResult,
                },
            )
            
            return response.parsed.feedback
            
        except Exception as e:
            return self.get_fallback_feedback(scores)
    
    def get_fallback_feedback(self, scores):
        """Provide basic feedback if Gemini API fails"""
        video_score = scores["video_acceptance_score"]
        overall = video_score.get('overall_score', 0)
        missing_assets_info = self.get_missing_assets_info(scores)
        
        feedback_base = ""
        if overall >= 0.8:
            feedback_base = "Excellent work! Your video meets most requirements with high quality production values. Focus on fine-tuning the remaining details."
        elif overall >= 0.6:
            feedback_base = "Good effort! Your video shows solid understanding. Review areas with lower scores and consider improving asset placement and visual consistency."
        elif overall >= 0.4:
            feedback_base = "Your video shows promise but needs significant improvements. Focus on including all required assets and matching the reference video more closely."
        else:
            feedback_base = "This video needs substantial revision. Please review the reference material carefully and ensure all required elements are included with proper placement and timing."
        
        return f"{feedback_base}\n\nASSET STATUS:\n{missing_assets_info}"

def get_detailed_feedback(evaluation_results):
    """Main function to generate feedback from evaluation results"""
    feedback_generator = VideoFeedbackGenerator()
    return feedback_generator.generate_feedback(evaluation_results)
