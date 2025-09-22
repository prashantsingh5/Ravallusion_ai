import os
import glob
import random
import torch
import cv2
import numpy as np
from tqdm import tqdm

from ultralytics import YOLO

from ravallusion_ai.core import extract_frames_from_video,pair_objects_by_bbox
from ravallusion_ai.core import *
from ravallusion_ai.core.feedback import get_detailed_feedback
from ravallusion_ai.core.audio_asset_matcher import AudioAssetMatcher

import os
import shutil
import uuid
from tqdm import tqdm
from PIL import Image
from collections import namedtuple

import logging

def get_audio_transcript(video_file_path : str = None,audio_file_path : str = None, save_dir :str = "test_aud"):

    if video_file_path is not None:
        # Create temp directory if it doesn't exist
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
            
        save_path = os.path.join(save_dir,f"{os.path.basename(video_file_path).split('.')[0]}.mp3")
        raw_video_transcript = transcribe_audio(extract_audio_from_video(video_file_path,save_path=save_path))

        return raw_video_transcript

    elif audio_file_path is not None:

        return transcribe_audio(audio_file_path)
    

def evaluate_audio_similarity(final_cut_video_path,random_selected_video_path):
    
    raw_video_audio_transcript = get_audio_transcript(final_cut_video_path)
    raw_video_embedding = EMBEDDING_MODEL.encode(raw_video_audio_transcript)

    other_video_audio_transcript = get_audio_transcript(random_selected_video_path) 
    assgn_video_embedding = EMBEDDING_MODEL.encode(other_video_audio_transcript)

    audio_score = cosine_similarity(raw_video_embedding,assgn_video_embedding)
    return audio_score
        

def evaluate_video_similarity(
    final_cut_frames,
    other_vid_frames,
    final_cut_obj_detector,
    other_obj_detector,
    image_comparator,
    asset_paths,
    iou_thresh=0.5,
    similarity_thresh=0.7,
    weights={
        'host': 0.1,
        'visual': 0.3,
        'textual': 0.1,
        'asset_presence': 0.3,
        'accurate_placement': 0.2
    }
):
    
    total_frames = min(len(other_vid_frames),len(final_cut_frames))
    fc = FontClassifier(device="cpu")

    host_visible_in_final_count = 0
    host_not_in_other_count = 0
    total_text_objects = 0
    valid_text_score_sum = 0
    total_paired_objects = 0
    similar_paired_objects = 0
    
    assets_used_final = set() 
    accurately_placed_assets = set() 

    #######################################
    assets_used = set()
    host_visibility = total_frames
    valid_text_score = 0
    accurate_placement_score = 0
        

    # logging.info("\n=== Starting Video Evaluation ===\n")
    # logging.info(f"Number of frames to process: {total_frames}")


    for idx in tqdm(range(total_frames), desc="Detecting & Comparing"):


        temp_dir_final = os.path.join("temp_cropped_final", f"Frame_{idx}")
        temp_dir_other = os.path.join("temp_cropped_other", f"Frame_{idx}")
        os.makedirs(temp_dir_final, exist_ok=True)
        os.makedirs(temp_dir_other, exist_ok=True)

        log_message = f"\n--- Frame {idx} ---\n"
                
        final_objs = final_cut_obj_detector.predict_and_crop(final_cut_frames[idx], temp_dir=temp_dir_final)
        other_objs = other_obj_detector.predict_and_crop(other_vid_frames[idx], temp_dir=temp_dir_other)

        # Host visibility check
        final_has_host = any(obj['class'] == 'Host' for obj in final_objs)
        other_has_host = any(obj['class'] == 'Host' for obj in other_objs)

        if final_has_host:
            host_visible_in_final_count += 1
            if not other_has_host:
                host_not_in_other_count += 1
                # log_message += "Host detected in final cut but not in uploaded video.\n"
            else:
                pass
                # log_message += "Host detected in both videos.\n"

        # Textual objects
        # TODO : Add OCR text-to-text comparison
        textual_objects = [obj for obj in other_objs if obj['class']=='Text']
        if textual_objects:
            for txt_obj in textual_objects:
                total_text_objects += 1
                txt_info = fc.predict_single(txt_obj['cropped_image_path'])
                valid_text_score_sum += txt_info['class_idx']
                # log_message += f"Text object detected. Font Classifier result: {txt_info}.\n"

        # Compare visual objects
        final_visual_objects = [final_obj for final_obj in final_objs if final_obj['class']=='Image' or final_obj['class'] =="Cutoff" ]
        other_visual_objects = [other_obj for other_obj in other_objs if other_obj['class']=='Image' or other_obj['class'] =="Cutoff" ]
        pairs, _, _ = pair_objects_by_bbox(final_visual_objects, other_visual_objects, iou_thresh=iou_thresh)


        # Check if visual objects detected are assets or not 
        for obj in other_visual_objects:
            for asset_path in asset_paths:
                if asset_path not in assets_used_final:
                    score = image_comparator.get_similarity_score(obj['cropped_image_path'], asset_path)
                    if score >= similarity_thresh:
                        assets_used_final.add(asset_path)
                        # log_message += f"Detected asset {os.path.basename(asset_path)} in the uploaded video.\n"
# ------------------------ debug step for asset found folder step start --------------------                        
                        
                        # Save detected asset to asset_found folder
                        # asset_found_dir = "asset_found"
                        # os.makedirs(asset_found_dir, exist_ok=True)
                        
                        # Copy the cropped object image that matched the asset
                        # asset_name = os.path.basename(asset_path)
                        # asset_name_no_ext = os.path.splitext(asset_name)[0]
                        # cropped_image_name = f"frame_{idx}_{asset_name_no_ext}_detected.png"
                        # cropped_save_path = os.path.join(asset_found_dir, cropped_image_name)
                        # shutil.copy2(obj['cropped_image_path'], cropped_save_path)
                        
                        # Also copy the original asset for reference
                        # original_asset_save_path = os.path.join(asset_found_dir, f"original_{asset_name}")
                        # if not os.path.exists(original_asset_save_path):
                        #     shutil.copy2(asset_path, original_asset_save_path)
                        
                        # log_message += f"   --> Saved detected asset to {cropped_save_path}\n"
# ------------------------- debug step for asset found folder step end --------------------

                        break 
        
        for f_obj, o_obj in pairs:
            total_paired_objects += 1
            score = image_comparator.get_similarity_score(f_obj['cropped_image_path'], o_obj['cropped_image_path'])
            if score >= similarity_thresh:
                similar_paired_objects += 1
                # log_message += f"Paired visual object match found with similarity score: {score:.2f}.\n"
                
                for asset_path in assets_used_final:
                    asset_score = image_comparator.get_similarity_score(o_obj['cropped_image_path'], asset_path)
                    if asset_score >= similarity_thresh:
                        accurately_placed_assets.add(asset_path)
                        # log_message += f"   --> Object also matched asset {os.path.basename(asset_path)}. Accurate placement !!\n"
                        break

        # logging.info(log_message)

        # shutil.rmtree(temp_dir_final)
        # shutil.rmtree(temp_dir_other)

        # logging.info("\n=== Final Scoring ===\n")

    # Final score calculations
    # host_score = host_not_in_other_count / max(1, host_visible_in_final_count)     # PENALTY SYSTEM
    host_score = 1 - (host_not_in_other_count / max(1, host_visible_in_final_count))    # REWARD SYSTEM
    valid_text_score = valid_text_score_sum / max(1, total_text_objects)
    
    # Use the single set for the final asset presence score
    asset_presence_score = len(assets_used_final) / max(1, len(asset_paths))
    
    visual_similarity_score = similar_paired_objects / max(1, total_paired_objects)
    accurate_placement_score = len(accurately_placed_assets) / max(1, len(asset_paths))

    # logging.info(f"Host Score: {host_score:.2f}")
    # logging.info(f"Text Score: {valid_text_score:.2f}")
    # logging.info(f"Asset Presence Score: {asset_presence_score:.2f}")
    # logging.info(f"Visual Similarity Score: {visual_similarity_score:.2f}")
    # logging.info(f"Accurate Asset Placement Score: {accurate_placement_score:.2f}")

    # print("\n--- Final Scores ---")
    # print(f"Host Score: {host_score:.2f}")
    # print(f"Text Score: {valid_text_score:.2f}")
    # print(f"Asset Presence Score: {asset_presence_score:.2f}")
    # print(f"Visual Similarity Score: {visual_similarity_score:.2f}")
    # print(f"Accurate Asset Placement Score: {accurate_placement_score:.2f}")

# ---------------------------------------Asset Debugging Info start---------------------------------------

    # Asset detection summary
    # print("\n--- Asset Detection Summary ---")
    # print(f"Total assets to check: {len(asset_paths)}")
    # print(f"Assets found in student video: {len(assets_used_final)}")
    # print(f"Assets correctly placed: {len(accurately_placed_assets)}")
    
    # print("\nAssets available for evaluation:")
    # for asset_path in asset_paths:
    #     asset_name = os.path.basename(asset_path)
    #     print(f"  - {asset_name}")
    
    # print("\nAssets detected in student video:")
    # if assets_used_final:
    #     for asset_path in assets_used_final:
    #         asset_name = os.path.basename(asset_path)
    #         placement_status = "correctly placed" if asset_path in accurately_placed_assets else "incorrectly placed"
    #         print(f"  ✓ {asset_name} ({placement_status})")
    # else:
    #     print("  No assets detected")
    
    # print("\nMissing assets:")
    # missing_assets = set(asset_paths) - assets_used_final
    # if missing_assets:
    #     for asset_path in missing_assets:
    #         asset_name = os.path.basename(asset_path)
    #         print(f"  ✗ {asset_name}")
    # else:
    #     print("  None - all assets detected")
# --------------------Debug step for asset found folder --------------------
    # Asset found folder summary
    # asset_found_dir = "asset_found"
    # if os.path.exists(asset_found_dir) and assets_used_final:
    #     print(f"\n--- Detected Assets Saved ---")
    #     print(f"Detected asset images saved to: {os.path.abspath(asset_found_dir)}")
    #     saved_files = [f for f in os.listdir(asset_found_dir) if f.endswith('.png') or f.endswith('.jpg')]
    #     print(f"Total files saved: {len(saved_files)}")
    #     for file in saved_files:
    #         print(f"  📁 {file}")

# ---------------------------------------Asset Debugging Info end ---------------------------------------

    overall_score = (
        host_score * weights['host'] +
        visual_similarity_score * weights['visual'] +
        valid_text_score * weights['textual'] +
        asset_presence_score * weights['asset_presence'] +
        accurate_placement_score * weights['accurate_placement']
    )

    # print(overall_score)
    # logging.info(f"Overall Score: {overall_score:.2f}")

    # Calculate missing assets for feedback
    missing_assets = set(asset_paths) - assets_used_final

    return {
        "overall_score": overall_score,
        "host_score": host_score if host_visible_in_final_count else 1,
        "text_score": valid_text_score if total_text_objects else 1,
        "asset_presence_score": asset_presence_score,
        "visual_similarity_score": visual_similarity_score,
        "accurate_placement_score": accurate_placement_score,
        "assets_used": assets_used_final,
        "missing_assets": missing_assets,
        "total_assets": asset_paths,
    }   


def load_assets_from_directory(assets_dir_path):
    """
    Load all assets from directory including images and video first frames.
    Supports PNG, JPG, JPEG images and extracts first frame from videos.
    """
    asset_paths = []
    
    if not os.path.exists(assets_dir_path):
        return asset_paths
    
    # Get all files in assets directory
    for file_path in glob.glob(os.path.join(assets_dir_path, "*")):
        if os.path.isfile(file_path):
            file_ext = os.path.splitext(file_path)[1].lower()
            
            # Handle image files
            if file_ext in ['.png', '.jpg', '.jpeg']:
                asset_paths.append(file_path)
            
            # Handle video files
            elif file_ext in ['.mp4', '.avi', '.mov', '.mkv', '.wmv']:
                video_frame_path = extract_first_frame_from_video(file_path, assets_dir_path)
                if video_frame_path:
                    asset_paths.append(video_frame_path)
    
    return asset_paths


def extract_first_frame_from_video(video_path, assets_dir):
    """Extract first frame from video and save as PNG for asset matching."""
    try:
        video_name = os.path.splitext(os.path.basename(video_path))[0]
        frame_save_path = os.path.join(assets_dir, f"{video_name}_first_frame.png")
        
        # Skip if frame already exists
        if os.path.exists(frame_save_path):
            return frame_save_path
        
        # Extract first frame
        cap = cv2.VideoCapture(video_path)
        ret, frame = cap.read()
        cap.release()
        
        if ret:
            cv2.imwrite(frame_save_path, frame)
            return frame_save_path
        
    except Exception:
        pass
    
    return None


def evaluate_videos_with_paths(final_cut_video_path, to_be_evaluated_video_path, assets_dir_path):
    """
    New function that accepts specific paths instead of random selection.
    
    Args:
        final_cut_video_path (str): Path to the final cut video file
        to_be_evaluated_video_path (str): Path to the video to be evaluated
        assets_dir_path (str): Path to the directory containing asset files
    
    Returns:
        dict: Dictionary containing all evaluation scores and feedback
    """
    # Load assets from directory (images and video frames)
    asset_paths = load_assets_from_directory(assets_dir_path)
    print(f"Loaded {len(asset_paths)} assets for matching")
    
    # Extract frames from both videos
    final_cut_vid_duration, final_cut_no_of_frames, final_cut_vid_frames_save_path = extract_frames_from_video(
        final_cut_video_path, 'final_cut_video_frames', every=3
    )
    other_vid_duration, other_no_of_frames, other_vid_frames_save_path = extract_frames_from_video(
        to_be_evaluated_video_path, 'other_video_frames', every=3
    )
    
    # Get sorted frame paths
    final_cut_frames = sorted(
        glob.glob(final_cut_vid_frames_save_path + '/*'), 
        key=lambda x: int(os.path.splitext(os.path.basename(x))[0])
    )
    other_vid_frames = sorted(
        glob.glob(other_vid_frames_save_path + '/*'), 
        key=lambda x: int(os.path.splitext(os.path.basename(x))[0])
    )
    
    # Initialize detectors and comparator
    final_cut_obj_detector = ObjectsDetector(save_dir="final_cut_video", debug=True)
    other_obj_detector = ObjectsDetector(save_dir="other_video", debug=True)
    
    # Use CUDA if available, otherwise fallback to CPU
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    image_comparator = ImageComparator(device=device)
    
    # Perform evaluations
    video_acceptance_score = evaluate_video_similarity(
        final_cut_frames, other_vid_frames, final_cut_obj_detector, 
        other_obj_detector, image_comparator, asset_paths
    )
    response = evaluate_video(to_be_evaluated_video_path)
    asgn_video_score, asgn_video_feedback = response.parsed.total_score, response.parsed.feedback
    
    # Original audio similarity score
    audio_similarity_score = evaluate_audio_similarity(final_cut_video_path, to_be_evaluated_video_path) / 100
    
    # NEW: Audio asset matching
    audio_asset_matcher = AudioAssetMatcher()
    detected_audio_assets, total_audio_assets, detected_count, audio_asset_score = audio_asset_matcher.detect_audio_assets(
        to_be_evaluated_video_path, assets_dir_path
    )
    
    # Combine audio scores according to the specified logic
    final_audio_score = audio_asset_matcher.combine_audio_scores(
        audio_similarity_score, audio_asset_score, total_audio_assets
    )
    
    # Generate comprehensive feedback for student
    comprehensive_results = {
        "video_acceptance_score": video_acceptance_score,
        "gemini_video_score": asgn_video_score,
        "gemini_video_feedback": asgn_video_feedback,
        "audio_similarity_score": audio_similarity_score,  # Original audio score
        "audio_asset_score": audio_asset_score,  # New audio asset score
        "final_audio_score": final_audio_score,  # Combined audio score
        "detected_audio_assets": detected_audio_assets,  # List of detected assets
        "total_audio_assets": total_audio_assets,  # Total available assets
        "detected_audio_count": detected_count,  # Number of detected assets
        "asset_paths_used": asset_paths,
        "final_cut_frames_count": len(final_cut_frames),
        "evaluated_frames_count": len(other_vid_frames)
    }
    
    student_feedback = get_detailed_feedback(comprehensive_results)
    comprehensive_results["student_feedback"] = student_feedback
    
    # Clean up temporary debugging folders
    cleanup_debug_folders()
    
    # Return comprehensive results
    return comprehensive_results


def cleanup_debug_folders():
    """
    Remove all temporary and debugging folders created during processing.
    This keeps the module clean and only returns scores and feedback.
    """
    import shutil
    
    debug_folders = [
        "temp_cropped_final",
        "temp_cropped_other", 
        "asset_found",
        "test_aud",
        "final_cut_video_frames",
        "other_video_frames",
        "final_cut_video",
        "other_video"
    ]
    
    for folder in debug_folders:
        if os.path.exists(folder):
            try:
                shutil.rmtree(folder)
                # print(f"Cleaned up debugging folder: {folder}")  # Uncomment for debugging
            except Exception as e:
                # print(f"Warning: Could not remove {folder}: {e}")  # Uncomment for debugging
                pass


def main():
    # ORIGINAL MAIN FUNCTION - COMMENTED OUT TO PRESERVE EXISTING FUNCTIONALITY
    # This is the original random selection logic that was working before
    """
    test_dir = "C:\\Users\\pytorch\\Desktop\\final\\test"
    asset_paths = glob.glob(f"{os.path.join(test_dir,'assets')}/*.png")
    final_cut_video_path = glob.glob(f"{os.path.join(test_dir,'final_cut_video')}/*")[0]
    to_be_evaluated_video_paths = glob.glob(f"{os.path.join(test_dir,'to_be_evaluated_videos')}/*")

    random_selected_video_path = random.choice(to_be_evaluated_video_paths)

    
    final_cut_vid_duration, final_cut_no_of_frames, final_cut_vid_frames_save_path = extract_frames_from_video(final_cut_video_path,'final_cut_video_frames',every=5)
    other_vid_duration, other_no_of_frames, other_vid_frames_save_path = extract_frames_from_video(random_selected_video_path,'other_video_frames',every=5)
    # print(final_cut_vid_duration,other_vid_duration,final_cut_no_of_frames,other_no_of_frames)

    # final_cut_no_of_frames = 100
    # final_cut_vid_frames_save_path = "final_cut_video_frames/V7-100_ Correct Video.mp4"
    # other_vid_frames_save_path = "other_video_frames/V4-Wrong Fonts.mp4"
    # print(final_cut_vid_frames_save_path,other_vid_frames_save_path)

    final_cut_frames = sorted(glob.glob(final_cut_vid_frames_save_path+'/*'), key=lambda x : int(os.path.splitext(os.path.basename(x))[0]))
    other_vid_frames = sorted(glob.glob(other_vid_frames_save_path+'/*'), key=lambda x : int(os.path.splitext(os.path.basename(x))[0]))

    final_cut_obj_detector = ObjectsDetector(save_dir="final_cut_video",debug=True)
    other_obj_detector = ObjectsDetector(save_dir="other_video",debug=True)

    # Use CUDA if available, otherwise fallback to CPU
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    image_comparator = ImageComparator(device=device)
    final_cut_host_present_count = 0
    other_vid_host_present_count = 0

    video_acceptance_score = evaluate_video_similarity(final_cut_frames,other_vid_frames,final_cut_obj_detector,other_obj_detector,image_comparator,asset_paths)
    response = evaluate_video(random_selected_video_path)
    asgn_video_score, asgn_video_feedback = response.parsed.total_score,response.parsed.feedback
    audio_similarity_score = evaluate_audio_similarity(final_cut_video_path,random_selected_video_path)/100
    print(video_acceptance_score)
    print(asgn_video_score,asgn_video_feedback)
    print(audio_similarity_score)
    """
    
    # NEW IMPLEMENTATION - Using the new function with existing paths for compatibility
    # This maintains the same behavior as before but now uses the parameterized function
    test_dir = "C:\\Users\\pytorch\\Desktop\\final\\test"
    assets_dir = os.path.join(test_dir, 'assets')
    final_cut_video_path = glob.glob(f"{os.path.join(test_dir,'final_cut_video')}/*")[0]
    to_be_evaluated_video_paths = glob.glob(f"{os.path.join(test_dir,'to_be_evaluated_videos')}/*")
    
    random_selected_video_path = random.choice(to_be_evaluated_video_paths)
    
    # Call the new parameterized function
    results = evaluate_videos_with_paths(
        final_cut_video_path=final_cut_video_path,
        to_be_evaluated_video_path=random_selected_video_path,
        assets_dir_path=assets_dir
    )
    
    # Print results in the same format as original
    print(results["video_acceptance_score"])
    print(results["gemini_video_score"], results["gemini_video_feedback"])
    print(results["audio_similarity_score"])

if __name__ == "__main__":
    main()