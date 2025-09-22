import cv2
import os
import shutil
import numpy as np
from tqdm import tqdm


def extract_frames_from_video(video_path, save_dir="temp", start=-1, end=-1, every=1):
 
    video_dir, video_filename = os.path.split(video_path)  
    if os.path.exists(os.path.join(save_dir,video_filename)):
        shutil.rmtree(os.path.join(save_dir,video_filename))
    os.makedirs(os.path.join(save_dir,video_filename))

    print(f"[+] Extracting frames of {video_filename} !")
    capture = cv2.VideoCapture(video_path)  

    if start < 0:  
        start = 0
    if end < 0:  
        end = int(frame_count:=capture.get(cv2.CAP_PROP_FRAME_COUNT))


    capture.set(1, start)  
    frame = start  
    while_safety = 0  
    saved_count = 0  

    fps = capture.get(cv2.CAP_PROP_FPS)
    duration_of_video = frame_count/fps

    progress_bar = iter(tqdm(range(end),desc='Extracting frames : '))
    while frame < end:

        _, image = capture.read()

        if while_safety > 500: 
            break

        if image is None:  
            while_safety += 1 
            continue  

        if frame % every == 0:  
            while_safety = 0 
            next(progress_bar)
            save_path = os.path.join(save_dir, video_filename, "{:010d}.png".format(frame)) 
            if not os.path.exists(save_path) : 
                cv2.imwrite(save_path, image)  
                saved_count += 1 
        
        frame += 1  

    capture.release()

    return duration_of_video, saved_count, os.path.join(save_dir,video_filename)




def bbox_iou(box1, box2):
    xA = max(box1[0], box2[0])
    yA = max(box1[1], box2[1])
    xB = min(box1[2], box2[2])
    yB = min(box1[3], box2[3])

    interArea = max(0, xB - xA) * max(0, yB - yA)
    box1Area = (box1[2] - box1[0]) * (box1[3] - box1[1])
    box2Area = (box2[2] - box2[0]) * (box2[3] - box2[1])
    iou = interArea / float(box1Area + box2Area - interArea + 1e-6)
    return iou

def center_distance(box1, box2):
    c1 = ((box1[0] + box1[2]) / 2, (box1[1] + box1[3]) / 2)
    c2 = ((box2[0] + box2[2]) / 2, (box2[1] + box2[3]) / 2)
    return np.sqrt((c1[0] - c2[0]) ** 2 + (c1[1] - c2[1]) ** 2)



def pair_objects_by_bbox(objs_a, objs_b, iou_thresh=0.3):

    matched_pairs = []
    unmatched_a = objs_a.copy()
    unmatched_b = objs_b.copy()

    used_b_indices = set()

    for obj_a in objs_a:
        best_match = None
        best_idx_b = None
        best_score = 0

        for idx_b, obj_b in enumerate(objs_b):
            if idx_b in used_b_indices:
                continue

            iou = bbox_iou(obj_a['bbox'], obj_b['bbox'])

            if iou >= iou_thresh :
                if iou > best_score:
                    best_score = iou
                    best_match = obj_b
                    best_idx_b = idx_b

        if best_match:
            matched_pairs.append((obj_a, best_match))
            used_b_indices.add(best_idx_b)
            unmatched_a.remove(obj_a)
            unmatched_b.remove(best_match)

    return matched_pairs, unmatched_a, unmatched_b
