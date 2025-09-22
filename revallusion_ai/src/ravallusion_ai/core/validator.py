
##put all these here


# def pair_objects_by_bbox(objs_a, objs_b, iou_thresh=0.3):


#     def bbox_iou(box1, box2):
#         """Compute IoU between two xyxy boxes."""
#         xA = max(box1[0], box2[0])
#         yA = max(box1[1], box2[1])
#         xB = min(box1[2], box2[2])
#         yB = min(box1[3], box2[3])

#         interArea = max(0, xB - xA) * max(0, yB - yA)
#         box1Area = (box1[2] - box1[0]) * (box1[3] - box1[1])
#         box2Area = (box2[2] - box2[0]) * (box2[3] - box2[1])
#         iou = interArea / float(box1Area + box2Area - interArea + 1e-6)
#         return iou

#     def center_distance(box1, box2):
#         """Euclidean distance between bbox centers."""
#         c1 = ((box1[0] + box1[2]) / 2, (box1[1] + box1[3]) / 2)
#         c2 = ((box2[0] + box2[2]) / 2, (box2[1] + box2[3]) / 2)
#         return np.sqrt((c1[0] - c2[0]) ** 2 + (c1[1] - c2[1]) ** 2)

#     matched_pairs = []
#     unmatched_a = objs_a.copy()
#     unmatched_b = objs_b.copy()

#     used_b_indices = set()

#     for obj_a in objs_a:
#         best_match = None
#         best_idx_b = None
#         best_score = 0

#         for idx_b, obj_b in enumerate(objs_b):
#             if idx_b in used_b_indices:
#                 continue

#             iou = bbox_iou(obj_a['bbox'], obj_b['bbox'])
#             # dist = center_distance(obj_a['bbox'], obj_b['bbox'])

#             # Match criteria: nearby & overlapping enough
#             if iou >= iou_thresh :
#                 if iou > best_score:
#                     best_score = iou
#                     best_match = obj_b
#                     best_idx_b = idx_b

#         if best_match:
#             matched_pairs.append((obj_a, best_match))
#             used_b_indices.add(best_idx_b)
#             unmatched_a.remove(obj_a)
#             unmatched_b.remove(best_match)

#     return matched_pairs, unmatched_a, unmatched_b



# def evaluate_video_similarity(
#     final_cut_frames,
#     other_vid_frames,
#     final_cut_obj_detector,
#     other_obj_detector,
#     image_comparator,
#     asset_paths,
#     iou_thresh=0.7,
#     similarity_thresh=0.8
# ):
#     # replace by total_frames = min(other_vid_frames,final_cut_frames)
#     total_frames = 10

#     host_visible_final = 0
#     host_visible_other = 0
#     total_pairs = 0
    
#     asset_use_count = 0
#     unmatched_asset_use_count = 0

#     assets_used = set()

#     temp_dir_final = os.path.join("temp_cropped_final", str(uuid.uuid4()))
#     temp_dir_other = os.path.join("temp_cropped_other", str(uuid.uuid4()))
#     os.makedirs(temp_dir_final, exist_ok=True)
#     os.makedirs(temp_dir_other, exist_ok=True)

#     print("\n=== Video Evaluation Debug Logs ===\n")

#     for idx in tqdm(range(total_frames), desc="Detecting & Comparing"):
        
#         similar_pairs = 0
#         print(idx,len(final_cut_frames),len(other_vid_frames))
#         final_objs = final_cut_obj_detector.predict_and_crop(final_cut_frames[idx], temp_dir=temp_dir_final)
#         other_objs = other_obj_detector.predict_and_crop(other_vid_frames[idx], temp_dir=temp_dir_other)

#         # Host visibility check
#         final_has_host = any(obj['class'] == 'Host' for obj in final_objs)
#         other_has_host = any(obj['class'] == 'Host' for obj in other_objs)

#         host_visible_final += int(final_has_host)
#         host_visible_other += int(other_has_host)

#         print(f"\n[Frame {idx}]")
#         print(f"  Host in final cut: {final_has_host}, Host in other video: {other_has_host}")

#         # Pair objects
#         pairs, unmatched_final, unmatched_other = pair_objects_by_bbox(final_objs, other_objs, iou_thresh=iou_thresh)

#         # Similarity for paired objects
#         for f_obj, o_obj in pairs:
#             score = image_comparator.get_similarity_score(f_obj['cropped_image_path'], o_obj['cropped_image_path'])
#             if score >= similarity_thresh:
#                 for asset_path in asset_paths:
#                     if asset_path not in assets_used:
#                         score_2 = image_comparator.get_similarity_score(f_obj['cropped_image_path'], asset_path)
#                         if score_2 >= similarity_thresh:
#                             assets_used.add(asset_path)
#                 similar_pairs += 1
#             total_pairs += 1

#         for obj in unmatched_other:
#             for asset_path in asset_paths:
#                 if asset_path not in assets_used:
#                     score = image_comparator.get_similarity_score(obj['cropped_image_path'], asset_path)
#                     if score >= similarity_thresh:
#                         assets_used.add(asset_path)
#                         unmatched_asset_use_count += 1
#                         print(f"Unmatched object matched asset {asset_path} (score={score:.3f})")
#                         break

#         for obj in other_objs:
#             for asset_path in asset_paths:
#                 if asset_path not in assets_used:
#                     score = image_comparator.get_similarity_score(obj['cropped_image_path'], asset_path)
#                     if score >= similarity_thresh:
#                         assets_used.add(asset_path)
#                         asset_use_count += 1
#                         break
        
#         print(f'Asset Used : {assets_used}')
#         print(f'Similar Pairs found : {similar_pairs}')
#         print()



#     shutil.rmtree(temp_dir_final)
#     shutil.rmtree(temp_dir_other)

#     host_score = host_visible_other / max(1, host_visible_final)
#     pair_match_score = similar_pairs / max(1, total_pairs)
#     asset_use_ratio = len(assets_used) / max(1, len(asset_paths))
    
#     return host_score, pair_match_score, asset_use_ratio
