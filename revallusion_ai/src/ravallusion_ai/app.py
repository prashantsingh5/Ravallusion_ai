import streamlit as st
import os
from run import *

log_file_path = "video_similarity_log.log"
logging.basicConfig(
    filename=log_file_path,
    level=logging.INFO,
    format='%(asctime)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
st.set_page_config(page_title="Ravallusion Video Evaluator P2", layout="wide")

st.title("🎬 Ravallusion Video Evaluator P2")
col1, col2 = st.columns(2)
with col1:
    final_cut_video = st.file_uploader("Upload Final Cut Video", type=['mp4', 'mov', 'avi'])
with col2:
    assignment_video = st.file_uploader("Upload Assignment Video", type=['mp4', 'mov', 'avi'])

if final_cut_video or assignment_video:
    st.markdown("---")
    st.subheader("Video Previews")
    preview_col1, preview_col2 = st.columns(2)
    with preview_col1:
        if final_cut_video:
            st.video(final_cut_video)
            st.markdown(f"**Final Cut:** `{final_cut_video.name}`")
    with preview_col2:
        if assignment_video:
            st.video(assignment_video)
            st.markdown(f"**Assignment Video:** `{assignment_video.name}`")
    st.markdown("---")
if final_cut_video and assignment_video:
    st.success("Videos uploaded successfully! Click 'Run Evaluation' to proceed.")
    if st.button("Run Evaluation"):
        with st.spinner("Processing videos and calculating scores..."):
            try:
                if not os.path.exists("temp_uploads"):
                    os.makedirs("temp_uploads")

                final_cut_video_path = os.path.join("temp_uploads", final_cut_video.name)
                with open(final_cut_video_path, "wb") as f:
                    f.write(final_cut_video.getbuffer())

                assignment_video_path = os.path.join("temp_uploads", assignment_video.name)
                with open(assignment_video_path, "wb") as f:
                    f.write(assignment_video.getbuffer())


                test_dir = "/home/kishanm/Documents/DeepLearning-torch/Ravallusion-AI/test/"
                asset_paths = glob.glob(f"{os.path.join(test_dir,'assets')}/*.png")
                final_cut_vid_duration, final_cut_no_of_frames, final_cut_vid_frames_save_path = extract_frames_from_video(final_cut_video_path, 'final_cut_video_frames',every=5)
                other_vid_duration, other_no_of_frames, other_vid_frames_save_path = extract_frames_from_video(assignment_video_path, 'other_video_frames',every=5)

                final_cut_frames = sorted(glob.glob(final_cut_vid_frames_save_path+'/*'),key=lambda x : int((x.split('/')[-1]).split('.')[0]))
                other_vid_frames = sorted(glob.glob(other_vid_frames_save_path+'/*'),key=lambda x : int((x.split('/')[-1]).split('.')[0]))


                final_cut_obj_detector = ObjectsDetector(save_dir="final_cut_video",debug=True)
                other_obj_detector = ObjectsDetector(save_dir="other_video",debug=True)
                image_comparator = ImageComparator()

                video_acceptance_results = evaluate_video_similarity(final_cut_frames, other_vid_frames, final_cut_obj_detector, other_obj_detector, image_comparator, asset_paths)
                response = evaluate_video(assignment_video_path)
                asgn_video_score, asgn_video_feedback = response.parsed.total_score, response.parsed.feedback
                audio_similarity_score = evaluate_audio_similarity(final_cut_video_path, assignment_video_path) 
             

                st.success("Evaluation complete! Here are the results:")

                st.markdown("---")
                
                st.subheader("📊 Summary Scores")

                col_summary1, col_summary2, col_summary3 = st.columns(3)
                with col_summary1:
                    st.metric(label="Overall Video Acceptance Score", value=f"{video_acceptance_results['overall_score']*100:.2f} %")
                with col_summary2:
                    st.metric(label="Graphics Animation Score", value=f"{asgn_video_score:.2f} %")
                with col_summary3:
                    st.metric(label="Audio Similarity Score", value=f"{audio_similarity_score:.2f} %")

                
                with st.expander("🔊 Audio Quality"):
                    st.markdown(f'**How similar the audio is from the final cut video:** {audio_similarity_score:.2f} %')
                
                with st.expander("🖼️ Asset Usage"):
                    st.markdown(f"**Number of assets used:** {len(video_acceptance_results['assets_used'])} / {len(asset_paths)}")
                    st.markdown(f"**Asset Presence Score:** {video_acceptance_results['asset_presence_score']*100:.2f} %")
                    st.markdown(f"**Accurate Placement Score:** {video_acceptance_results['accurate_placement_score']*100:.2f} %")
                
                with st.expander("🧍 Subject Appearance"):
                    st.markdown(f"**How many times the host's face appeared in the video:** {video_acceptance_results['host_score']*100:.2f} %")
                
                with st.expander("🎨 Design Element"):
                    st.markdown(f"**Font used is good with a score of:** {video_acceptance_results['text_score']*100:.2f} %")
                
                with st.expander("🎬 Visual Editing"):
                    st.markdown('**How were the graphics and animation in the video?**')
                    st.markdown(f"**Visual Similarity Score:** {asgn_video_score:.2f} %")
                
                st.markdown("---")
                st.subheader("📝 Assignment Video Feedback")
                st.write(asgn_video_feedback)

                st.markdown("---")
                st.subheader('Assets used :')
                for itr,i in enumerate(video_acceptance_results['assets_used']):
                    st.write(f"{itr} : {i}")

                
            except Exception as e:
                st.error(f"An error occurred during evaluation: {e}")

