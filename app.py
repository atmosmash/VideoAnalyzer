import cv2
import numpy as np
import pandas as pd
import streamlit as st
from collections import Counter
from moviepy.editor import VideoFileClip, AudioFileClip

# Streamlit UI
st.title("Video Analyzer")

acknowledgment = st.checkbox("I, the undersigned RSM, hereby authorize PUB.Co to utilize my video for any media-related activities on the website.")

if acknowledgment:
    st.write("The guidelines are:")
    
    st.write("### Primary Guidelines:")
    st.write("1. The car should be black.")
    st.write("2. The car should create smoke.")
    
    st.write("### Environmental Guidelines:")
    st.write("3. The grass should be green.")
    st.write("4. The road should be gray in color.")
    
    guideline_choice = st.multiselect("Select guideline(s) for validation:", ["Primary Guidelines", "Environmental Guidelines"], default=["Primary Guidelines"])
    
    uploaded_file = st.file_uploader("Upload MP4 file", type=["mp4"])
    if uploaded_file is not None:
        video_path = "uploaded_video.mp4"
        with open(video_path, "wb") as f:
            f.write(uploaded_file.read())
        
        # Function to check if a frame meets the specified rules
        def check_rule(frame, guideline_choice):
            height, width, _ = frame.shape
            rule_statuses = {}
            
            if "Primary Guidelines" in guideline_choice:
                # Rule 1: Check if the car is black
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                _, thresh = cv2.threshold(gray, 50, 255, cv2.THRESH_BINARY_INV)
                contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                car_detected = any(cv2.contourArea(cnt) > 5000 for cnt in contours)
                rule_statuses["Rule 1"] = "Followed" if car_detected else "Violated"
                
                # Rule 2: Check if the car is creating smoke
                smoke_region = frame[height//2:, :]
                smoke_pixels = np.count_nonzero((smoke_region > 200).all(axis=2))
                rule_statuses["Rule 2"] = "Followed" if smoke_pixels >= 500 else "Violated"
                
            if "Environmental Guidelines" in guideline_choice:
                # Rule 3: Check if the grass in the background is green
                hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
                lower_green = np.array([35, 40, 40])
                upper_green = np.array([85, 255, 255])
                green_mask = cv2.inRange(hsv, lower_green, upper_green)
                green_pixels = np.count_nonzero(green_mask)
                rule_statuses["Rule 3"] = "Followed" if green_pixels > 1000 else "Violated"
                
                # Rule 4: Check if the road is gray
                road_region = frame[height//2:, :width//2]  # Checking bottom center for road color
                gray_pixels = np.count_nonzero((road_region > 100).all(axis=2))
                rule_statuses["Rule 4"] = "Followed" if gray_pixels > 5000 else "Violated"
                
            return rule_statuses
        
        cap = cv2.VideoCapture(video_path)
        results = {rule: {} for rule in ("Rule 1", "Rule 2", "Rule 3", "Rule 4")}
        frame_no = 0
        fps = cap.get(cv2.CAP_PROP_FPS)
        
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            
            time_sec = int(frame_no / fps)
            rule_statuses = check_rule(frame, guideline_choice)
            
            for rule, status in rule_statuses.items():
                if time_sec not in results[rule]:
                    results[rule][time_sec] = []
                results[rule][time_sec].append(status)
            
            frame_no += 1
        
        cap.release()
        
        final_results = [["Time (mm:ss)", "Violated Rules"]]
        for sec in results["Rule 1"]:
            time_formatted = f"{sec // 60:02}:{sec % 60:02}"
            violated_rules = [rule for rule in results if results[rule].get(sec) and Counter(results[rule][sec]).most_common(1)[0][0] == "Violated"]
            if violated_rules:
                final_results.append([time_formatted, ", ".join(violated_rules)])
        
        if len(final_results) > 1:
            results_df = pd.DataFrame(final_results[1:], columns=final_results[0])
            st.write("### Video is not validated since violations are found:")
            st.dataframe(results_df)
        else:
            st.write("### No violations found, video is validated.")
        
        # Step 3: Ask user if they wish to merge audio
        audio_choice = st.radio("Select an audio file to merge:", ("Audio 1", "Audio 2", "Audio 3"))
        
        if st.button("Proceed with Merging Audio"):
            audio_map = {"Audio 1": "audio1.mp3", "Audio 2": "audio2.mp3", "Audio 3": "audio3.mp3"}
            audio_path = audio_map[audio_choice]
            
            video_clip = VideoFileClip(video_path)
            audio_clip = AudioFileClip(audio_path)
            
            # Trim audio to match video duration
            if audio_clip.duration > video_clip.duration:
                audio_clip = audio_clip.subclip(0, video_clip.duration)
            
            video_clip = video_clip.set_audio(audio_clip)
            output_path = "merged_video.mp4"
            video_clip.write_videofile(output_path, codec="libx264", audio_codec="aac")
            
            st.write("### Merged Video Ready")
            st.video(output_path)