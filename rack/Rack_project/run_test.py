import cv2
import os
import sys
from tqdm import tqdm
from RobustRackDetector import RobustRackDetector

# --- CONFIGURATION ---
INPUT_FOLDER = "dataset"   # Where your images are
OUTPUT_FOLDER = "results"  # Where results go
MODEL_NAME = "Rack_Detection_model_6.pt"  # It will auto-download
TEXTURE_THRESHOLD = 25.0   # Adjust this if detection is bad!

def main():
    # 1. Setup
    if not os.path.exists(INPUT_FOLDER):
        print(f"ERROR: No '{INPUT_FOLDER}' folder found!")
        return
    
    # 2. Initialize Detector
    detector = RobustRackDetector(model_path=MODEL_NAME, texture_thresh=TEXTURE_THRESHOLD)
    
    # 3. Get Images
    valid_ext = ['.jpg', '.jpeg', '.png']
    files = [f for f in os.listdir(INPUT_FOLDER) if os.path.splitext(f)[1].lower() in valid_ext]
    
    print(f"Found {len(files)} images. Processing...")

    # 4. Loop through every image
    for file_name in tqdm(files):
        img_path = os.path.join(INPUT_FOLDER, file_name)
        frame = cv2.imread(img_path)
        
        if frame is None: continue
        
        # RUN THE LOGIC
        annotated_frame, result = detector.process_frame(frame)
        
        # Save Result
        save_path = os.path.join(OUTPUT_FOLDER, "res_" + file_name)
        cv2.imwrite(save_path, annotated_frame)

    print("\nDone! Go check the 'results' folder.")

if __name__ == "__main__":
    main()