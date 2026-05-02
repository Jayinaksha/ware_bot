import cv2
import os
import argparse
from tqdm import tqdm 
from robust_rack_detector import RobustRackDetector 

def test_dataset(dataset_path, model_path, output_path):
    # 1. Setup
    if not os.path.exists(output_path):
        os.makedirs(output_path)
    
    # 2. Initialize your Detector
    print("Initializing Gabor Detector...")
    try:
        detector = RobustRackDetector(model_path=model_path)
    except Exception as e:
        print(f"Failed to initialize detector: {e}")
        return
    
    # 3. Get Images
    valid_ext = ['.jpg', '.png', '.jpeg']
    try:
        images = [f for f in os.listdir(dataset_path) if os.path.splitext(f)[1].lower() in valid_ext]
    except FileNotFoundError:
        print(f"Error: Dataset path not found: {dataset_path}")
        return

    print(f"Found {len(images)} images in {dataset_path}")

    # 4. Loop through images
    results_log = []
    
    for img_name in tqdm(images):
        full_path = os.path.join(dataset_path, img_name)
        frame = cv2.imread(full_path)
        
        if frame is None:
            continue
            
        # --- THE CORE LOGIC ---
        annotated_frame, result = detector.process_frame(frame)
        # ----------------------
        
        # Save Result
        save_name = os.path.join(output_path, f"res_{img_name}")
        
        # Optional: Add extra debug text for the Gabor response
        status = "DETECTED" if result.detected else "REJECTED"
        color = (0, 255, 0) if result.detected else (0, 0, 255)
        cv2.putText(annotated_frame, f"Status: {status}", (10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)
        
        cv2.imwrite(save_name, annotated_frame)
        results_log.append(f"{img_name}: {status} (Err: {result.center_error_x:.2f})")

    # 5. Summary
    print("\n--- Summary ---")
    print(f"Check the '{output_path}' folder to see the Gabor filter results.")
    with open(os.path.join(output_path, "log.txt"), "w") as f:
        f.write("\n".join(results_log))

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=str, required=True, help="Folder containing raw images")
    parser.add_argument("--output", type=str, required=True, help="Folder to save results")
    parser.add_argument("--model", type=str, default="yolo11n.pt", help="Path to YOLO model")
    args = parser.parse_args()
    
    test_dataset(args.input, args.model, args.output)
