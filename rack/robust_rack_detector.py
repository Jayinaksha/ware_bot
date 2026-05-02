import cv2
import numpy as np
from ultralytics import YOLO
import argparse
import os

class RackResult:
    """Standardized output data for the robot control loop"""
    def __init__(self, detected=False, box=None, center_error_x=0.0, distance_factor=0.0):
        self.detected = detected
        self.box = box              # [x1, y1, x2, y2]
        self.center_error_x = center_error_x  # -1.0 (Left) to +1.0 (Right)
        self.distance_factor = distance_factor # Rough proxy for distance (box width)

class RobustRackDetector:
    def __init__(self, model_path, confidence=0.40):
        # 1. Load the AI Model
        print(f"Loading Rack Model: {model_path}")
        try:
            self.model = YOLO(model_path)
        except Exception as e:
            print(f"Error loading model: {e}")
            # For testing without model, we might want to pass, but for now raise
            raise
            
        self.conf_threshold = confidence
        
        # --- GABOR FILTER SETUP ---
        # We create a bank of filters to detect vertical and horizontal lines (wire mesh)
        self.filters = self.build_filters()
        self.TEXTURE_THRESHOLD = 30.0 # Tune this value based on testing

    def build_filters(self):
        """Creates a bank of Gabor filters for texture analysis"""
        filters = []
        ksize = 31  # Kernel size
        
        # We want to detect lines at different orientations
        # 0 = Vertical, 90 = Horizontal (approx)
        thetas = [0, np.pi/4, np.pi/2, 3*np.pi/4] 
        
        for theta in thetas:
            # params: (ksize, sigma, theta, lambda, gamma, psi, ktype)
            # sigma: standard deviation of the gaussian envelope
            # theta: orientation of the normal to the parallel stripes
            # lambda: wavelength of the sinusoidal factor
            # gamma: spatial aspect ratio
            # psi: phase offset
            kern = cv2.getGaborKernel((ksize, ksize), 4.0, theta, 10.0, 0.5, 0, ktype=cv2.CV_32F)
            kern /= 1.5*kern.sum()
            filters.append(kern)
        return filters

    def process_gabor(self, img, filters):
        """Applies the filter bank and returns the combined response"""
        accum = np.zeros_like(img)
        for kern in filters:
            fimg = cv2.filter2D(img, cv2.CV_8UC3, kern)
            np.maximum(accum, fimg, accum)
        return accum

    def get_alignment_error(self, frame_width, x1, x2):
        """Calculates how far off-center the rack is (-1.0 to 1.0)"""
        box_center = (x1 + x2) / 2.0
        frame_center = frame_width / 2.0
        return (box_center - frame_center) / (frame_width / 2.0)

    def validate_texture_gabor(self, frame, x1, y1, x2, y2):
        """
        The 'Ghost Buster' v2: Checks if the box contains a wire mesh structure using Gabor Filters.
        More robust to lighting than Canny.
        """
        # 1. Crop ROI
        roi = frame[y1:y2, x1:x2]
        roi_area = (x2 - x1) * (y2 - y1)
        if roi_area <= 0: return False

        # 2. Pre-process
        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        
        # 3. Apply Gabor Filters
        response = self.process_gabor(gray, self.filters)
        
        # 4. Calculate Energy (Mean response)
        mean_response = np.mean(response)
        std_response = np.std(response)
        
        # A rack should have high response (lots of lines) and high variance (lines vs holes)
        # print(f"Gabor Mean: {mean_response:.2f}, Std: {std_response:.2f}") # Debug
        
        if mean_response > self.TEXTURE_THRESHOLD:
            return True
        else:
            return False

    def process_frame(self, frame):
        """
        Main pipeline: YOLO -> Gabor Validation -> Alignment Calculation
        Returns: (annotated_frame, RackResult)
        """
        results = self.model(frame, verbose=False, conf=self.conf_threshold)
        frame_h, frame_w = frame.shape[:2]
        
        best_result = RackResult()
        
        if not results[0].boxes:
            return frame, best_result

        detected_racks = []

        # --- LOOP 1: Filter Candidates ---
        for box in results[0].boxes:
            coords = box.xyxy[0].cpu().numpy().astype(int)
            x1, y1 = max(0, coords[0]), max(0, coords[1])
            x2, y2 = min(frame_w, coords[2]), min(frame_h, coords[3])
            conf = float(box.conf[0].cpu().numpy())

            # 1. Run the Gabor Validation
            is_valid_rack = self.validate_texture_gabor(frame, x1, y1, x2, y2)

            if is_valid_rack:
                detected_racks.append((coords, conf))
                
                # Draw accepted boxes in Green
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(frame, f"Rack {conf:.2f}", (x1, y1 - 10), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            else:
                # Draw rejected boxes in Red (for debugging)
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 1)

        # --- LOOP 2: Select Best Target ---
        if detected_racks:
            detected_racks.sort(key=lambda x: x[1], reverse=True)
            best_coords, best_conf = detected_racks[0]
            
            x1, y1, x2, y2 = best_coords
            
            # Calculate Navigation Data
            error_x = self.get_alignment_error(frame_w, x1, x2)
            width_px = x2 - x1
            
            best_result = RackResult(True, best_coords, error_x, width_px)
            
            # Draw Target Sight
            center_x = int((x1 + x2) / 2)
            center_y = int((y1 + y2) / 2)
            cv2.circle(frame, (center_x, center_y), 5, (0, 255, 255), -1)
            cv2.putText(frame, f"Err: {error_x:.2f}", (x1, y2+20), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

        return frame, best_result

def main():
    parser = argparse.ArgumentParser(description="Robust Rack Detector Test (Gabor)")
    parser.add_argument("--image", type=str, required=True, help="Path to test image")
    parser.add_argument("--model", type=str, default="yolo11n.pt", help="Path to YOLO model")
    parser.add_argument("--output", type=str, default="output.jpg", help="Path to save output image")
    args = parser.parse_args()

    if not os.path.exists(args.image):
        print(f"Error: Image not found at {args.image}")
        return

    try:
        detector = RobustRackDetector(args.model)
        frame = cv2.imread(args.image)
        if frame is None:
            print("Error: Could not read image")
            return

        annotated_frame, result = detector.process_frame(frame)
        
        cv2.imwrite(args.output, annotated_frame)
        print(f"Result saved to {args.output}")
        print(f"Detected: {result.detected}")
        if result.detected:
            print(f"Error X: {result.center_error_x:.2f}")
            print(f"Distance Factor: {result.distance_factor}")

    except Exception as e:
        print(f"Execution failed: {e}")

if __name__ == "__main__":
    main()
