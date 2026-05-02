import cv2
import numpy as np
from ultralytics import YOLO

class RackResult:
    def __init__(self, detected=False, center_error_x=0.0):
        self.detected = detected
        self.center_error_x = center_error_x

class RobustRackDetector:
    def __init__(self, model_path='Rack_Detection_model_6.pt', confidence=0.65, texture_thresh=5.0):
        # 1. Load AI Model
        print(f"Loading YOLO Model: {model_path}...")
        self.model = YOLO(model_path)
        self.conf_threshold = confidence
        self.texture_threshold = texture_thresh
        
        # 2. Build Gabor Filters (The "Wire Mesh" Detector)
        self.filters = self._build_filters()

    def _build_filters(self):
        """Creates mathematical filters that glow when they see wire mesh."""
        filters = []
        ksize = 31
        # Look for lines at 0, 45, 90, 135 degrees
        for theta in [0, np.pi/4, np.pi/2, 3*np.pi/4]:
            kern = cv2.getGaborKernel((ksize, ksize), 4.0, theta, 10.0, 0.5, 0, ktype=cv2.CV_32F)
            kern /= 1.5*kern.sum()
            filters.append(kern)
        return filters

    def _validate_texture(self, frame, x1, y1, x2, y2):
        """Checks if the box area looks like a mesh or a flat wall."""
        roi = frame[y1:y2, x1:x2]
        if roi.size == 0: return False

        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        accum_response = np.zeros_like(gray, dtype=np.float32)
        
        # Apply filters
        for kern in self.filters:
            fimg = cv2.filter2D(gray, cv2.CV_32F, kern)
            np.maximum(accum_response, fimg, accum_response)
        
        # Calculate "Messiness" score
        mean_response = np.mean(accum_response)
        
        # Debug print (Optional: uncomment to see scores in terminal)
        # print(f"Texture Score: {mean_response:.2f}")
        
        return mean_response > self.texture_threshold

    def process_frame(self, frame):
        # 1. AI Detection
        results = self.model(frame, verbose=False, conf=self.conf_threshold)
        h, w = frame.shape[:2]
        best_result = RackResult()
        
        detected_racks = []

        if results[0].boxes:
            for box in results[0].boxes:
                coords = box.xyxy[0].cpu().numpy().astype(int)
                x1, y1, x2, y2 = coords
                conf = float(box.conf[0].cpu().numpy())

                # 2. The Texture Check
                is_valid = self._validate_texture(frame, x1, y1, x2, y2)

                if is_valid:
                    detected_racks.append((coords, conf))
                    # Draw GREEN box (Confirmed)
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    cv2.putText(frame, f"Rack {conf:.2f}", (x1, y1-10), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                else:
                    # Draw RED box (Rejected Wall/Box)
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
                    cv2.putText(frame, "Rejected", (x1, y1-10), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

        # 3. Pick the Best Rack for Navigation
        if detected_racks:
            # Sort by confidence
            detected_racks.sort(key=lambda x: x[1], reverse=True)
            (x1, y1, x2, y2), _ = detected_racks[0]
            
            center_x = (x1 + x2) / 2
            # Error: -1.0 (Left) to +1.0 (Right)
            best_result.center_error_x = (center_x - (w / 2)) / (w / 2)
            best_result.detected = True
            
            # Draw Target Circle
            cv2.circle(frame, (int(center_x), int((y1+y2)/2)), 8, (0, 255, 255), -1)

        return frame, best_result