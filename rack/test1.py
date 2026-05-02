import cv2
import numpy as np
from ultralytics import YOLO

class RackResult:
    """Standardized output data for the robot control loop"""
    def __init__(self, detected=False, box=None, center_error_x=0.0, distance_factor=0.0):
        self.detected = detected
        self.box = box #(x1, y1, x2, y2)
        self.center_error_x = center_error_x
        self.distance_factor = distance_factor

class RobustRackDetector:
    def __init__(self, model_path, confidence=0.90):
        #Load the ai model
        print(f"Loading Rack Model: {model_path}")
        self.model = YOLO(model_path)
        self.conf_threshold = confidence

        # Tuning Parameters
        self.CANNY_LOW = 30
        self.CANNY_HIGH = 100
        #A rack must be at least 5% "wire mesh" pixels walls are usually <1%.
        self.EDGE_DENSITY_MIN = 0.05
        self.EDGE_COUNT_MIN = 30

    def get_alignment_error(self, frame_width, x1, x2):
        """Calculate how far off-center the rack is (-1.0 to 1.0)"""
        box_center = (x1+x2) / 2.0
        frame_center = frame_width / 2.0
        # Normalize: -1 (left edge) to ) (center) to +1 (right edge)
        return (box_center - frame_center) / (frame_width / 2.0)
    
    def validate_mesh_texture(self, frame, x1, y1, x2, y2):
        """The 'Ghost Buster': Checks if the box contains a wire mesh structure."""
        roi = frame[y1:y2, x1:x2]
        roi_area = (x2-x1) * (y2-y1)
        if roi_area <= 0: return False

        #Preprocess (Gray + Blur)
        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (3, 3), 1.0)

        edges = cv2.Canny(blurred, self.CANNY_LOW, self.CANNY_HIGH)

        # Morphological close (connects broken wire segments)
        kernel = cv2.getStructuringElement(cv2.MORPH_CLOSE, (2, 2))
        edges_closed = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel, iterations=1)

        #Calculate Density
        edge_pixels = cv2.countNonZero(edges_closed)
        density = edge_pixels / roi_area

        #the verdict is it complex enough to be a rack?
        if density > self.EDGE_DENSITY_MIN and edge_pixels > self.EDGE_COUNT_MIN:
            return True
        else: 
            return False
        
    def process_frame(self, frame):
        """main pipeline: YOLO -> Mesh Validation -> Alignment Calculation"""
        #Resturns (Annotated_frame, RackResutls)
        results = self.model(frame, verbose=False, conf=self.conf_threshold)
        frame_h, frame_w = frame.shape[:2]

        best_result = RackResult()
        highest_confidence = 0.0

        if not results[0].boxes:
            return frame, best_result  #No detections
        
        detected_racks = []

        #LOOP : filter candidates
        for box in results[0].boxes:
            coords = box.xyxy[0].cpu().numpy().astype(int) #x1, y1, x2, y2
            x1, y1 = max(0, coords[0]), max(0, coords[1])
            x2, y2 = min(frame_w-1, coords[2]), min(frame_h, coords[3])
            conf = float(box.conf[0].cpu().numpy())

            # Run the mesh validation 
            is_valid_rack = self.validate_mesh_texture(frame, x1, y1, x2, y2)

            if is_valid_rack:
                detected_racks.append((coords, conf))

                # Draw accepted boxes in green
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                
            ### LOOP 2 Select the best target
            # If we found multiple valid racks, pick the one with highest confidence
            if detected_racks:
                detected_racks.sort(key=lambda x: x[1], reverse=True)
                best_coords, best_conf = detected_racks[0]

                x1, y1, x2, y2 = best_coords
                error_x = self.get_alignment_error(frame_w, x1, x2)

                width_px = x2 - x1
                best_result = RackResult(True, best_coords, error_x, width_px)

                center_x = int((x1 + x2) / 2)
                center_y = int((y1 + y2) / 2)
                cv2.circle(frame, (center_x, center_y), 5, (0, 255, 255), -1)
                cv2.putText(frame, f"Err: {error_x:.2f}", (x1, y2+20),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
                
                return frame, best_result
        

