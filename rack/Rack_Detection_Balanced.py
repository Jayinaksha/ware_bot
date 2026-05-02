import cv2
from ultralytics import YOLO
import numpy as np
import sys
import os
from pathlib import Path
from datetime import datetime


LAPLACIAN_THRESHOLD = 0.1
CONTRAST_THRESHOLD = 15.0


CANNY_LOW_THRESHOLD = 30
CANNY_HIGH_THRESHOLD = 100
EDGE_AREA_MIN_RATIO = 0.05  
EDGE_DENSITY_THRESHOLD = 30 


NMS_IOU_THRESHOLD = 0.1  
CONFIDENCE_THRESHOLD = 0.40 
YOLO_MODEL_PATH = r"C:\Users\admin\Documents\yolo11n.pt"
TEST_IMAGES_FOLDER = r"C:\Users\admin\Documents\test_images (2)\test_images"


OUTPUT_BASE_FOLDER = r"C:\Users\admin\Documents\rack_detection_results"

def create_output_directory():
    """Create timestamped output directory for results"""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = os.path.join(OUTPUT_BASE_FOLDER, f"detection_run_{timestamp}")
    
    os.makedirs(output_dir, exist_ok=True)
    
   
    os.makedirs(os.path.join(output_dir, "valid_detections"), exist_ok=True)
    os.makedirs(os.path.join(output_dir, "rejected_images"), exist_ok=True)
    os.makedirs(os.path.join(output_dir, "edge_maps"), exist_ok=True)
    
    return output_dir

def get_image_files(folder_path):
    """Get all image files from the specified folder"""
    valid_extensions = ('.jpg', '.jpeg', '.png', '.bmp', '.tiff')
    image_files = []
    
    try:
        for file in os.listdir(folder_path):
            if file.lower().endswith(valid_extensions):
                image_files.append(os.path.join(folder_path, file))
    except FileNotFoundError:
        print(f"ERROR: Folder not found: {folder_path}", file=sys.stderr)
        return []
    
    return sorted(image_files)

def load_models():
    """Load YOLO model with settings to detect multiple racks"""
    try:
        model = YOLO(YOLO_MODEL_PATH)
        
        model.overrides['conf'] = 0.35
        model.overrides['iou'] = 0.3
        return model
    except Exception as e:
        print(f"CRITICAL ERROR: Could not load '{YOLO_MODEL_PATH}'. {e}", file=sys.stderr)
        sys.exit(1)

def check_image_quality(frame):
    """Validate image quality using Laplacian and contrast scores"""
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    contrast_score = gray.std()
    
    denoised_gray = cv2.medianBlur(gray, 5)
    laplacian_score = cv2.Laplacian(denoised_gray, cv2.CV_64F).var()
    
    print(f"Image Quality Scores:")
    print(f"  Laplacian (Sharpness): {laplacian_score:.4f} (Threshold: >{LAPLACIAN_THRESHOLD})")
    print(f"  Contrast: {contrast_score:.2f} (Threshold: >{CONTRAST_THRESHOLD})")
    
    if contrast_score < CONTRAST_THRESHOLD:
        print("  ✗ REJECTED: Image has low contrast")
        return False, None
    else:
        print("  ✓ ACCEPTED: Image quality check passed")
        return True, gray

def calculate_iou(box1, box2):
    """Calculate Intersection over Union (IoU) between two bounding boxes"""
    x1_min, y1_min, x1_max, y1_max = box1
    x2_min, y2_min, x2_max, y2_max = box2
    
    inter_xmin = max(x1_min, x2_min)
    inter_ymin = max(y1_min, y2_min)
    inter_xmax = min(x1_max, x2_max)
    inter_ymax = min(y1_max, y2_max)
    
    if inter_xmax < inter_xmin or inter_ymax < inter_ymin:
        return 0.0
    
    inter_area = (inter_xmax - inter_xmin) * (inter_ymax - inter_ymin)
    
    box1_area = (x1_max - x1_min) * (y1_max - y1_min)
    box2_area = (x2_max - x2_min) * (y2_max - y2_min)
    union_area = box1_area + box2_area - inter_area
    
    return inter_area / union_area if union_area > 0 else 0.0

def apply_minimal_nms(boxes_with_conf):
    """
    Apply MINIMAL NMS - only merge EXTREME overlaps (>90%)
    Allows separate racks to be detected
    """
    if not boxes_with_conf:
        return []
    
   
    sorted_boxes = sorted(boxes_with_conf, key=lambda x: x[1], reverse=True)
    
    keep_boxes = []
    suppressed_count = 0
    
    for (box_i, conf_i, class_i, stats_i) in sorted_boxes:
        is_duplicate = False
        
        for (box_k, conf_k, class_k, stats_k) in keep_boxes:
            iou = calculate_iou(box_i, box_k)
            
           
            if iou > 0.90:
                is_duplicate = True
                suppressed_count += 1
                print(f"    ↪ Suppressed EXTREME duplicate (IoU: {iou:.3f}, conf: {conf_i:.2f})")
                break
        
        if not is_duplicate:
            keep_boxes.append((box_i, conf_i, class_i, stats_i))
            print(f"    ✓ Kept detection (conf: {conf_i:.2f})")
    
    print(f"\n  NMS Summary: Suppressed {suppressed_count} extreme overlaps")
    return keep_boxes

def detect_edges_in_roi(frame, x1, y1, x2, y2):
    """
    Detect edges within a Region of Interest (ROI)
    Optimized for wire mesh structures
    """
    
    roi = frame[y1:y2, x1:x2]
    roi_area = (x2 - x1) * (y2 - y1)
    
    
    if len(roi.shape) == 3:
        roi_gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    else:
        roi_gray = roi
    
   
    blurred = cv2.GaussianBlur(roi_gray, (3, 3), 1.0)
    
   
    edges = cv2.Canny(blurred, CANNY_LOW_THRESHOLD, CANNY_HIGH_THRESHOLD)
    
   
    kernel = cv2.getStructuringElement(cv2.MORPH_CLOSE, (2, 2))
    edges_closed = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel, iterations=1)
    
    
    edge_count = cv2.countNonZero(edges_closed)
    edge_percentage = (edge_count / roi_area) * 100 if roi_area > 0 else 0
    
    return edges_closed, edge_percentage, edge_count

def validate_rack_with_edges(frame, box_coords):
    """
    Validate detected rack using edge detection
    Filters false positives (wall frames, etc.)
    Returns: is_valid, edge_stats
    """
    x1, y1, x2, y2 = box_coords
    
   
    edges, edge_percentage, edge_count = detect_edges_in_roi(frame, x1, y1, x2, y2)
    
   
    roi_area = (x2 - x1) * (y2 - y1)
    min_edge_pixels = max(int(roi_area * EDGE_AREA_MIN_RATIO), EDGE_DENSITY_THRESHOLD)
    
   
    has_sufficient_edges = edge_count >= min_edge_pixels
    
    edge_stats = {
        'edge_count': edge_count,
        'edge_percentage': edge_percentage,
        'min_required': min_edge_pixels,
        'is_valid': has_sufficient_edges
    }
    
    print(f"  Edge Analysis:")
    print(f"    Edge Pixels: {edge_count} (Required: {min_edge_pixels})")
    print(f"    Edge Density: {edge_percentage:.2f}%")
    print(f"    Valid: {'✓ RACK' if has_sufficient_edges else '✗ FALSE POSITIVE'}")
    
    return has_sufficient_edges, edge_stats

def detect_racks(frame, model):
    """
    Detect racks using YOLO with edge validation to filter false positives
    Allows detection of multiple separate racks
    """
    
    results = model(frame, conf=0.35, verbose=False)
    
    if not results[0].boxes:
        print("No objects detected by YOLO")
        return []
    
    all_detections = []
    
    print(f"\nYOLO detected {len(results[0].boxes)} box(es)")
    
    for i, box in enumerate(results[0].boxes):
        coords = box.xyxy[0].cpu().numpy().astype(int)
        x1, y1 = max(0, coords[0]), max(0, coords[1])
        x2, y2 = min(frame.shape[1], coords[2]), min(frame.shape[0], coords[3])
        
        conf = float(box.conf[0].cpu().numpy())
        cls_id = int(box.cls[0].cpu().numpy())
        class_name = model.names[cls_id]
        
       
        if conf < CONFIDENCE_THRESHOLD:
            print(f"  Skipping low-confidence detection ({conf:.2f} < {CONFIDENCE_THRESHOLD})")
            continue
        
        print(f"\n--- Detection {i} ---")
        print(f"'{class_name}' (Confidence: {conf:.2f})")
        print(f"Location: [{x1}, {y1}, {x2}, {y2}]")
        print(f"Size: {x2-x1}x{y2-y1} pixels")
        
       
        is_valid_rack, edge_stats = validate_rack_with_edges(frame, (x1, y1, x2, y2))
        
        if is_valid_rack:
            all_detections.append(((x1, y1, x2, y2), conf, class_name, edge_stats))
            print(f"✓ Accepted: Real rack detected")
        else:
            print(f"✗ Rejected: Not a real rack")
    
    
    print(f"\n{'='*50}")
    print(f"Applying MINIMAL NMS (only extreme overlaps >90%)...")
    print(f"{'='*50}")
    
    filtered_detections = apply_minimal_nms(all_detections)
    
    
    found_results = []
    for (x1, y1, x2, y2), conf, class_name, edge_stats in filtered_detections:
        found_results.append({
            'text': f"{class_name}: {conf:.2f}",
            'box': (x1, y1, x2, y2),
            'class_name': class_name,
            'confidence': conf,
            'edge_stats': edge_stats
        })
    
    return found_results

def process_single_image(image_path, model, output_dir):
    """Process a single image and save results"""
    image_name = os.path.basename(image_path)
    print(f"\n{'='*70}")
    print(f"Processing: {image_name}")
    print(f"{'='*70}")
    
    try:
        frame = cv2.imread(image_path)
        if frame is None:
            print(f"ERROR: Could not read image: {image_path}")
            return None
    except Exception as e:
        print(f"ERROR reading image: {e}", file=sys.stderr)
        return None
    
   
    is_good, _ = check_image_quality(frame)
    
    if not is_good:
        rejected_path = os.path.join(output_dir, "rejected_images", image_name)
        cv2.imwrite(rejected_path, frame)
        print(f"Image rejected: {rejected_path}")
        return {
            'image': image_name,
            'status': 'REJECTED_QUALITY',
            'detections': 0
        }
    
   
    detection_results = detect_racks(frame, model)
    
    if detection_results:
        print(f"\n{'='*50}")
        print(f"FINAL RESULTS: {len(detection_results)} rack(s) detected ✓")
        print(f"{'='*50}")
        
       
        for idx, res in enumerate(detection_results, 1):
            (x1, y1, x2, y2) = res['box']
            label = res['text']
            
           
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            
            
            cv2.putText(frame, label, (x1, y1 - 10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            
           
            edge_badge = f"Edges: {res['edge_stats']['edge_count']}"
            cv2.putText(frame, edge_badge, (x1, y2 + 20),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 165, 255), 1)
            

            rack_num = f"Rack #{idx}"
            cv2.putText(frame, rack_num, (x1, y1 - 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        
      
        output_path = os.path.join(output_dir, "valid_detections", image_name)
        cv2.imwrite(output_path, frame)
        print(f"✓ Result saved: {output_path}")
        
        return {
            'image': image_name,
            'status': 'SUCCESS',
            'detections': len(detection_results)
        }
    else:
        print("\nNo racks detected")
        rejected_path = os.path.join(output_dir, "rejected_images", image_name)
        cv2.imwrite(rejected_path, frame)
        
        return {
            'image': image_name,
            'status': 'NO_DETECTIONS',
            'detections': 0
        }

def generate_summary_report(results, output_dir):
    """Generate a summary report of all processing results"""
    report_path = os.path.join(output_dir, "SUMMARY_REPORT.txt")
    
    with open(report_path, 'w') as f:
        f.write("="*70 + "\n")
        f.write("RACK DETECTION BATCH PROCESSING SUMMARY\n")
        f.write("="*70 + "\n\n")
        
        f.write("PARAMETERS:\n")
        f.write(f"  YOLO Confidence: {CONFIDENCE_THRESHOLD}\n")
        f.write(f"  Edge Validation: ENABLED (filters false positives)\n")
        f.write(f"  Edge Density Min: {EDGE_AREA_MIN_RATIO * 100:.1f}%\n")
        f.write(f"  Edge Pixel Threshold: {EDGE_DENSITY_THRESHOLD}\n")
        f.write(f"  NMS IoU Threshold: {NMS_IOU_THRESHOLD} (minimal)\n\n")
        
        f.write(f"Total Images Processed: {len(results)}\n")
        f.write(f"Successful Detections: {sum(1 for r in results if r['status'] == 'SUCCESS')}\n")
        f.write(f"Quality Rejections: {sum(1 for r in results if r['status'] == 'REJECTED_QUALITY')}\n")
        f.write(f"No Detections: {sum(1 for r in results if r['status'] == 'NO_DETECTIONS')}\n")
        f.write(f"Total Racks Detected: {sum(r.get('detections', 0) for r in results)}\n\n")
        
        f.write("DETAILED RESULTS:\n")
        f.write("-"*70 + "\n")
        
        for result in results:
            f.write(f"\nImage: {result['image']}\n")
            f.write(f"Status: {result['status']}\n")
            if result['detections'] > 0:
                f.write(f"Racks Detected: {result['detections']}\n")
            f.write("\n")
    
    print(f"\n✓ Summary report saved: {report_path}")

def main():
    print("="*70)
    print("RACK DETECTION - MULTIPLE RACK WITH FALSE POSITIVE FILTERING")
    print("="*70)
    print(f"Source: {TEST_IMAGES_FOLDER}\n")
    
    
    output_dir = create_output_directory()
    print(f"Output directory: {output_dir}\n")
    
   
    image_files = get_image_files(TEST_IMAGES_FOLDER)
    
    if not image_files:
        print(f"No images found in {TEST_IMAGES_FOLDER}")
        return
    
    print(f"Found {len(image_files)} image(s) to process\n")
    
   
    model = load_models()
    print("YOLO model loaded\n")
    
   
    results = []
    for idx, image_path in enumerate(image_files, 1):
        print(f"\n[{idx}/{len(image_files)}]")
        result = process_single_image(image_path, model, output_dir)
        if result:
            results.append(result)
    
   
    print("\n" + "="*70)
    generate_summary_report(results, output_dir)
    
    print("\n" + "="*70)
    print("BATCH PROCESSING COMPLETE")
    print(f"Results: {output_dir}")
    print("="*70)

if __name__ == "__main__":
    main()
