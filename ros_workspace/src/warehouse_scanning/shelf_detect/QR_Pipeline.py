import cv2
import time
import threading
from queue import Queue, Empty
from ultralytics import YOLO
from concurrent.futures import ThreadPoolExecutor
from zxingcpp import read_barcode
import json


# Handles all camera-related operations: open, read frames, close, etc.
# This keeps camera logic separate from scanning and processing logic
class CameraHandler:
    def __init__(self, cam_index=0, resolution=(640, 480), exposure=-5):

        self.cam_index = cam_index  # Which camera device is used
        self.resolution = resolution  # image resolution
        self.exposure = exposure  # Manual exposure control
        self.cap = None

    # Open camera and configure parameters
    def open(self):
        self.cap = cv2.VideoCapture(self.cam_index)

        # Set resolution
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.resolution[0])
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.resolution[1])

        # Set Manual exposure mode
        self.cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, 0.25)
        self.cap.set(cv2.CAP_PROP_EXPOSURE, float(self.exposure))

        # This Reduce internal buffering delay
        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

    # Read a frame (image) from camera
    # ret means success or failure
    def read(self):
        ret, frame = self.cap.read()
        return ret, frame

    # Release camera
    def close(self):
        if self.cap:
            self.cap.release()



# QRProcessor
# Responsible for:
# - Running YOLO detection on each frame
# - Crop out the regions where qr exists
# - Image preprocessing before decoding
# - Running ZXing decoding (multi-threaded)
# Stores list of unique QR codes detected across all batches.
class QRProcessor:
    def __init__(self, model_path="qr_model.pt", max_threads=3, pad=12):

        # Load YOLO model for qr detection
        self.model = YOLO(model_path)

        # Number of threads for decoding each crop
        self.max_threads = max_threads

        # Padding added around YOLO detection box (helps QR decode)
        self.pad = pad

        # Store QR codes so we don’t decode the same QR repeatedly
        self.unique_qrs = set()



    # Preprocess cropped region to improve ZXing decoding
    # We return two processed images: enhanced grayscale and thresholded
    # ZXing might decode one but not the other

    def preprocess_crop(self, crop):
        gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
        gray = cv2.medianBlur(gray, 3)  # Reduce noise

        # CLAHE helps boost local contrast (useful in low lighting)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        enhanced = clahe.apply(gray)

        # Adaptive threshold to create binary version
        bw = cv2.adaptiveThreshold(
            enhanced, 255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY,
            11, 2
        )
        return [enhanced, bw]


    # start decoding and Returns the decoded data
    def decode(self, crop):
        processed_crops = self.preprocess_crop(crop)

        # Try multiple preprocessed versions
        for img in processed_crops:
            result = read_barcode(img)
            if result and result.text:
                return result.text

        return None


    # Decode all crops in parallel using thread pool to speeds up scanning when multiple QRs are detected
    def parallel_decode(self, crops):
        with ThreadPoolExecutor(max_workers=self.max_threads) as exec:
            return list(exec.map(self.decode, crops))


    # Extract padded QR regions from YOLO results
    def extract_crops(self, frame, detections):
        crops, coords = [], []

        for box in detections[0].boxes:
            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)

            # Apply padding safely within frame boundaries
            x1 = max(0, x1 - self.pad)
            y1 = max(0, y1 - self.pad)
            x2 = min(frame.shape[1], x2 + self.pad)
            y2 = min(frame.shape[0], y2 + self.pad)

            crop = frame[y1:y2, x1:x2]

            # Skip tiny crops (cannot hold a QR)
            if crop.shape[0] >= 40 and crop.shape[1] >= 40:
                crops.append(crop)
                coords.append((x1, y1, x2, y2))

        return crops, coords




    # Process all frames in a batch:
    # - Run YOLO on each frame
    # - Extract crops
    # - Decode using parallel ZXing
    # - Save any new QR

    def process_frame_batch(self, frame_batch):
        for idx, frame in enumerate(frame_batch):

            # YOLO detection
            results = self.model(frame, verbose=False)

            if not results[0].boxes:
                continue  # No QR found in this frame

            crops, coords = self.extract_crops(frame, results)
            if not crops:
                continue

            # Decode all cropped qr
            decoded_list = self.parallel_decode(crops)

            # Take the first successful decode per frame
            for c_idx, qr_data in enumerate(decoded_list):
                if qr_data:  # QR detected
                    if qr_data not in self.unique_qrs:
                        print(f"NEW QR FOUND: {qr_data}")
                        self.unique_qrs.add(qr_data)

                        # Save crop image for verification/debug
                        x1, y1, x2, y2 = coords[c_idx]
                        crop_img = frame[y1:y2, x1:x2]
                        cv2.imwrite(f"found_{qr_data}.jpg", crop_img)


                        try:
                            with open("qr_log.jsonl", "a") as json_file:
                                json.dump({"qr": qr_data}, json_file)
                                json_file.write("\n")  # newline for JSONL format
                        except Exception as e:
                            print(f"Unable to write QR to json file: {e}")

        # Print summary after each batch
        print(f"[Processor] Unique QR count: {len(self.unique_qrs)}")
        print(f"[Processor] Unique QR list: {list(self.unique_qrs)}\n")




# CaptureThread -- works as Producer
# Waits for a trigger from robot -> captures N frames -> sends batch to queue
# This thread is not processes frames, it only produces new data and add to queue for further processing
class CaptureThread(threading.Thread):
    def __init__(self, camera, queue, frames_per_stop=10):
        super().__init__()
        self.camera = camera
        self.queue = queue
        self.frames_per_stop = frames_per_stop

        self.trigger = False  # When True -> capture next batch
        self._alive = True  # Thread life control

    # Called by ros command according to position of camera on vertical height
    def start_capture(self):
        self.trigger = True

    # Safe thread shutdown
    def stop_thread(self):
        self._alive = False

    # Main thread loop
    def run(self):
        while self._alive:

            # Only capture when robot requests it
            if self.trigger:
                batch = []

                # Capture a fixed number of frames when camera stop moving
                for _ in range(self.frames_per_stop):
                    ret, frame = self.camera.read()
                    if ret:
                        batch.append(frame)

                    # Small sleep helps camera provide fresh frames
                    time.sleep(0.015)


                # Send batch to processing thread to queue
                self.queue.put(batch)

                # Reset trigger until next stop event
                self.trigger = False

            # this sleep to avoid busy CPU
            time.sleep(0.005)




# ProcessingThread -- works as Consumer
# Continuously waits for batches in queue -> processes them -> updates QR list
# Automatically runs in background while robot moves
class ProcessingThread(threading.Thread):
    def __init__(self, processor, queue):
        super().__init__()
        self.processor = processor
        self.queue = queue
        self._alive = True

    # Called when shutting down system
    def stop_thread(self):
        self._alive = False
        self.queue.put(None)  # None value to break thread loop

    # Thread loop
    def run(self):
        while self._alive:
            try:
                # Wait for new batch
                batch = self.queue.get(timeout=0.5)

            except Empty:
                continue  # Continues if no batch is in queue

            if batch is None:
                break  # Shutdown signal

            # Run detection + decoding for this batch
            self.processor.process_frame_batch(batch)

            self.queue.task_done()


# ScannerController
# - Starts camera and threads
# - Manages system start/stop
class ScannerController:
    def __init__(self):
        self.camera = CameraHandler()
        self.processor = QRProcessor("qr_model.pt")

        # Queue between capture and processing threads
        self.queue = Queue(maxsize=3)

        # Worker threads
        self.capture_thread = CaptureThread(self.camera, self.queue)
        self.processing_thread = ProcessingThread(self.processor, self.queue)



    # Calls by ROS according to position or state of camera
    def Start_Camera(self):

        # When Robot reached TOP/MID/BOTTOM -> capture frames.
        self.capture_thread.start_capture()

    def robot_started_moving(self):

        # Robot starts moving. Processing happens automatically.
        # This function exists for future expansion (if needed)

        pass

    # System Lifecycle
    def start(self):

        # Open camera and launch worker threads
        self.camera.open()
        self.capture_thread.start()
        self.processing_thread.start()

    def stop(self):

        # Stop threads safely
        self.capture_thread.stop_thread()
        self.processing_thread.stop_thread()


        # Wait for threads to exit properly
        self.capture_thread.join()
        self.processing_thread.join()


        # Close the camera
        self.camera.close()


    # Demo function for testing only
    def demo_sequence(self):
        self.start()

        # TOP STOP
        self.Start_Camera()
        time.sleep(2)  # Simulate robot moving

        self.robot_started_moving()

        # MID STOP
        self.Start_Camera()
        time.sleep(2)
        self.robot_started_moving()

        # BOTTOM STOP
        self.Start_Camera()
        time.sleep(2)

        print("\nDetected QRs:", self.processor.unique_qrs)
        self.stop()




if __name__ == "__main__":
    controller = ScannerController()
    controller.demo_sequence()
