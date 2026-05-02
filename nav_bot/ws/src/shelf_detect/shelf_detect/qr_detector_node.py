import rclpy
from rclpy.node import Node
from std_msgs.msg import String
import cv2
from ultralytics import YOLO

class QRNode(Node):
    def __init__(self):
        super().__init__('qr_detector_node')
        self.publisher_ = self.create_publisher(String, 'qr_data', 10)
        
        # Load your model
        self.model = YOLO("qr_model.pt") 
        self.cap = cv2.VideoCapture(0) # Camera Index 0
        
        # Timer to run loop
        self.timer = self.create_timer(0.1, self.timer_callback)

    def timer_callback(self):
        ret, frame = self.cap.read()
        if not ret:
            return

        # Run inference using your pipeline logic
        results = self.model.predict(frame, conf=0.5, verbose=False)
        
        for result in results:
            for box in result.boxes:
                # Assuming class 0 is QR_Code
                if int(box.cls[0]) == 0: 
                    # Simulating decoding logic (YOLO detects, standard lib decodes usually)
                    # If your model detects detection, we publish "Detected"
                    # Or integrate pyzbar here if the model only finds location
                    msg = String()
                    msg.data = "QR_DETECTED" 
                    self.publisher_.publish(msg)
                    self.get_logger().info("QR Code Detected!")

def main():
    rclpy.init()
    node = QRNode()
    rclpy.spin(node)
    rclpy.shutdown()

if __name__ == '__main__':
    main()