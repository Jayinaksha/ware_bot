import cv2
import numpy as np
import argparse
import os
import json

def decode_qr(image_path):
    """
    Decodes a QR code from an image.
    Returns: (decoded_text, points, straight_qrcode)
    """
    if not os.path.exists(image_path):
        print(f"Error: Image not found at {image_path}")
        return None, None, None

    img = cv2.imread(image_path)
    if img is None:
        print(f"Error: Could not read image {image_path}")
        return None, None, None

    # Initialize the QRCode detector
    detector = cv2.QRCodeDetector()
    
    # Detect and decode
    data, points, straight_qrcode = detector.detectAndDecode(img)

    if data:
        print(f"QR Code detected: {data}")
        return data, points, img
    else:
        print("No QR code detected")
        return None, None, img

def main():
    parser = argparse.ArgumentParser(description="QR Code Decoder")
    parser.add_argument("--image", type=str, required=True, help="Path to image with QR code")
    parser.add_argument("--output", type=str, default="qr_output.jpg", help="Path to save annotated image")
    args = parser.parse_args()

    data, points, img = decode_qr(args.image)

    if data:
        # Draw bounding box
        if points is not None:
            points = points[0].astype(int)
            for i in range(len(points)):
                pt1 = tuple(points[i])
                pt2 = tuple(points[(i+1) % len(points)])
                cv2.line(img, pt1, pt2, (0, 255, 0), 3)
        
        # Draw text
        cv2.putText(img, data, (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        
        cv2.imwrite(args.output, img)
        print(f"Annotated image saved to {args.output}")
        
        # Validation logic for the competition
        # Format: RACKID_SHELFID_ITEMCODE (e.g., R03_S2_ITM430)
        parts = data.split('_')
        if len(parts) == 3:
            print("Format Valid: YES")
            print(f"Rack: {parts[0]}")
            print(f"Shelf: {parts[1]}")
            print(f"Item: {parts[2]}")
        else:
            print("Format Valid: NO")

if __name__ == "__main__":
    main()
