import cv2
import numpy as np
import json
from typing import List, Tuple, Dict

class ParkingSpaceDetector:
    
    def __init__(self, config):
        self.config = config
        self.parking_spaces = []
        
    
    def define_parking_spaces_manual(self, image: np.ndarray) -> List[Dict]:
        
        spaces = []
        current_points = []
        
        def mouse_callback(event, x, y, flags, param):
            nonlocal current_points
            if event == cv2.EVENT_LBUTTONDOWN:
                current_points.append((x, y))
                cv2.circle(img_copy, (x, y), 5, (0, 255, 0), -1)
                cv2.imshow("Define Parking Spaces", img_copy)
                
                if len(current_points) == 4:
                    # Draw the parking space
                    pts = np.array(current_points, np.int32)
                    cv2.polylines(img_copy, [pts], True, (0, 255, 0), 2)
                    cv2.imshow("Define Parking Spaces", img_copy)
                    
                    spaces.append({
                        'id': len(spaces),
                        'corners': current_points.copy()
                    })
                    current_points = []
        
        img_copy = image.copy()
        cv2.imshow("Define Parking Spaces", img_copy)
        cv2.setMouseCallback("Define Parking Spaces", mouse_callback)
        
        print("Click 4 corners of each parking space")
        print("Press 's' to save, 'q' to quit")
        
        while True:
            key = cv2.waitKey(1) & 0xFF
            if key == ord('s'):
                break
            elif key == ord('q'):
                break
        
        cv2.destroyAllWindows()
        return spaces
    
    def save_parking_spaces(self, spaces: List[Dict], filename: str):
        
        filepath = self.config.PROCESSED_DIR / filename
        with open(filepath, 'w') as f:
            json.dump(spaces, f, indent=4)
    
    def load_parking_spaces(self, filename: str) -> List[Dict]:
        filepath = self.config.PROCESSED_DIR / filename
        with open(filepath, 'r') as f:
            return json.load(f)
    
    def extract_parking_roi(self, image: np.ndarray, 
                          space: Dict) -> np.ndarray:
        
        corners = np.array(space['corners'], dtype=np.float32)

        output_width_px = int(self.config.STANDARD_PARKING_WIDTH * self.config.ROI_PIXELS_PER_METER)
        output_height_px = int(self.config.STANDARD_PARKING_LENGTH * self.config.ROI_PIXELS_PER_METER)

        dst_corners = np.array([
            [0, 0],
            [output_width_px - 1, 0],
            [output_width_px - 1, output_height_px - 1],
            [0, output_height_px - 1]
        ], dtype=np.float32)

        M = cv2.getPerspectiveTransform(corners, dst_corners)
        roi = cv2.warpPerspective(image, M, (output_width_px, output_height_px))

        return roi
