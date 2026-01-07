# src/main.py
import cv2
import numpy as np
from pathlib import Path
import xml.etree.ElementTree as ET

from config import Config
from parking_space_detector import ParkingSpaceDetector
from feature_extractor import FeatureExtractor
from occupancy_classifier import OccupancyClassifier
from width_estimator import WidthEstimator

class ParkingSystem:

    
    def __init__(self):
        self.config = Config()
        self.detector = ParkingSpaceDetector(self.config)
        self.extractor = FeatureExtractor(self.config)
        self.classifier = OccupancyClassifier(self.config)
        self.estimator = WidthEstimator(self.config)
        
        # Load the predefined parking spaces
        self.parking_spaces = self.detector.load_parking_spaces('parking_spaces.json')
        print(f"Loaded {len(self.parking_spaces)} parking spaces.")
        
    def process_single_image(self, image_path: str, visualize=True):
        
        if self.estimator is None:
            self._initialize_estimator()
            
        image = cv2.imread(image_path)
        if image is None:
            print(f"Error: Could not load image {image_path}")
            return [], None
            
        result_image = image.copy()
        results = []

        for space in self.parking_spaces:
            space_id = space['id']
            corners = np.array(space['corners'])
            
            # 1. Extract ROI
            roi = self.detector.extract_parking_roi(image, space)
            
            # 2. Extract Features
            features = self.extractor.extract_combined_features(roi)
            
            # 3. Classify Occupancy
            prediction, confidence = self.classifier.predict(features)
            occupancy_status = 'Occupied' if prediction == 1 else 'Empty'
            
            space_result = {
                'space_id': space_id,
                'occupancy': occupancy_status,
                'confidence': float(confidence),
                'parking_width': 0.0,
                'assessment': None
            }

            # 4. Measure Widths & Assess Fit
            parking_width_m = self.estimator.measure_parking_space_width(roi)
            space_result['parking_width'] = parking_width_m
            
            color = (0, 255, 0) 
            
            if occupancy_status == 'Occupied':
                color = (0, 0, 255) 
                
                # 5. Detect vehicle and measure width
                has_vehicle, mask, contour_area = self.estimator.detect_vehicle_in_space(roi)
                if has_vehicle:
                    roi_area = roi.shape[0] * roi.shape[1] # Total pixels in the ROI
                    occupancy_percentage = contour_area / roi_area

                    
                    MIN_OCCUPANCY_THRESHOLD = 0.3

                    if occupancy_percentage < MIN_OCCUPANCY_THRESHOLD:
                        
                        occupancy_status = "Empty"
                        space_result['occupancy'] = "Empty"
                        color = (0, 255, 0) 
                        has_vehicle = False 
            
                if has_vehicle:
                    assessment, offset = self.estimator.assess_centering(roi, mask)

                    space_result['assessment'] = assessment
                    
            
            results.append(space_result)

            # 7. Visualize
            if visualize:
                cv2.polylines(result_image, [corners.astype(np.int32)], True, color, 2)
                
                
                text_pos = (corners[0][0], corners[0][1] - 10)
                cv2.putText(result_image, f"{space_id}: {occupancy_status}", 
                            text_pos, cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
                
                if occupancy_status == 'Occupied' and space_result['assessment']:
                     cv2.putText(result_image, f"Fit: {space_result['assessment']['status']}", 
                            (text_pos[0], text_pos[1] - 15), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255,255,0), 1)

        return results, result_image

if __name__ == "__main__":

    BASE_PATH = "c:/Users/mhdda/parking_system/data/PKLot/UFPR05" #2013-03-20_17_05_12
    # C:/Users/mhdda/parking_system/data/PKLot/UFPR05/Sunny/2013-03-02/2013-03-02_06_45_00.jpg
    TEST_IMAGE_PATHS = [
        f"{BASE_PATH}/Sunny/2013-03-02/2013-03-02_06_45_00.jpg",
        f"{BASE_PATH}/Cloudy/2013-03-13/2013-03-13_07_25_01.jpg",
        f"{BASE_PATH}/Cloudy/2013-03-13/2013-03-13_07_30_01.jpg",
        f"{BASE_PATH}/Cloudy/2013-03-13/2013-03-13_10_50_05.jpg",
        f"{BASE_PATH}/Rainy/2013-03-05/2013-03-05_16_15_11.jpg" 
    ]
    
    MODEL_FILE = "c:/Users/mhdda/parking_system/models/svm_model.pkl"
    
    print("--- Starting Parking System (Slideshow Mode) ---")
    print("Press any key to advance to the next image. Press 'q' to quit.")
    
    system = ParkingSystem()
    
    try:
        system.classifier.load_model(MODEL_FILE)
    except FileNotFoundError:
        print(f"Error: Model file '{MODEL_FILE}' not found in 'models/' directory.")
        print("Please run the training script first.")
        exit()

    for i, image_path in enumerate(TEST_IMAGE_PATHS):
        
        print("\n" + "="*50)
        print(f"Processing image {i+1}/{len(TEST_IMAGE_PATHS)}: {image_path}")
        print("="*50)
        
        results, result_image = system.process_single_image(image_path, visualize=True)
        
        if result_image is None:
            print("Skipping this image.")
            continue
            
        for res in results:
            print(f"--- Space {res['space_id']} ---")
            print(f"  Status: {res['occupancy']} (Conf: {res['confidence']:.2f})")
            print(f"  Parking Width: {res['parking_width']:.2f}m")
            if res['occupancy'] == 'Occupied':
                print(f"  Assessment: {res['assessment']['status']}")
                if res['assessment']:
                    print(f"  Offset: {res['assessment']['offset_m']:.2f}m")


        display_image = result_image.copy()
        h, w = display_image.shape[:2]
        if h > 900: # If image height is > 900px, resize it
            scale = 900 / h
            display_image = cv2.resize(display_image, (int(w * scale), int(h * scale)))
            
        cv2.imshow("Parking Analysis Result", display_image)
        
        image_name = Path(image_path).stem
        output_path = f"data/results/result_{image_name}.jpg"
        cv2.imwrite(output_path, result_image)
        print(f"\nResult image saved to {output_path}")
        
        print("\nPress any key to load next image, or 'q' to quit...")
        key = cv2.waitKey(0)
        if key == ord('q'):
            break
    
    cv2.destroyAllWindows()
    print("Slideshow finished.")