# run_space_definer.py

import cv2
import sys
from pathlib import Path


src_path = Path(__file__).parent / "src"
sys.path.append(str(src_path))

from config import Config
from parking_space_detector import ParkingSpaceDetector

REFERENCE_IMAGE_PATH = "C:/Users/mhdda/parking_system/data/PKLot/UFPR05/Sunny/2013-03-02/2013-03-02_06_45_00.jpg"

OUTPUT_JSON_FILE = "parking_spaces.json"

def main():
    print("--- Parking Space Definer ---")
    
    config = Config()
    detector = ParkingSpaceDetector(config)
    
    config.PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    
    image_path = Path(REFERENCE_IMAGE_PATH)
    if not image_path.exists():
        print(f"Error: Reference image not found at {REFERENCE_IMAGE_PATH}")
        return
        
    image = cv2.imread(str(image_path))
    if image is None:
        print(f"Error: Could not read image from {image_path}")
        return
        
    print(f"Loaded reference image: {REFERENCE_IMAGE_PATH}")
    
    # 2. Run the manual definition tool
    print("\nStarting manual definition tool...")
    spaces = detector.define_parking_spaces_manual(image)
    
    if not spaces:
        print("No spaces were defined. Exiting.")
        return
        
    # 3. Save the defined spaces
    print(f"\nDefined {len(spaces)} spaces.")
    detector.save_parking_spaces(spaces, OUTPUT_JSON_FILE)
    print(f"Successfully saved parking spaces to data/processed/{OUTPUT_JSON_FILE}")

if __name__ == "__main__":
    main()