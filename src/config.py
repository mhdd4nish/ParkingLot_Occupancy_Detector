import os
from pathlib import Path

class Config:
    # Paths
    BASE_DIR = Path(__file__).parent.parent
    DATA_DIR = BASE_DIR / "data"
    PKLOT_DIR = DATA_DIR / "PKLot" / "UFPR05"
    PROCESSED_DIR = DATA_DIR / "processed"
    RESULTS_DIR = DATA_DIR / "results"
    MODELS_DIR = BASE_DIR / "models"
    
    # Parking space dimensions (standard in meters)
    STANDARD_PARKING_WIDTH = 2.7  # meters
    STANDARD_PARKING_LENGTH = 5.0  # meters
    REQUIRED_CLEARANCE = 0.15  # 15cm on each side
    
    ROI_PIXELS_PER_METER = 50
    # Image processing parameters
    CANNY_THRESHOLD1 = 50
    CANNY_THRESHOLD2 = 150
    HOUGH_THRESHOLD = 100
    HOUGH_MIN_LINE_LENGTH = 50
    HOUGH_MAX_LINE_GAP = 10
    
    # Feature extraction
    LBP_RADIUS = 3
    LBP_POINTS = 24
    HOG_ORIENTATIONS = 9
    HOG_PIXELS_PER_CELL = (8, 8)
    HOG_CELLS_PER_BLOCK = (2, 2)
    
    # Classification
    SVM_KERNEL = 'rbf'
    SVM_C = 1.0
    SVM_GAMMA = 'scale'
    TEST_SPLIT_RATIO = 0.2
    RANDOM_STATE = 42