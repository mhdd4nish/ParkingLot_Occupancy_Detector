# run_training.py

import cv2
import sys
import numpy as np
from pathlib import Path
from tqdm import tqdm
import time
import random

src_path = Path(__file__).parent / "src"
sys.path.append(str(src_path))

from config import Config
from feature_extractor import FeatureExtractor
from occupancy_classifier import OccupancyClassifier


SEGMENTED_DATA_DIR = "C:/Users/mhdda/parking_system/data/PKLot_segmented/UFPR05"

MODEL_OUTPUT_FILE = "svm_model.pkl"


MAX_SAMPLES_PER_CLASS = 12000 

def load_dataset(base_path: Path, extractor: FeatureExtractor):
    
    features_list = []
    labels_list = []


    empty_paths = list(base_path.glob("**/Empty/*.jpg"))
    occupied_paths = list(base_path.glob("**/Occupied/*.jpg"))
    
    if not empty_paths or not occupied_paths:
        print(f"Error: Dataset not found at {base_path}")
        print("Please check the SEGMENTED_DATA_DIR path.")
        return None, None
        
    random.shuffle(empty_paths)
    random.shuffle(occupied_paths)
    if MAX_SAMPLES_PER_CLASS:
        empty_paths = empty_paths[:MAX_SAMPLES_PER_CLASS]
        occupied_paths = occupied_paths[:MAX_SAMPLES_PER_CLASS]

    print(f"Loading {len(empty_paths)} 'Empty' samples...")
    for img_path in tqdm(empty_paths, desc="Processing Empty"):
        img = cv2.imread(str(img_path))
        if img is not None:
            features = extractor.extract_combined_features(img)
            features_list.append(features)
            labels_list.append(0) # 0 for Empty

    print(f"Loading {len(occupied_paths)} 'Occupied' samples...")
    for img_path in tqdm(occupied_paths, desc="Processing Occupied"):
        img = cv2.imread(str(img_path))
        if img is not None:
            features = extractor.extract_combined_features(img)
            features_list.append(features)
            labels_list.append(1) # 1 for Occupied
            
    return np.array(features_list), np.array(labels_list)

def main():
    print("--- Occupancy Classifier Trainer ---")
    start_time = time.time()
    
    config = Config()
    extractor = FeatureExtractor(config)
    
    config.MODELS_DIR.mkdir(parents=True, exist_ok=True)
    
    # 1. Load Dataset
    print("Loading dataset...")
    data_path = Path(SEGMENTED_DATA_DIR)
    X, y = load_dataset(data_path, extractor)
    
    if X is None:
        return
        
    print(f"\nDataset loaded. Total samples: {len(X)}")
    print(f"Feature vector size: {X.shape[1]}")
    
    print("\nPreparing dataset for training (splitting and scaling)...")
    classifier = OccupancyClassifier(config, classifier_type='svm')
    
    X_train, X_test, y_train, y_test = classifier.prepare_dataset(X, y)
    
    print(f"Training samples: {len(X_train)}")
    print(f"Test samples: {len(X_test)}")
    
    # 3. Train the Model
    print("\n--- Training SVM Model ---")
    classifier.train(X_train, y_train)
    print("--- Training Complete ---")
    
    # 4. Evaluate the Model
    print("\n--- Evaluating Model on Test Set ---")
    classifier.evaluate(X_test, y_test)
    
    # 5. Save the Final Model

    print(f"\nSaving model to {config.MODELS_DIR / MODEL_OUTPUT_FILE}...")
    classifier.save_model(MODEL_OUTPUT_FILE)
    
    end_time = time.time()
    print(f"\nTotal training process completed in {end_time - start_time:.2f} seconds.")

if __name__ == "__main__":
    main()