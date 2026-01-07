# src/run_feature_comparison.py

import cv2
import sys
import numpy as np
from pathlib import Path
from tqdm import tqdm
import time
import random
import matplotlib.pyplot as plt

# --- Import project modules ---
from config import Config
from feature_extractor import FeatureExtractor
from occupancy_classifier import OccupancyClassifier

# --- Import scikit-learn metrics ---
from sklearn.metrics import roc_curve, auc
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

# --- Configuration ---
SEGMENTED_DATA_DIR = "C:/Users/mhdda/parking_system/data/PKLot_segmented/UFPR05"
MAX_SAMPLES_PER_CLASS = 12000  # Use the same 10k sample limit
RANDOM_STATE = 42
# ---------------------

def load_separate_datasets(base_path: Path, extractor: FeatureExtractor):
    """
    Loads all datasets, but keeps features separate.
    Returns: A dictionary of feature_sets and a single 'y' label array
    """
    
    features_lbp = []
    features_hog = []
    features_color = []
    features_combined = []
    labels_list = []
    
    # --- 1. Load file paths (shuffled) ---
    empty_paths = list(base_path.glob("**/Empty/*.jpg"))
    occupied_paths = list(base_path.glob("**/Occupied/*.jpg"))
    random.shuffle(empty_paths)
    random.shuffle(occupied_paths)
    
    if MAX_SAMPLES_PER_CLASS:
        empty_paths = empty_paths[:MAX_SAMPLES_PER_CLASS]
        occupied_paths = occupied_paths[:MAX_SAMPLES_PER_CLASS]
        
    all_paths = [(p, 0) for p in empty_paths] + [(p, 1) for p in occupied_paths]
    random.shuffle(all_paths) # Shuffle again to mix empty/occupied for processing
    
    print(f"Loading and extracting features for {len(all_paths)} total images...")
    
    # --- 2. Extract all features in one loop ---
    for img_path, label in tqdm(all_paths, desc="Extracting all features"):
        img = cv2.imread(str(img_path))
        if img is None:
            continue
            
        # Extract features one by one
        lbp = extractor.extract_lbp_features(img)
        hog = extractor.extract_hog_features(img)
        color = extractor.extract_color_histogram(img)
        
        # Append to their respective lists
        features_lbp.append(lbp)
        features_hog.append(hog)
        features_color.append(color)
        features_combined.append(np.concatenate([lbp, hog, color]))
        labels_list.append(label)
            
    # --- 3. Convert to numpy arrays ---
    datasets = {
        "LBP": np.array(features_lbp),
        "HOG": np.array(features_hog),
        "Color": np.array(features_color),
        "Combined": np.array(features_combined)
    }
    
    y = np.array(labels_list)
    
    return datasets, y

def train_and_get_roc_data(config, feature_name, X, y):
    """
    Helper function to train one SVM and return its ROC data.
    """
    print(f"\n--- Processing Model: {feature_name} ---")
    
    # 1. Split and Scale
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y)
    
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # 2. Train
    print(f"Training SVM on '{feature_name}' features...")
    classifier = OccupancyClassifier(config, classifier_type='svm')
    # We don't need the verbose output here
    classifier.classifier.verbose = False 
    classifier.classifier.fit(X_train_scaled, y_train)
    
    # 3. Get Probabilities
    print("Getting probabilities...")
    y_probs = classifier.classifier.predict_proba(X_test_scaled)[:, 1]
    
    # 4. Get ROC data
    fpr, tpr, _ = roc_curve(y_test, y_probs)
    roc_auc = auc(fpr, tpr)
    
    print(f"'{feature_name}' Model AUC: {roc_auc:.4f}")
    
    return fpr, tpr, roc_auc

def main():
    print("--- Feature Performance Comparison (ROC Analysis) ---")
    print("WARNING: This will take a very long time (training 4 SVMs).")
    start_time = time.time()
    
    config = Config()
    extractor = FeatureExtractor(config)
    
    # 1. Load all datasets
    datasets, y = load_separate_datasets(Path(SEGMENTED_DATA_DIR), extractor)
    
    roc_results = {}
    
    # 2. Train and evaluate each model one by one
    for feature_name, X in datasets.items():
        fpr, tpr, roc_auc = train_and_get_roc_data(config, feature_name, X, y)
        roc_results[feature_name] = (fpr, tpr, roc_auc)
        
    print("\n--- All models trained. Generating plot... ---")

    # 3. Plot all ROC curves on one graph
    plt.figure(figsize=(10, 8))
    
    # Plot LBP
    fpr, tpr, roc_auc = roc_results["LBP"]
    plt.plot(fpr, tpr, lw=2, label=f'LBP (AUC = {roc_auc:.4f})')
    
    # Plot HOG
    fpr, tpr, roc_auc = roc_results["HOG"]
    plt.plot(fpr, tpr, lw=2, label=f'HOG (AUC = {roc_auc:.4f})')
    
    # Plot Color
    fpr, tpr, roc_auc = roc_results["Color"]
    plt.plot(fpr, tpr, lw=2, label=f'Color (AUC = {roc_auc:.4f})')
    
    # Plot Combined (make it stand out)
    fpr, tpr, roc_auc = roc_results["Combined"]
    plt.plot(fpr, tpr, color='black', lw=3, linestyle='--',
             label=f'Combined (AUC = {roc_auc:.4f})')

    # Plot the 50/50 "random guess" line
    plt.plot([0, 1], [0, 1], color='gray', lw=2, linestyle='--')
    
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve Comparison of Individual Features')
    plt.legend(loc="lower right")
    
    # 4. Save the plot
    output_path = config.RESULTS_DIR / "feature_comparison_roc_curve.png"
    plt.savefig(output_path)
    
    print(f"\nComparison plot saved to: {output_path}")
    
    end_time = time.time()
    print(f"Total analysis completed in {end_time - start_time:.2f} seconds.")

if __name__ == "__main__":
    main()