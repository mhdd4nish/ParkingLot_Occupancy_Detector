# Smart Parking Lot Occupancy Detector

A computer vision and machine learning pipeline that detects parking space occupancy using a custom-trained SVM classifier and classic CV heuristics.

## Features
* **Custom Feature Extraction:** Combines HOG, LBP, and HSV Color Histograms for robust detection.
* **Perspective Transformation:** Flattens angled parking spaces into standard top-down ROIs.
* **Confidence & Veto System:** Uses Otsu thresholding and contour math to double-check SVM predictions and eliminate false positives (like pedestrians or shadows).
* **Centering Assessment:** Calculates how well a vehicle is parked within the lines.


## Installation
1. Clone this repository:
2. Install dependencies:
   \`pip install -r requirements.txt\`

## Usage
*Note: The raw PKLot dataset and pre-trained models are not included due to size limits.*

**1. Define Parking Spaces:**
Run `python run_space_definer.py` to manually click the 4 corners of each parking spot on a reference image. This saves the coordinates to `parking_spaces.json`.

**2. Train the Model:**
Download the PKLot dataset, and run:
`python run_training.py`
This will extract HOG/LBP/Color features and train the SVM, saving it to `models/svm_model.pkl`.

**3. Run the Live Detector:**
`python src/main.py`
This runs the slideshow mode, processing images and displaying the bounding boxes, occupancy status, and parking fit assessments.