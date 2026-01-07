import cv2
import numpy as np
from skimage.feature import local_binary_pattern, hog
from skimage import exposure

class FeatureExtractor:
    """
    Extract features from parking space ROIs for classification
    Implements LBP (Local Binary Patterns) and HOG (Histogram of Oriented Gradients)
    """
    
    def __init__(self, config):
        self.config = config
    
    def extract_lbp_features(self, image: np.ndarray) -> np.ndarray:

        # Convert to grayscale if needed
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image
        
        # Resize to standard size
        gray = cv2.resize(gray, (128, 128))
        
        # Compute LBP
        lbp = local_binary_pattern(
            gray,
            P=self.config.LBP_POINTS,
            R=self.config.LBP_RADIUS,
            method='uniform'
        )
        
        # Calculate histogram
        n_bins = self.config.LBP_POINTS + 2
        hist, _ = np.histogram(
            lbp.ravel(),
            bins=n_bins,
            range=(0, n_bins),
            density=True
        )
        
        return hist
    
    def extract_hog_features(self, image: np.ndarray) -> np.ndarray:

        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image
        
        # Resize to standard size
        gray = cv2.resize(gray, (128, 128))
        
        # Compute HOG
        features, hog_image = hog(
            gray,
            orientations=self.config.HOG_ORIENTATIONS,
            pixels_per_cell=self.config.HOG_PIXELS_PER_CELL,
            cells_per_block=self.config.HOG_CELLS_PER_BLOCK,
            visualize=True,
            feature_vector=True
        )
        
        return features
    
    def extract_color_histogram(self, image: np.ndarray) -> np.ndarray:

        # Resize to standard size
        image = cv2.resize(image, (128, 128))
        
        # Convert to HSV for better color representation
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        
        # Compute histograms for each channel
        hist_h = cv2.calcHist([hsv], [0], None, [32], [0, 180])
        hist_s = cv2.calcHist([hsv], [1], None, [32], [0, 256])
        hist_v = cv2.calcHist([hsv], [2], None, [32], [0, 256])
        
        # Normalize
        hist_h = cv2.normalize(hist_h, hist_h).flatten()
        hist_s = cv2.normalize(hist_s, hist_s).flatten()
        hist_v = cv2.normalize(hist_v, hist_v).flatten()
        
        # Concatenate
        hist = np.concatenate([hist_h, hist_s, hist_v])
        
        return hist
    
    def extract_combined_features(self, image: np.ndarray) -> np.ndarray:
 
        lbp_feat = self.extract_lbp_features(image)
        hog_feat = self.extract_hog_features(image)
        color_feat = self.extract_color_histogram(image)
        
        # Combine all features
        combined = np.concatenate([lbp_feat, hog_feat, color_feat])
        
        return combined