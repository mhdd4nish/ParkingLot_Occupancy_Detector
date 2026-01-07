import cv2
import numpy as np
import random

class WidthEstimator:
    """
    Measures vehicle widths and assesses fit based on
    standardized, top-down ROIs.
    """

    def __init__(self, config):
        self.config = config

    def measure_parking_space_width(self, roi: np.ndarray) -> float:

        return self.config.STANDARD_PARKING_WIDTH

    def detect_vehicle_in_space(self, roi: np.ndarray) -> (bool, np.ndarray, float):

        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)

        otsu_thresh_val, car_shadow_mask = cv2.threshold(gray, 0, 255, 
                                                    cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)

        # 2. Create a mask of just the original image (ignoring black padding)
        padding_mask = cv2.threshold(gray, 1, 255, cv2.THRESH_BINARY)[1]


        final_car_mask = cv2.bitwise_and(car_shadow_mask, padding_mask)

        kernel = np.ones((3, 3), np.uint8)
        clean_mask = cv2.morphologyEx(final_car_mask, cv2.MORPH_CLOSE, kernel, iterations=3)

        contours, _ = cv2.findContours(clean_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            return False, None, 0.0

        largest_contour = max(contours, key=cv2.contourArea)
        contour_area = cv2.contourArea(largest_contour)

        
        if contour_area < 1000: 
            return False, None, 0.0

        vehicle_mask = np.zeros_like(clean_mask)
        cv2.drawContours(vehicle_mask, [largest_contour], -1, (255), -1)

        return True, vehicle_mask, contour_area

    def assess_centering(self, roi: np.ndarray, vehicle_mask: np.ndarray) -> (dict, float):

        assessment = {
            'status': 'N/A',
            'offset_m': 0.0
        }

        # 1. Find the largest contour from the mask
        contours, _ = cv2.findContours(vehicle_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            assessment['status'] = "Centering Error"
            return assessment, 0.0

        largest_contour = max(contours, key=cv2.contourArea)

        # 2. Find the center of the car 
        M = cv2.moments(largest_contour)
        if M["m00"] == 0:
            assessment['status'] = "Centering Error"
            return assessment, 0.0

        car_center_x_px = int(M["m10"] / M["m00"])

        # 3. Find the center of the parking space (the ROI image)
        space_center_x_px = roi.shape[1] / 2

        # 4. Calculate the offset in pixels
        pixel_offset = abs(car_center_x_px - space_center_x_px)

        # 5. Convert offset to meters
        offset_m = pixel_offset / self.config.ROI_PIXELS_PER_METER
        assessment['offset_m'] = round(offset_m, 2)

        if offset_m < 0.15:  
            assessment['status'] = "Well Centered"
        elif offset_m < 0.3: 
            assessment['status'] = "Slightly Off-Center"
        else:
            assessment['status'] = "Poorly Centered"

        return assessment, offset_m