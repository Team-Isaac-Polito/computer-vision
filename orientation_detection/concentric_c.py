import cv2
import math
import numpy as np
from itertools import combinations

class OrientationDetection:
    def __init__(self):
        # Tunable parameters for preprocessing
        self.hsv_threshold = 90
        self.mask_morph_kernel_size = (3, 3)
        self.mask_morph_iterations = 1
        self.blur_kernel_size = (3, 3)
        self.block_size = 13
        self.C = 3
        
        # Tunable parameters for contour filtering
        self.min_contour_area = 5
        self.aspect_ratio_min = 0.6
        self.aspect_ratio_max = 1.4
        self.max_solidity = 0.7
        self.scale_factor = 0.9
        self.max_gap_ratio = 0.3

        # Tunable parameters for concentric C detection
        self.min_distance = 10
        
        # Debug option to display intermediate images
        self.debug = False
        self.text_font = cv2.FONT_HERSHEY_SIMPLEX
        self.text_scale = 0.8
        self.text_thickness = 1

    def process_image(self, frame):
        # Convert input frame to HSV color space for color filtering
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        
        # Define the range for black color in HSV
        lower_black = np.array([0, 0, 0])
        upper_black = np.array([180, 255, self.hsv_threshold])
        
        # Create a mask for black color
        mask = cv2.inRange(hsv, lower_black, upper_black)

        # Dilate the mask to broaden the black regions
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, self.mask_morph_kernel_size)
        morph_mask = cv2.dilate(mask, kernel, iterations=self.mask_morph_iterations)

        # Apply the mask to isolate black regions
        filtered = frame.copy()
        filtered[morph_mask == 0] = [255, 255, 255]  # Set non-black regions to white
        
        # Convert the highlighted image to grayscale
        image = cv2.cvtColor(filtered, cv2.COLOR_BGR2GRAY)
        
        # Apply Gaussian blur to reduce noise
        image = cv2.GaussianBlur(image, self.blur_kernel_size, 0)
        
        # Apply adaptive thresholding to isolate the object (inversion applied)
        binary = cv2.adaptiveThreshold(
            image, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
            cv2.THRESH_BINARY_INV, self.block_size, self.C
        )
        
        # Find contours on the thresholded result
        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # Debug: show intermediate images
        if self.debug:
            debug_images = {
                "Original": frame,
                "HSV": hsv,
                "Mask": mask,
                "Morph Mask": morph_mask,
                "Binary": binary,
                "Contours": cv2.drawContours(cv2.cvtColor(binary, cv2.COLOR_GRAY2BGR), contours, -1, (0, 255, 0), 2)
            }

            for name, img in debug_images.items():
                height, width = img.shape[:2]
                cv2.namedWindow(name, cv2.WINDOW_NORMAL)
                cv2.resizeWindow(name, width, height)
                cv2.imshow(name, img)
        
        # Prepare an output image for visualization
        output_image = frame.copy()
        detected_angles = []       # List to store calculated orientation angles
        orientation_labels = []    # List to store associated textual orientation labels
        detected_areas = []      # List to store areas of detected contours
        centroids = []          # List to store centroids of detected contours
        labels = [
            "(R) RIGHT", "(TR) TOP RIGHT", "(T) TOP", "(TL) TOP LEFT",
            "(L) LEFT", "(BL) BOTTOM LEFT", "(B) BOTTOM", "(BR) BOTTOM RIGHT"
        ]
        annotations = []  # List to store text annotation parameters for later drawing

        # Process each contour found in the image
        for contour in contours:
            if len(contour) < 5:
                continue  # Skip contours with too few points

            # Filter contours by area
            area = cv2.contourArea(contour)
            if area < self.min_contour_area:
                continue
            detected_areas.append(area)

            # Filter by aspect ratio using the bounding rectangle
            x, y, w, h = cv2.boundingRect(contour)
            aspect_ratio = float(w) / h
            if aspect_ratio < self.aspect_ratio_min or aspect_ratio > self.aspect_ratio_max:
                continue

            # Ensure the contour has a significant hollow region
            hull = cv2.convexHull(contour)
            hull_area = cv2.contourArea(hull)
            solidity = float(area) / hull_area if hull_area > 0 else 0
            if solidity > self.max_solidity:  # Skip contours that are too solid (not hollow enough)
                continue

            # Fit an ellipse around the contour to obtain center, axes, and angle
            ellipse = cv2.fitEllipse(contour)
            (center, axes, ellipse_angle) = ellipse
            (xc, yc) = center

            # Reduce the size of the ellipse by scaling down its axes
            axes = (axes[0] * self.scale_factor, axes[1] * self.scale_factor)
            ellipse = (center, axes, ellipse_angle)

            # Create a mask for the contour
            contour_mask = np.zeros_like(image, dtype=np.uint8)
            cv2.fillPoly(contour_mask, [contour], 255)

            # Create a mask for the ellipse boundary
            ellipse_boundary = np.zeros_like(image, dtype=np.uint8)
            cv2.ellipse(ellipse_boundary, ellipse, 255, 1)

            # Subtract the contour mask from the ellipse boundary to find the gap
            gap_mask = cv2.subtract(ellipse_boundary, contour_mask)

            # Skip the contour if the length of gap is very long relative to ellipse boundary
            if np.sum(gap_mask) >= self.max_gap_ratio * np.sum(ellipse_boundary):
                continue

            # Locate the gap region in the ellipse boundary
            contours_gap, _ = cv2.findContours(gap_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            if contours_gap:
                # Find the largest gap contour
                gap_contour = max(contours_gap, key=cv2.contourArea)

                # Compute the center of the gap
                gap_points = gap_contour[:, 0, :]
                gap_center = tuple(np.mean(gap_points, axis=0).astype(int))

                # Debug: Visualize the gap center
                if self.debug:
                    gap_radius = max(1, int(math.sqrt(area) * 0.1))  # Adjust radius based on area
                    cv2.circle(output_image, gap_center, gap_radius, (0, 0, 255), -1)

                # Compute the orientation angle
                final_angle = math.degrees(-math.atan2(gap_center[1] - yc, gap_center[0] - xc))
                final_angle = final_angle % 360  # Normalize angle to [0, 360]

                # Debug: Draw a line from ellipse center to gap center
                if self.debug:
                    line_thickness = max(1, int(math.sqrt(area) * 0.01))  # Adjust thickness based on area
                    cv2.line(output_image, (int(xc), int(yc)), gap_center, (255, 0, 0), line_thickness)

                # Save the computed angle
                detected_angles.append(final_angle)

                # Map the angle to an orientation label
                label = labels[int((final_angle + 22.5) % 360 // 45)]
                orientation_labels.append(label)

                # Compute the center of the region surrounded by the contour for clustering
                moments = cv2.moments(contour)
                if moments["m00"] != 0: # M(0, 0) - Area of the object
                    cx = int(moments["m10"] / moments["m00"])  # Center x-coordinate
                    cy = int(moments["m01"] / moments["m00"])  # Center y-coordinate
                    centroid = (cx, cy)
                else:
                    centroid = (0, 0)  # Default to (0, 0) if contour area is zero
                centroids.append(centroid)

                # Draw the fitted ellipse and filled contour
                ellipse_thickness = max(1, int(math.sqrt(area) * 0.05))  # Adjust thickness based on area
                cv2.ellipse(output_image, ellipse, (0, 255, 0), ellipse_thickness)
                if self.debug:
                    cv2.drawContours(output_image, [contour], -1, (255, 0, 0), -1)

                # Annotate the output image with the orientation label
                text_scale = self.text_scale * max(0.4, min(1.5, math.sqrt(area) * 0.01))  # Adjust text scale based on area
                text_thickness = self.text_thickness * max(1, int(math.sqrt(area) * 0.05))  # Adjust text thickness based on area
                text_size = cv2.getTextSize(label, self.text_font, text_scale, text_thickness)[0]
                text_x = int(x + (w - text_size[0]) / 2)
                text_y = max(20, int(y - 10))

                # Store the annotation details
                annotations.append({
                    "label": label,
                    "position": (text_x, text_y),
                    "font": self.text_font,
                    "scale": text_scale,
                    "color": (146, 44, 202),
                    "thickness": text_thickness
                })
            
        # Draw every stored text annotation on top of all shapes to avoid overlaying
        for ann in annotations:
            cv2.putText(output_image, ann["label"], ann["position"], ann["font"],
                        ann["scale"], ann["color"], ann["thickness"])

        # Handle concentric Cs
        concentric_c = zip(detected_areas, orientation_labels, centroids)
        concentric_c = sorted(concentric_c, key=lambda x: x[0])  # Sort by area (ascending)
        concentric_c_labels = ['Inner C', 'Middle C', 'Outer C']

        # Find the closest 3 centroids based on their positions
        concentric_c_dict = {}
        if len(centroids) >= 3:
            min_distance = self.min_distance
            closest_cluster = None

            # Iterate through all combinations of 3 centroids
            for cluster in combinations(centroids, 3):
                # Calculate the sum of pairwise distances in the cluster
                distance = (
                    math.sqrt((cluster[0][0] - cluster[1][0]) ** 2 + (cluster[0][1] - cluster[1][1]) ** 2) +
                    math.sqrt((cluster[1][0] - cluster[2][0]) ** 2 + (cluster[1][1] - cluster[2][1]) ** 2) +
                    math.sqrt((cluster[2][0] - cluster[0][0]) ** 2 + (cluster[2][1] - cluster[0][1]) ** 2)
                )
                # Update the closest cluster if a smaller distance is found
                if distance < min_distance:
                    min_distance = distance
                    closest_cluster = cluster

            # Assign labels to the closest cluster
            if closest_cluster:
                for idx, centroid in enumerate(closest_cluster):
                    for area, label, c in concentric_c:
                        if c == centroid:
                            concentric_c_dict[concentric_c_labels[idx]] = (label, centroid, area)
                            break

            # Debug print out of results
            for key, value in concentric_c_dict.items():
                print(f"{key}: {value[0]} centered at {value[1]} with area {value[2]}")
            if min_distance < self.min_distance:
                print("Closest cluster found with minimum distance: ", min_distance)
        
        return output_image, concentric_c_dict, detected_angles

if __name__ == "__main__":
    od = OrientationDetection()
    cap = cv2.VideoCapture(0)  # Capture video from webcam

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        processed_frame, concentric_c, angles = od.process_image(frame)
        cv2.imshow("Orientation Detection", processed_frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()