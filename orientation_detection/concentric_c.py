import cv2
import math
import numpy as np
from itertools import combinations

class OrientationDetection:
    def __init__(self):
        # Tunable parameters for preprocessing
        self.hsv_threshold = 50
        self.blur_kernel_size = (5, 5)
        self.mask_morph_kernel_size = (5, 5)
        self.mask_morph_iterations = 1
        self.morph_kernel_size = (5, 5)
        self.morph_iterations = 1
        self.threshold = 0
        self.canny_threshold1 = 100
        self.canny_threshold2 = 200
        self.skip_edge_detection = True 
        
        # Tunable parameters for contour filtering
        self.min_contour_area = 50
        self.aspect_ratio_min = 0.7
        self.aspect_ratio_max = 1.3
        self.max_solidity = 0.7
        self.scale_factor = 0.9
        
        # Debug option to display intermediate images
        self.debug = True
        self.text_font = cv2.FONT_HERSHEY_SIMPLEX
        self.text_scale = 0.8
        self.text_thickness = 2
        self.min_distance = 200

    def process_image(self, frame):
        # Convert input frame to HSV color space for color filtering
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        
        # Define the range for black color in HSV
        lower_black = np.array([0, 0, 0])
        upper_black = np.array([180, 255, self.hsv_threshold])
        
        # Create a mask for black color
        mask = cv2.inRange(hsv, lower_black, upper_black)

        # Apply morphological operations to clean up the mask
        morph_mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, self.mask_morph_kernel_size, iterations=self.mask_morph_iterations)
        
        # Apply the mask to isolate black regions
        filtered = frame.copy()
        filtered[morph_mask == 0] = [255, 255, 255]  # Set non-black regions to white
        
        # Convert the highlighted image to grayscale
        image = cv2.cvtColor(filtered, cv2.COLOR_BGR2GRAY)
        
        # Apply Gaussian blur to reduce noise
        image = cv2.GaussianBlur(image, self.blur_kernel_size, 0)
        
        # Apply thresholding with OTSU to isolate the object (inversion applied)
        _, binary = cv2.threshold(image, self.threshold, 255,
                      cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        
        # Apply morphological closing to fill gaps in the object boundary
        morph_kernel = np.ones(self.morph_kernel_size, np.uint8)
        morph = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, morph_kernel, iterations=self.morph_iterations)
        
        # skip edge detection if specified (for simple cases)
        if self.skip_edge_detection:
            # Skip edge detection and directly find contours on the morph result
            contours, _ = cv2.findContours(morph, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            edges = morph  # Use morph as the edge map for visualization
        else:
            # Perform Canny edge detection
            edges = cv2.Canny(morph, self.canny_threshold1, self.canny_threshold2)
            # Find contours on the edge map
            contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # Debug: show intermediate images if desired
        if self.debug:
            debug_images = {                
                "Original": frame,
                "Mask": mask,
                "Morph Mask": morph_mask,
                "Binary": binary,
                "Morph": morph,
                "Edges": edges,
                "Contours": cv2.drawContours(cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR), contours, -1, (0, 255, 0), 2)
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
                    cv2.circle(output_image, gap_center, 5, (0, 0, 255), -1)

                # Compute the orientation angle
                final_angle = math.degrees(-math.atan2(gap_center[1] - yc, gap_center[0] - xc))
                final_angle = final_angle % 360  # Normalize angle to [0, 360]

                # Debug: Draw a line from ellipse center to gap center
                if self.debug:
                    cv2.line(output_image, (int(xc), int(yc)), gap_center, (255, 0, 0), 2)

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
                cv2.ellipse(output_image, ellipse, (0, 255, 0), 2)
                if self.debug:
                    cv2.drawContours(output_image, [contour], -1, (255, 0, 0), -1)

                # Annotate the output image with the orientation label
                text_size = cv2.getTextSize(label, self.text_font, self.text_scale, self.text_thickness)[0]
                text_x = int(x + (w - text_size[0]) / 2)
                text_y = max(20, int(y - 10))
                cv2.putText(output_image, label, (text_x, text_y),
                            self.text_font, self.text_scale, (146, 44, 202), self.text_thickness)

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