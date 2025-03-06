import cv2

class MotionDetection:
    def __init__(self):
        self.bg_subtractor = cv2.createBackgroundSubtractorMOG2()
        self.first_image_received = False
        self.erosion_iterations = 1
        self.dilation_iterations = 1
        self.close_iterations = 1
        self.moving_average_weight = 0.5
        self.activation_threshold = 127
        self.max_area = 5000
        self.min_area = 500
        self.detection_limit = 5
        self.debug_contours = False

    def process_image(self, frame):
        fgimg = self.bg_subtractor.apply(frame)
        
        fgimg = cv2.morphologyEx(fgimg, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)), iterations=self.close_iterations)
        fgimg = cv2.erode(fgimg, None, iterations=self.erosion_iterations)
        fgimg = cv2.dilate(fgimg, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)), iterations=self.dilation_iterations)

        if not self.first_image_received:
            self.accumulated_image = fgimg.copy()
            self.first_image_received = True
        else:
            cv2.addWeighted(self.accumulated_image, (1 - self.moving_average_weight), fgimg, self.moving_average_weight, 0.0, self.accumulated_image)

        _, thresholded = cv2.threshold(self.accumulated_image, self.activation_threshold, 255, cv2.THRESH_BINARY)

        contours, _ = cv2.findContours(thresholded, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        if self.debug_contours:
            cv2.drawContours(frame, contours, -1, (0, 0, 255), 2)

        largest_contours = sorted(contours, key=cv2.contourArea, reverse=True)[:self.detection_limit]
        for contour in largest_contours:
            area = cv2.contourArea(contour)
            if self.min_area <= area <= self.max_area:
                bounding_rect = cv2.boundingRect(contour)
                cv2.rectangle(frame, bounding_rect, (0, 0, 255), 2)

        return frame

if __name__ == "__main__":
    md = MotionDetection()
    cap = cv2.VideoCapture(0)  # Capture video from webcam

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        processed_frame = md.process_image(frame)

        cv2.imshow("Motion Detection", processed_frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
