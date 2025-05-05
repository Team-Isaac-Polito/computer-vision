import cv2
from qr_apriltag_detection.qr_reader import process_qr_codes
from qr_apriltag_detection.apriltag_reader import process_apriltags

# Initialize webcam
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break
    
    # Detect and display QR codes
    frame = process_qr_codes(frame)

    # Detect and display AprilTags
    frame = process_apriltags(frame)

    # Display the frame
    cv2.imshow('QR-AprilTag Detection', frame)
    
    # Break the loop on 'q' key press
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release resources
cap.release()
cv2.destroyAllWindows()