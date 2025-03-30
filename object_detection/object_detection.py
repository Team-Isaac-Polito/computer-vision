# https://docs.ultralytics.com/tasks/detect/#val

from ultralytics import YOLO
import cv2
from pyzbar.pyzbar import decode
from apriltag import apriltag

# Load YOLO model
model = YOLO("best.pt")

# Initialize webcam
cap = cv2.VideoCapture(0)

# Initialize AprilTag detector
detector = apriltag("tagStandard41h12")

while True:
    ret, frame = cap.read()
    if not ret:
        break
    
    # Convert frame to tensor
    img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = model(img)
    
    # Process results
    for result in results:
        bbox = result.boxes.xyxy
        labels = [result.names[cls.item()] for cls in result.boxes.cls.int()]
        confs = result.boxes.conf
        for i, conf in enumerate(confs):
            if conf > 0.5:  # Confidence threshold
                x1, y1, x2, y2 = bbox[i]
                label = labels[i]
                cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (255, 0, 0), 2)
                cv2.putText(frame, f'{label}: {conf:.2f}', (int(x1), int(y1) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 0, 0), 2)
    
    # Process QR codes
    qr_codes = decode(frame)
    for qr in qr_codes:
        # Get bounding box and data
        x, y, w, h = qr.rect
        qr_data = qr.data.decode('utf-8')
        # Draw bounding box and display data
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
        cv2.putText(frame, f'QR: {qr_data}', (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
        print(f"QR Code Data: {qr_data}")

    # Process AprilTags
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)  # Convert frame to grayscale
    apriltag_detections = detector.detect(gray)

    for detection in apriltag_detections:
        # Print the whole detection dictionary to inspect the structure
        print(detection)

        # Access tag corners and tag ID
        corners = detection.get('lb-rb-rt-lt', None)  # Corners are stored under 'lb-rb-rt-lt'
        tag_id = detection.get('id', 'Unknown')

        if corners is not None:
            # Unpack the corners, which are stored as a 4x2 numpy array
            ptA, ptB, ptC, ptD = corners

            # Convert corner points to integers for drawing
            ptA = tuple(map(int, ptA))
            ptB = tuple(map(int, ptB))
            ptC = tuple(map(int, ptC))
            ptD = tuple(map(int, ptD))

            # Draw bounding box for the AprilTag
            cv2.line(frame, ptA, ptB, (0, 0, 255), 2)
            cv2.line(frame, ptB, ptC, (0, 0, 255), 2)
            cv2.line(frame, ptC, ptD, (0, 0, 255), 2)
            cv2.line(frame, ptD, ptA, (0, 0, 255), 2)

            # Display the tag ID on the frame
            cv2.putText(frame, f'AT_ID: {tag_id}', (ptA[0], ptA[1] - 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
            print(f"AprilTag Detected: ID {tag_id}")

            # Display additional tag information (optional)
            center_text = f'Center: ({int(detection["center"][0])}, {int(detection["center"][1])})'
            margin_text = f'Margin: {detection["margin"]:.2f}'

            # Display the center and margin info below the AT_ID
            cv2.putText(frame, center_text, (int(detection["center"][0]) + 10, int(detection["center"][1]) + 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 2)
            cv2.putText(frame, margin_text, (int(detection["center"][0]) + 10, int(detection["center"][1]) + 60),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 2)

    # Display the frame
    cv2.imshow('YOLO Detection', frame)
    
    # Break the loop on 'q' key press
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release resources
cap.release()
cv2.destroyAllWindows()