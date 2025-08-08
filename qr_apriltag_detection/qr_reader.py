import cv2

# pip install opencv-contrib-python

# Paths of the model files
# from https://github.com/WeChatCV/opencv_3rdparty
model_proto = "qr_apriltag_detection/models/detect.prototxt"
model_weights = "qr_apriltag_detection/models/detect.caffemodel"
sr_proto = "qr_apriltag_detection/models/sr.prototxt"
sr_weights = "qr_apriltag_detection/models/sr.caffemodel"

# Start WeChat QRCode detector
detector = cv2.wechat_qrcode_WeChatQRCode(
    model_proto, model_weights, sr_proto, sr_weights
)


def process_qr_codes(frame):
    # Detect and decode QR codes in the frame
    qr_texts, points = detector.detectAndDecode(frame)

    for text, pts in zip(qr_texts, points):
        pts = pts.astype(int)  # corners (4x2)

        # Draw a quadrilateral
        cv2.polylines(frame, [pts], isClosed=True, color=(0, 255, 0), thickness=2)

        # Calculate and mark the center
        cx = int(pts[:, 0].mean())
        cy = int(pts[:, 1].mean())
        cv2.circle(frame, (cx, cy), 6, (0, 255, 255), -1)

        # Print QR content to screen
        x0, y0 = pts[0]
        cv2.putText(
            frame, text, (x0, y0 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2
        )

        # Print to console
        print("QR Code Data:", text)
        print("Corners:")
        for i, (x, y) in enumerate(pts):
            print(f"  P{i}: x={x}, y={y}")
        print(f"Center: x={cx}, y={cy}")
        print("-" * 40)

    return frame
