import cv2
import winsound
import numpy as np

# Tải mô hình đã huấn luyện sẵn
MODEL_PATH = "eye_state_model.onnx"  # Đường dẫn tới mô hình
net = cv2.dnn.readNetFromONNX(MODEL_PATH)

# Hàm dự đoán trạng thái mắt
def predict_eye_state(eye_image, net):
    blob = cv2.dnn.blobFromImage(eye_image, scalefactor=1/255.0, size=(24, 24), mean=0, swapRB=True, crop=False)
    net.setInput(blob)
    output = net.forward()
    return np.argmax(output)  # 0: Nhắm, 1: Mở

# Phát hiện khuôn mặt và mắt
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_eye.xml")

# Khởi động camera
cap = cv2.VideoCapture(0)

# Biến đếm số khung hình mắt nhắm
closed_frames = 0

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5)

    for (x, y, w, h) in faces:
        roi_gray = gray[y:y+h, x:x+w]
        eyes = eye_cascade.detectMultiScale(roi_gray, scaleFactor=1.1, minNeighbors=5)

        for (ex, ey, ew, eh) in eyes:
            eye = roi_gray[ey:ey+eh, ex:ex+ew]
            eye_resized = cv2.resize(eye, (24, 24))  # Kích thước phù hợp với mô hình
            state = predict_eye_state(eye_resized, net)

            label = "Open" if state == 1 else "Closed"
            color = (0, 255, 0) if state == 1 else (0, 0, 255)
            cv2.rectangle(frame, (x+ex, y+ey), (x+ex+ew, y+ey+eh), color, 2)
            cv2.putText(frame, label, (x+ex, y+ey-10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

            # Cảnh báo nếu mắt nhắm nhiều khung hình
            if state == 0:  # Mắt nhắm
                closed_frames += 1
                if closed_frames >= 10:  # Nhắm liên tục 10 khung hình
                    winsound.Beep(2500, 1000)
            else:
                closed_frames = 0

    cv2.imshow("Drowsiness Detection", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
