import cv2
import numpy as np

# Tải mô hình ONNX
def load_model(model_path):
    net = cv2.dnn.readNetFromONNX(model_path)
    return net

# Hàm dự đoán trạng thái mắt
def predict_eye_state(eye_image, net):
    blob = cv2.dnn.blobFromImage(eye_image, scalefactor=1/255.0, size=(24, 24), mean=0, swapRB=True, crop=False)
    net.setInput(blob)
    output = net.forward()
    return np.argmax(output)  # 0: Nhắm, 1: Mở
