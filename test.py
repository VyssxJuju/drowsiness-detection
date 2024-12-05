import onnx

# Đọc mô hình
model_path = "eye_state_model.onnx"
model = onnx.load(model_path)
onnx.checker.check_model(model)

print("Mô hình hợp lệ và sẵn sàng sử dụng!")
