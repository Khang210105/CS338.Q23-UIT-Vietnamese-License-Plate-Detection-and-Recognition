# Vietnamese License Plate Detection And Recognition (CS338.Q23 - UIT)

Hệ thống phát hiện và nhận dạng biển số xe máy Việt Nam sử dụng các mô hình học sâu hiện đại: **YOLOv11** cho bài toán phát hiện vị trí (Detection) kết hợp cùng **CRNN** và **PARSeq** cho bài toán nhận dạng ký tự (Text Recognition). Hệ thống tích hợp thêm bộ lọc hậu xử lý dựa trên quy định biển số xe Việt Nam nhằm tối ưu hóa độ chính xác và một giao diện Web trực quan.

---

## 📌 Thành viên thực hiện
* **Sinh viên:** Phạm Minh Bảo Khang – 23520705
* **Giảng viên hướng dẫn:** ThS. Đỗ Văn Tiến
* **Môn học:** CS338.Q23 – Nhận dạng (Khoa Khoa học Máy tính - UIT)

---

## 🛠 Cấu trúc thư mục mã nguồn

```text
├── templates/
│   └── index.html          # Giao diện Web HTML cho ứng dụng
├── model/                  # (Thư mục cục bộ) Chứa file trọng số mô hình (.pt, .pth, .ckpt)
├── app.py                  # File chạy ứng dụng chính (Flask Web App)
├── train_yolo.py           # Huấn luyện mô hình YOLOv11 Detection
├── train_crnn.py           # Huấn luyện mô hình CRNN Text Recognition
├── train_parseq.py         # Huấn luyện mô hình PARSeq Text Recognition
├── eval_yolo.py            # Đánh giá mô hình YOLOv11 trên tập kiểm thử (Test set)
├── eval_crnn.py            # Đánh giá mô hình CRNN trên tập kiểm thử
├── eval_parseq.py          # Đánh giá mô hình PARSeq trên tập kiểm thử
├── preprocess.py           # Tiền xử lý, cắt vùng biển số để làm đầu vào cho bài toán nhận dạng
├── vn_plate.py             # Bộ quy tắc hậu xử lý sửa lỗi ký tự theo luật biển số xe Việt Nam
├── requirements.txt        # Danh sách các thư viện cần cài đặt
└── README.md               # Hướng dẫn sử dụng dự án


🚀 Hướng dẫn cài đặt và sử dụng
1. Chuẩn bị môi trường
Khuyến khích khởi tạo một môi trường ảo (venv hoặc conda) sử dụng Python từ 3.9 đến 3.11 trước khi cài đặt:

Bash
# Khởi tạo môi trường ảo (ví dụ với venv)
python -m venv venv

# Kích hoạt môi trường ảo
# Trên Windows:
.\venv\Scripts\activate
# Trên Linux/Mac:
source venv/bin/activate

# Cài đặt tất cả các thư viện phụ thuộc
pip install -r requirements.txt
2. Tiền xử lý dữ liệu (Preprocessing)
Trước khi đưa vùng ảnh ký tự biển số vào huấn luyện mô hình nhận dạng (CRNN / PARSeq), chạy file tiền xử lý để thực hiện chuẩn hóa và cắt tách phân vùng:

Bash
python preprocess.py
3. Huấn luyện các mô hình (Training)
Hệ thống hỗ trợ huấn luyện độc lập cho cả giai đoạn Phát hiện và Nhận dạng:

Bash
# Huấn luyện mô hình phát hiện vị trí biển số (YOLOv11)
python train_yolo.py

# Huấn luyện mô hình nhận dạng ký tự (CRNN)
python train_crnn.py

# Huấn luyện mô hình nhận dạng ký tự (PARSeq)
python train_parseq.py
4. Đánh giá mô hình (Evaluation)
Để kiểm tra các chỉ số học máy (Accuracy, Precision, Recall, mAP...) trên tập dữ liệu kiểm thử độc lập:

Bash
# Đánh giá YOLO
python eval_yolo.py

# Đánh giá CRNN
python eval_crnn.py

# Đánh giá PARSeq
python eval_parseq.py
5. Triển khai Ứng dụng Web Demo (Deployment)
Để chạy giao diện Web nhận diện cục bộ (Localhost), thực hiện lệnh sau:

Bash
python app.py
Sau đó, mở trình duyệt web và truy cập vào đường dẫn: http://127.0.0.1:5000/

🧠 Luồng xử lý chính của Hệ thống (Pipeline)
Đầu vào: Người dùng tải ảnh/video xe máy lên giao diện Web (app.py thông qua index.html).

Detection: Mô hình YOLOv11 (train_yolo.py) phát hiện và định vị tọa độ hộp bao (Bounding box) của biển số.

Crop & Preprocess: Hệ thống tiến hành cắt vùng ảnh chứa biển số và áp dụng các bộ lọc tiền xử lý ảnh (preprocess.py).

Recognition: Vùng ảnh biển số được chuyển qua mô hình nhận dạng CRNN (train_crnn.py) hoặc PARSeq (train_parseq.py) để trích xuất chuỗi văn bản text thô.

Post-processing: Chuỗi văn bản thô đi qua module vn_plate.py nhằm áp dụng thuật toán đối sánh mẫu và cấu trúc biển số xe máy Việt Nam để tự động phát hiện, sửa đổi các lỗi nhận diện ký tự phổ biến (nhầm lẫn giữa 8 và B, 0 và D, vị trí đặt chữ/số theo luật...).

Đầu ra: Trả về kết quả chuỗi biển số đã được tối ưu chuẩn xác lên màn hình giao diện.