
---

# 👨‍💼 Real-Time Employee Face Attendance System

## 🎯 Mục tiêu dự án

Xây dựng hệ thống **điểm danh khuôn mặt thời gian thực** sử dụng camera, kết hợp với kỹ thuật **nhúng đặc trưng ảnh (feature embedding)** và công cụ tìm kiếm tương đồng **FAISS**, giúp nhận diện và điểm danh nhân viên một cách nhanh chóng, chính xác.

---

## 🚀 Tính năng chính

* **Trích xuất đặc trưng 2 cấp (2-level embedding):**

  * **Level 1 - Raw Pixel Vector:**

    * Ảnh khuôn mặt được chuyển sang vector bằng cách flatten mảng pixel sử dụng NumPy.
    * Mục đích: So sánh với phương pháp hiện đại để thấy rõ sự khác biệt trong chất lượng embedding.
  * **Level 2 - Deep Feature Vector:**

    * Sử dụng mô hình học sâu như `InceptionResnetV1` để trích xuất đặc trưng ảnh.
    * Mô hình được dùng như **feature extractor** (hộp đen), không huấn luyện lại.

* **Tìm kiếm ảnh tương đồng bằng FAISS:**

  * Ảnh đầu vào từ camera được vector hóa và truy vấn trong cơ sở dữ liệu bằng **FAISS** (Facebook AI Similarity Search).
  * Dựa trên độ tương đồng cosine hoặc L2 distance, nếu vượt qua ngưỡng định sẵn → xác định danh tính nhân viên.

* **Điểm danh thời gian thực với giao diện trực quan:**

  * Ứng dụng xây dựng bằng **Streamlit**.
  * Tự động hiển thị tên nhân viên tương ứng với ảnh khuôn mặt khi được camera ghi nhận.
  * Có thể mở rộng để ghi log thời gian điểm danh, lưu lịch sử, v.v.

---

## 🛠️ Công nghệ sử dụng

| Thành phần           | Công cụ / Thư viện                           |
| -------------------- | -------------------------------------------- |
| Trích xuất đặc trưng | `InceptionResnetV1` (from `facenet-pytorch`) |
| Nhúng ảnh            | `NumPy`, `PyTorch`                           |
| Tìm kiếm tương đồng  | `FAISS`                                      |
| Giao diện            | `Streamlit`                                  |
| Ngôn ngữ             | `Python`                                     |
| Camera               | `OpenCV`                                     |

---

## 🔄 Quy trình hoạt động

1. Ảnh khuôn mặt được lấy từ camera.
2. Trích xuất feature vector bằng `InceptionResnetV1`.
3. So sánh vector này với database vector đã lưu bằng FAISS.
4. Nếu khoảng cách dưới ngưỡng → Xác định người dùng và điểm danh.
5. Giao diện hiển thị tên và trạng thái điểm danh theo thời gian thực.

---

## 🧪 Mục đích thử nghiệm 2 cấp embedding

* **Level 1 (Raw pixel)** giúp minh họa việc embedding đơn giản cho ra kết quả kém chính xác.
* **Level 2 (Deep features)** cho độ chính xác cao hơn rõ rệt → chứng minh vai trò quan trọng của feature extraction trong các hệ thống nhận dạng khuôn mặt.

---

## 📌 Hướng phát triển tiếp theo

* Thêm chức năng ghi log điểm danh (tên, thời gian, trạng thái).
* Cho phép đăng ký khuôn mặt mới thông qua giao diện.
* Tối ưu hóa tốc độ nhận diện với batch FAISS hoặc GPU FAISS.
* Triển khai trên thiết bị biên (Edge devices) như Raspberry Pi, Jetson Nano.

---

## 🧑‍💻 Demo & Cài đặt

*Đang cập nhật…*

---


