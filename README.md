🍄 Ứng dụng Web Phân loại Nấm bằng ID3

Ứng dụng web đơn giản dùng **Flask** và mô hình **Decision Tree (ID3)** để phân loại nấm trong dataset UCI thành **Ăn được** hoặc **Độc**.  
⚠️ **Chỉ phục vụ mục đích học tập, KHÔNG sử dụng để quyết định ăn nấm thật ngoài đời.**

---

## 🚀 Chức năng
- Huấn luyện mô hình cây quyết định ID3 từ tập dữ liệu `mushrooms.csv`.
- Giao diện web thân thiện (Bootstrap 5).
- Người dùng nhập đặc trưng (odor, cap-color, gill-color) → dự đoán kết quả.
- Hiển thị kết quả với màu sắc cảnh báo (xanh = ăn được, đỏ = độc).

---

## 📂 Cấu trúc thư mục
├── mushrooms.csv # Dataset UCI
├── train_model_ID3.py # Script huấn luyện ID3 + lưu model
├── mushroom_model.pkl # Mô hình đã train (tự sinh ra sau khi chạy train_model_ID3.py)
├── app_id3.py # Flask web app
└── templates/
└── index.html # Giao diện web (Bootstrap 5)

<img width="1789" height="649" alt="image" src="https://github.com/user-attachments/assets/f8bd295e-e962-4f4b-a155-ca9802d9e803" />
