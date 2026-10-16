# Kỹ Thuật Sử Dụng Trong FedNova + MOON

## 1. Học tương phản cấp mô hình (Model-Contrastive Learning - MOON)
Áp dụng học tương phản ở tầng biểu diễn đặc trưng tại Client. Mô hình cục bộ học cách kéo biểu diễn lại gần mô hình toàn cục (cặp dương) và đẩy xa khỏi mô hình cục bộ của chính nó ở round trước (cặp âm), khắc phục hiện tượng trôi mô hình do dữ liệu Non-IID mà không làm lộ dữ liệu thô.

## 2. Quản lý trạng thái cục bộ qua Context State
Lưu trữ trọng số mô hình round trước trực tiếp trong bộ nhớ trạng thái của Client trên thiết bị qua các round. Kỹ thuật này giúp client có đầy đủ thông tin để so sánh tương phản mà hoàn toàn không tốn thêm chi phí băng thông truyền tải qua mạng.

## 3. Chặn lan truyền ngược tối ưu bộ nhớ (Gradient Detachment)
Sử dụng cơ chế ngắt gradient khi forward qua mô hình toàn cục và mô hình vòng trước. Thao tác này ngăn GPU lưu trữ các ma trận kích hoạt trung gian, giải phóng đáng kể bộ nhớ VRAM và tăng tốc độ xử lý khi phải chạy cùng lúc ba mô hình.

## 4. Chuẩn hóa độ dời theo bước chạy thực tế (FedNova Normalization)
Client chia độ dời trọng số cho lượng learning rate tích lũy thực tế đã chạy trước khi gửi về server. Kỹ thuật này triệt tiêu sai lệch mục tiêu khi các client thực hiện số bước tính toán không bằng nhau do giới hạn phần cứng.

## 5. Tổng hợp theo bước nhảy hiệu dụng (Effective Step Aggregation)
Server tính toán bước nhảy hiệu dụng tự động dựa trên trung bình có trọng số của các client tham gia. Mô hình toàn cục tiến bước đồng đều và hội tụ chính xác về nghiệm tối ưu toàn cục.

## 6. Phân vùng dữ liệu Non-IID Dirichlet
Chia tập dữ liệu CIFAR-10 cho các client theo phân phối Dirichlet với hệ số tập trung 0.5, tạo ra sự mất cân bằng về cả tỷ lệ nhãn lẫn số lượng mẫu giữa các phân vùng nhằm phản ánh dữ liệu thực tế.

## 7. Mô phỏng thiết bị biên không đồng nhất (System Heterogeneity Simulation)
Bốc ngẫu nhiên số bước huấn luyện thực tế cho từng client trong khoảng từ mức sàn tối thiểu đến số batch tối đa của epoch, giả lập tình trạng thiết bị biên yếu bị timeout hoặc gián đoạn kết nối.

## 8. Đóng gói dữ liệu Flower Next-gen API
Phân tách độc lập dữ liệu tensor vào ArrayRecord và dữ liệu số vô hướng vào MetricRecord bên trong Message, tối ưu hóa quá trình serialize và truyền thông giữa ClientApp và ServerApp.
