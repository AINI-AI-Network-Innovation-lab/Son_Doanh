# Kỹ Thuật Sử Dụng Trong FedNova + SCAFFOLD

## 1. Hiệu chỉnh gradient bằng biến kiểm soát (Control Variates - SCAFFOLD)
Sử dụng hai biến kiểm soát cục bộ và toàn cục để khử thành phần lệch gradient (Client Drift) phát sinh do dữ liệu Non-IID giữa các client, hướng gradient cục bộ đi theo hướng của bài toán toàn cục.

## 2. Quản lý biến kiểm soát cục bộ qua Context State
Lưu trữ biến kiểm soát của riêng client trực tiếp trong bộ nhớ trạng thái trên thiết bị qua các round. Kỹ thuật này giúp client duy trì và theo dõi liên tục độ lệch gradient của chính mình qua các vòng huấn luyện.

## 3. Truyền độ lệch biến kiểm soát (Delta Control Variate)
Client chỉ gửi phần độ lệch thay đổi của biến kiểm soát về Server thay vì gửi toàn bộ giá trị. Server cộng dồn phần độ lệch này để cập nhật biến toàn cục mà không cần lưu trữ bảng trạng thái của tất cả client trong mạng.

## 4. Chuẩn hóa độ dời theo bước chạy thực tế (FedNova Normalization)
Client chia độ dời trọng số cho lượng learning rate tích lũy thực tế đã chạy trước khi gửi về server. Kỹ thuật này loại bỏ sai lệch mục tiêu khi các client thực hiện số bước huấn luyện không đồng đều do khác biệt phần cứng.

## 5. Tổng hợp theo bước nhảy hiệu dụng (Effective Step Aggregation)
Server tính toán bước nhảy hiệu dụng tự động dựa trên trung bình có trọng số số bước của các client tham gia. Mô hình toàn cục tiến bước đồng đều và hội tụ chính xác về nghiệm tối ưu toàn cục.

## 6. Phân vùng dữ liệu Non-IID Dirichlet
Chia tập dữ liệu CIFAR-10 cho các client theo phân phối Dirichlet với hệ số tập trung 0.5, tạo ra sự mất cân bằng về cả tỷ lệ nhãn lẫn số lượng mẫu giữa các phân vùng nhằm phản ánh dữ liệu thực tế.

## 7. Mô phỏng thiết bị biên không đồng nhất (System Heterogeneity Simulation)
Bốc ngẫu nhiên số bước huấn luyện thực tế cho từng client trong khoảng từ mức sàn tối thiểu đến số batch tối đa của epoch, giả lập tình trạng thiết bị biên yếu bị timeout hoặc gián đoạn kết nối.

## 8. Đóng gói dữ liệu Flower Next-gen API
Phân tách độc lập dữ liệu trọng số mô hình và biến kiểm soát vào ArrayRecord, dữ liệu số vô hướng vào MetricRecord bên trong Message, tối ưu hóa quá trình serialize và truyền thông giữa ClientApp và ServerApp.
