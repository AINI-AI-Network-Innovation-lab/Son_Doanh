# FedCW: Client Selection with Adaptive Weight in Heterogeneous Federated Learning

> **Nguồn tham khảo:** *FedCW: Client Selection with Adaptive Weight in Heterogeneous Federated Learning* (Wu, Pei, & Li, 2026).

---

## 1. Ý tưởng cốt lõi & Vấn đề giải quyết

Trong môi trường biên dị thể và mạng xe cộ (Vehicular Networks), hai thách thức lớn nhất của học liên kết (FL) là:
1. **Băng thông và tài nguyên hạn chế:** Không thể để toàn bộ client tham gia ở mỗi vòng, nhưng nếu chọn client ngẫu nhiên (như FedAvg) sẽ bỏ lỡ các cập nhật giá trị.
2. **Sự mất cân bằng giữa Chọn client (Client Selection) và Gán trọng số (Weight Allocation):** Hai khâu này thường bị tách rời, khiến mô hình bị thiên lệch khi dữ liệu phân tán không đồng nhất (**Non-IID**).

**FedCW** thống nhất hai bài toán trên vào một khung tối ưu hóa chung dựa trên **khoảng cách Euclidean** giữa mô hình cục bộ và mô hình toàn cục:
- **Giả thuyết cốt lõi:** Mô hình cục bộ nào có khoảng cách càng xa mô hình toàn cục (Euclidean distance lớn) thì chứng tỏ client đó đang sở hữu dữ liệu mang tính đặc trưng, đa dạng và khác biệt nhất. Ưu tiên các client này sẽ giúp mô hình toàn cục nhanh chóng thích ứng với toàn bộ miền dữ liệu và chống overfit vào đám đông.
- **Tiết kiệm tài nguyên:** Ban đầu huy động nhiều client để thu nạp đa dạng thông tin, sau đó giảm dần số lượng client tham gia theo hàm suy giảm số mũ khi mô hình đã tiệm cận hội tụ.

---

## 2. Các ký hiệu sử dụng

| Ký hiệu | Ý nghĩa |
| :--- | :--- |
| $N$ | Tổng số lượng client trong toàn bộ hệ thống |
| $\mathcal{K}_t$ | Tập hợp các client được lựa chọn tham gia huấn luyện ở vòng $t$ |
| $n_t$ | Số lượng client được chọn tại vòng $t$ ($n_t = |\mathcal{K}_t|$) |
| $w$ (hoặc $w^t$) | Bộ tham số của mô hình toàn cục tại server |
| $w_i$ (hoặc $w_i^t$) | Bộ tham số mô hình cục bộ của client $i$ sau huấn luyện |
| $d_i$ | Khoảng cách Euclidean giữa mô hình client $i$ và mô hình toàn cục $w$ |
| $\alpha$ | Tỷ lệ client tham gia ban đầu ($\alpha \in (0, 1]$) |
| $\lambda$ | Hệ số tốc độ suy giảm số lượng client theo thời gian ($\lambda > 0$) |
| $min\_clients$ | Ngưỡng số lượng client tham gia tối thiểu mỗi vòng |
| $n_k$ | Số lượng mẫu dữ liệu của client $k$ |
| $\beta$ | Siêu tham số điều chỉnh mức độ ảnh hưởng của khoảng cách Euclidean đến trọng số |
| $a_k$ (hoặc $a_k^t$) | Trọng số tổng hợp thích ứng của client $k$ |

---

## 3. Các bước thực hiện

### Bước 1 — Đo khoảng cách phân kỳ Euclidean ($d_i$) tại Server
Sau khi nhận được mô hình cục bộ từ các client, Server trực tiếp tính toán khoảng cách Euclidean mà không cần client gửi thêm thông tin:

$$
d_i = \|w - w_i\|_2 = \sqrt{\sum_k (w_k - w_{i,k})^2}
$$

Server sắp xếp danh sách các client theo thứ tự $d_i$ **giảm dần** (từ lớn nhất đến nhỏ nhất).

### Bước 2 — Chọn lọc client tham gia (Dynamic Client Selection)
Để tối ưu hóa chi phí giao tiếp và thích ứng theo từng giai đoạn huấn luyện:
- Tỷ lệ chọn client $\alpha_t$ suy giảm theo hàm mũ qua số vòng $t$:

$$
\alpha_t = \alpha \cdot \exp(-\lambda t)
$$

- Số lượng client $n_t$ được lựa chọn tham gia ở vòng $t$ là:

$$
n_t = \max(\lceil N \cdot \alpha_t \rceil, min\_clients)
$$

Thuật toán sẽ chọn ra **$n_t$ client có khoảng cách $d_i$ lớn nhất** để đưa vào tập $\mathcal{K}_t$. Cơ chế này đảm bảo:
- *Giai đoạn đầu:* Lấy số lượng client lớn để tiếp nạp tối đa sự đa dạng dữ liệu.
- *Giai đoạn sau:* Thu hẹp số client tham gia xuống mức tối thiểu ($min\_clients$) để tập trung tinh chỉnh hội tụ, tiết kiệm băng thông mạng.

### Bước 3 — Phân bổ trọng số thích ứng (Adaptive Weight Allocation)
Thay vì chỉ gán trọng số theo kích thước dữ liệu $n_k$ như FedAvg, FedCW kết hợp cả **quy mô dữ liệu** và **độ phân kỳ mô hình ($d_k$)**:

$$
a_k = \frac{n_k \cdot \exp(\beta d_k)}{\sum_{i \in \mathcal{K}_t} n_i \cdot \exp(\beta d_i)}
$$

- **Khi $\beta > 0$:** Client có khoảng cách $d_k$ càng lớn (mô hình càng phân kỳ so với toàn cục) sẽ nhận trọng số càng cao. Điều này thúc đẩy mô hình toàn cục hấp thụ nhanh các phân phối hiếm/thiểu số dưới điều kiện Non-IID nặng.
- **Khi $\beta < 0$:** Client có khoảng cách nhỏ hơn được ưu tiên, giúp mô hình hội tụ mượt mà về vùng ổn định trung tâm nếu dữ liệu có nhiều ngoại lai.

### Bước 4 — Tổng hợp mô hình toàn cục
Mô hình toàn cục được cập nhật từ các client trong tập $\mathcal{K}_t$:

$$
w^{t+1} = \sum_{k \in \mathcal{K}_t} a_k w_k^{t+1}
$$

---

## 4. Phân tích lý thuyết hội tụ

Trong bài báo gốc, nhóm tác giả đã chứng minh toán học (Theorem 1) về cận dừng ergodic (Ergodic Stationarity Bound) của thuật toán:

$$
\frac{1}{T} \sum_{t=0}^{T-1} \mathbb{E}\|\nabla F(w^t)\|^2 \le \frac{4(F(w^0) - F^*)}{\eta T} + 3\bar{\varepsilon}^2 + 6L\eta\sigma^2
$$

- Đại lượng $\bar{\varepsilon}^2$ phản ánh độ lệch sinh ra từ việc chỉ chọn một tập con client ($\mathcal{K}_t \subset [N]$) và gán trọng số không đồng nhất $a_k$.
- Việc duy trì $min\_clients$ hợp lý và điều hòa $|\beta|$ giúp chặn cận sai số $\bar{\varepsilon}^2$, đảm bảo thuật toán đạt tốc độ hội tụ $\mathcal{O}(1/\sqrt{T})$ dưới điều kiện hàm mục tiêu không lồi (non-convex).

---

## 5. Ưu điểm & Nhược điểm kỹ thuật

### Ưu điểm:
1. **Tiết kiệm mạnh mẽ chi phí giao tiếp và thời gian:** Thực nghiệm cho thấy FedCW giảm **33% – 35%** dung lượng truyền thông và rút ngắn **17%** thời gian huấn luyện so với FedAvg trên MNIST và CIFAR-10.
2. **Khai thác hiệu quả dữ liệu Non-IID:** Bằng cách chủ động chọn và tăng trọng số cho các mô hình có độ phân kỳ cao, thuật toán tránh được bẫy hội tụ cục bộ của các lớp đa số.
3. **Server-side computing:** Toàn bộ việc tính khoảng cách Euclidean và sắp xếp được xử lý tại Server ($O(Nd)$ với $N$ client và $d$ chiều tham số), không tạo gánh nặng tính toán cho client biên.

### Nhược điểm:
1. **Dễ tổn thương trước Byzantine / Adversarial Attack:** Giả định "càng phân kỳ càng tốt" có thể bị lợi dụng. Kẻ tấn công chỉ cần gửi lên các bộ tham số nhiễu hoặc đối nghịch có $d_i$ cực lớn để vừa được chọn tham gia, vừa chiếm trọng số áp đảo, phá hủy hoàn toàn mô hình toàn cục.
2. **Rủi ro rò rỉ quyền riêng tư (Side-channel Privacy):** Xếp hạng dựa trên khoảng cách tham số có thể gián tiếp làm lộ đặc trưng phân phối dữ liệu của các client.
3. **Nhạy cảm với siêu tham số:** Tốc độ suy giảm $\lambda$ và hệ số trọng số $\beta$ cần được tinh chỉnh cẩn thận tùy thuộc vào mức độ Non-IID của hệ thống.
