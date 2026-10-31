# FedDAWA: Federated Dual Adaptive Weight Aggregation

> **Nguồn tham khảo:** *U-FedTomAtt: Ultra-lightweight Federated Learning with Attention for Tomato Disease Recognition* (George et al., 2026).

---

## 1. Ý tưởng cốt lõi & Vấn đề giải quyết

Trong học liên kết (FL) truyền thống như **FedAvg**, trọng số tổng hợp của mỗi client chỉ phụ thuộc hoàn toàn vào số lượng mẫu dữ liệu cục bộ ($N_k / N_{total}$). Điều này dẫn đến hai nghịch lý lớn trong môi trường dữ liệu **Non-IID** hoặc mất cân bằng:
- **Client dữ liệu lớn nhưng chất lượng kém:** Một client nắm giữ số lượng mẫu áp đảo nhưng dữ liệu bị nhiễu, nhãn sai hoặc mô hình học kém (loss cao) sẽ nhận trọng số rất lớn, kéo lệch mô hình toàn cục (**client drift**).
- **Client dữ liệu ít nhưng chất lượng cao:** Một client sở hữu dữ liệu đặc trưng quan trọng và mô hình học rất tốt (loss thấp) lại bị đánh giá thấp chỉ vì số lượng mẫu ít.

Ngược lại, nếu chỉ dựa vào hiệu năng mô hình (validation loss) mà bỏ qua số lượng mẫu, mô hình toàn cục sẽ mất đi tính đại diện thống kê từ các tập dữ liệu lớn.

**FedDAWA** giải quyết mâu thuẫn này bằng cách kết hợp lồi (**convex combination**) giữa **quy mô dữ liệu (Data Quantity)** và **chất lượng học tập cục bộ (Model Quality)**, thông qua hệ số thích ứng $\beta \in [0, 1]$ để cân bằng động giữa hai yếu tố trong quá trình tổng hợp toàn cục.

---

## 2. Các ký hiệu sử dụng

| Ký hiệu | Ý nghĩa |
| :--- | :--- |
| $K$ | Tổng số client tham gia huấn luyện |
| $\mathcal{D}_k$ (hoặc $N_k$) | Kích thước tập dữ liệu huấn luyện cục bộ của client $k$ |
| $\mathcal{D}$ (hoặc $N_{total}$) | Tổng số mẫu dữ liệu của toàn bộ $K$ client: $N_{total} = \sum_{k=1}^K N_k$ |
| $L_k$ | Hàm mất mát kiểm định cục bộ (Local validation loss) của client $k$ sau $E$ epoch |
| $P_k$ | Thành phần tỷ trọng kích thước dữ liệu của client $k$ |
| $Q_k$ | Thành phần tỷ trọng chất lượng mô hình (nghịch đảo loss) của client $k$ |
| $\beta$ | Siêu tham số điều tiết trade-off giữa kích thước dữ liệu và chất lượng mô hình ($\beta \in [0, 1]$) |
| $\alpha_k$ | Trọng số tổng hợp cuối cùng của client $k$ |
| $\epsilon$ | Hằng số làm mịn nhằm chống chia cho $0$ ($\epsilon = 10^{-6}$) |

---

## 3. Các bước thực hiện

### Bước 1 — Tính thành phần tỷ trọng kích thước dữ liệu ($P_k$)
Thành phần này đại diện cho mức độ đóng góp theo quy mô dữ liệu của từng client:

$$
P_k = \frac{\mathcal{D}_k}{\mathcal{D}} = \frac{N_k}{N_{total}}
$$

Tổng tỷ trọng trên toàn bộ $K$ client luôn bằng $1$: $\sum_{k=1}^K P_k = 1$.

### Bước 2 — Tính thành phần nghịch đảo Validation Loss ($Q_k$)
Để đánh giá chất lượng học tập cục bộ, thuật toán sử dụng giá trị nghịch đảo của loss kiểm định $L_k$. Client nào đạt loss càng thấp (học càng tốt, dự đoán càng chuẩn xác) thì giá trị nghịch đảo $1/(L_k + \epsilon)$ càng lớn, sau đó được chuẩn hóa trên toàn bộ $K$ client:

$$
Q_k = \frac{\frac{1}{L_k + \epsilon}}{\sum_{j=1}^K \frac{1}{L_j + \epsilon}}
$$

Tổng thành phần chất lượng cũng được chuẩn hóa về $1$: $\sum_{k=1}^K Q_k = 1$.

### Bước 3 — Tổ hợp lồi thích ứng tạo trọng số $\alpha_k$
Trọng số đóng góp của client $k$ được xác định bằng tổ hợp lồi giữa hai thành phần $P_k$ và $Q_k$:

$$
\alpha_k = \beta \cdot P_k + (1 - \beta) \cdot Q_k
$$

- **Khi $\beta = 1$:** Trọng số hoàn toàn phụ thuộc vào kích thước dữ liệu (trở về cơ chế giống FedAvg).
- **Khi $\beta = 0$:** Trọng số hoàn toàn phụ thuộc vào chất lượng mô hình (nghịch đảo loss).
- **Khi $0 < \beta < 1$:** Thuật toán cân bằng thích ứng giữa số lượng và chất lượng học tập.

Do cả $P_k$ và $Q_k$ đều đã được chuẩn hóa có tổng bằng 1, tổng trọng số của tất cả các client tự nhiên thỏa mãn:

$$
\sum_{k=1}^K \alpha_k = \beta \sum_{k=1}^K P_k + (1 - \beta) \sum_{k=1}^K Q_k = \beta(1) + (1 - \beta)(1) = 1
$$

### Bước 4 — Tổng hợp mô hình toàn cục
Mô hình toàn cục tại vòng giao tiếp tiếp theo được cập nhật thông qua trung bình có trọng số:

$$
w^{t+1} = \sum_{k=1}^K \left( \frac{\alpha_k}{\sum_{j=1}^K \alpha_j} \right) w_k^{t+1} = \sum_{k=1}^K \alpha_k w_k^{t+1}
$$

Hàm mục tiêu toàn cục tối ưu hóa tương ứng:

$$
F(w) = \sum_{k=1}^K \alpha_k F_k(w)
$$

---

## 4. Vai trò của siêu tham số $\beta$ theo mức độ Non-IID

Thực nghiệm trong nghiên cứu chỉ ra rằng giá trị $\beta$ tối ưu phụ thuộc chặt chẽ vào mức độ phân tán và không đồng nhất (Non-IID) của dữ liệu giữa các client:

- **Khi dữ liệu Non-IID rất mạnh (ví dụ SLIF-Tomato thu thập thực tế ngoài đồng ruộng):** $\beta$ tối ưu ở mức cao ($\beta \approx 0.8$). Lúc này, việc duy trì trọng số dữ liệu lớn giúp mô hình bao quát được nhiều dạng biến thể dữ liệu phức tạp, tránh việc bị chi phối quá mức bởi một vài client có loss thấp giả tạo.
- **Khi dữ liệu phân phối tương đối đồng đều / ít nhiễu (ví dụ PlantVillage thu trong phòng thí nghiệm):** $\beta$ tối ưu ở mức thấp ($\beta \approx 0.2$). Lúc này chất lượng mô hình (loss thấp) phản ánh chính xác năng lực học tập, việc ưu tiên $Q_k$ giúp mô hình toàn cục hội tụ nhanh và sắc nét hơn.

---

## 5. Ưu điểm & Nhược điểm kỹ thuật

### Ưu điểm:
1. **Chi phí truyền thông và tính toán gần như bằng 0:** Client chỉ cần gửi thêm một đại lượng vô hướng duy nhất là validation loss $L_k$ cùng với trọng số mô hình. Server tính toán tổ hợp tuyến tính trong $O(K)$.
2. **Kìm hãm hiện tượng client drift:** Ngăn chặn các client có tập dữ liệu lớn nhưng kém chất lượng thao túng hướng cập nhật của mô hình toàn cục.
3. **Linh hoạt và dễ tích hợp:** Có thể áp dụng ngay trên bất kỳ kiến trúc mạng nơ-ron nào (CNN, Transformer, MobileNet) mà không cần can thiệp vào hàm mất mát cục bộ như FedProx hay thêm biến điều khiển như SCAFFOLD.

### Nhược điểm:
1. **Phụ thuộc vào tập validation cục bộ:** Nếu tập validation của client quá nhỏ hoặc không đại diện, chỉ số $L_k$ có thể bị dao động mạnh hoặc bị overfit cục bộ (loss rất nhỏ nhưng thực tế mô hình kém khái quát hóa).
2. **Nguy cơ bảo mật (Adversarial/Byzantine Attack):** Client ác ý có thể cố tình gửi giá trị loss giả mạo cực nhỏ ($L_k \to 0$) để chiếm trọn trọng số tổng hợp của toàn hệ thống, đòi hỏi cần thêm cơ chế kiểm định/chặn ngưỡng loss bất thường.
