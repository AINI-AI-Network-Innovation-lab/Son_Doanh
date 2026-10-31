# FedWAvg: Weighted Averaging Federated Learning Based on Example Forgetting Events

> **Nguồn tham khảo:** *Weighted Averaging Federated Learning Based on Example Forgetting Events in Label Imbalanced Non-IID* (Hong, Kang, & Lee, *Applied Sciences*, 2022).

---

## 1. Ý tưởng cốt lõi & Vấn đề giải quyết

Trong học sâu, **sự kiện quên mẫu (Example Forgetting Event)** là hiện tượng một mẫu dữ liệu đã từng được mô hình phân loại chính xác ở vòng/epoch trước, nhưng sau khi cập nhật trọng số lại bị phân loại sai ở vòng hiện tại. Các mẫu trải qua hiện tượng này được gọi là **mẫu dễ quên (forgettable examples)** — chúng thường là các trường hợp khó học (hard samples), mẫu nằm ở biên phân tách hoặc thuộc các lớp thiểu số.

Trong học liên kết (FL) với dữ liệu **mất cân bằng nhãn (Label-Imbalanced Non-IID)**:
- Mô hình toàn cục tổng hợp theo FedAvg thường bị chi phối bởi các client có lớp đa số, dẫn đến hiện tượng **quên lãng thảm họa (catastrophic forgetting)** đối với các đặc trưng của lớp thiểu số.
- Khi các client nhận mô hình toàn cục về huấn luyện tiếp, các client sở hữu lớp thiểu số sẽ thấy số lượng mẫu bị đoán sai trở lại tăng vọt.

**FedWAvg** giải quyết vấn đề này bằng nguyên lý đơn giản nhưng hiệu quả:
> *Client nào có số lượng mẫu bị quên ($x_k$) càng nhiều, chứng tỏ tri thức của client đó đang bị mô hình toàn cục "bỏ rơi" hoặc dữ liệu client đó đang chịu áp lực lớn nhất. Cần chủ động **tăng trọng số tổng hợp** cho client này để ép mô hình toàn cục chú ý và duy trì các đặc trưng đó.*

---

## 2. Các ký hiệu sử dụng

| Ký hiệu | Ý nghĩa |
| :--- | :--- |
| $N$ | Tổng số lượng client tham gia huấn luyện |
| $x_k$ | Số lượng mẫu dễ quên (forgettable examples) được ghi nhận tại client $k$ |
| $\text{norm}(x_k)$ | Tỷ lệ chuẩn hóa số lượng mẫu dễ quên của client $k$ trên toàn hệ thống |
| $\alpha$ | Siêu tham số điều chỉnh mức độ ưu tiên theo tỷ lệ mẫu bị quên ($\alpha \in [0, 1]$) |
| $r_k$ | Hệ số đóng góp tương đối đã được điều chỉnh của client $k$ |
| $w_k$ | Trọng số tổng hợp chính thức của client $k$ tại server |
| $w_k^{local}$ | Bộ tham số mô hình sau huấn luyện cục bộ của client $k$ |
| $w^{global}$ | Bộ tham số mô hình toàn cục mới sau tổng hợp |

---

## 3. Các bước thực hiện

### Bước 1 — Đếm số lượng mẫu dễ quên ($x_k$) tại Client
Trong suốt quá trình huấn luyện cục bộ, client $k$ theo dõi kết quả dự đoán của từng mẫu dữ liệu $i$:
- Nếu mẫu $i$ ở bước trước được phân loại đúng ($y_i = \hat{y}_i$), nhưng sau khi cập nhật lại bị đoán sai ($y_i \neq \hat{y}_i$), mẫu $i$ được tính là một sự kiện quên (forgetting event).
- Tổng số mẫu bị rơi vào trạng thái này tại client $k$ được ký hiệu là $x_k$.
- Client $k$ chỉ cần gửi con số nguyên vô hướng $x_k$ cùng với trọng số mô hình lên Server.

### Bước 2 — Chuẩn hóa tỷ lệ mẫu dễ quên trên toàn Server
Server thu thập $x_k$ từ toàn bộ $N$ client và tính toán tỷ lệ chuẩn hóa:

$$
\text{norm}(x_k) = \frac{x_k}{\sum_{i=1}^N x_i}
$$

Tổng tỷ lệ chuẩn hóa của toàn bộ hệ thống luôn bằng 1: $\sum_{k=1}^N \text{norm}(x_k) = 1$.

### Bước 3 — Điều chỉnh hệ số tỷ trọng $r_k$ theo siêu tham số $\alpha$
Để kiểm soát mức độ tác động của các mẫu quên và tránh việc thiên lệch cực đoan, thuật toán sử dụng siêu tham số $\alpha \in [0, 1]$ để điều tiết:

$$
r_k = (1 - \alpha) + \alpha \cdot \Big( N \cdot \text{norm}(x_k) \Big)
$$

- **Khi $\alpha = 0$:** $r_k = 1$ với mọi client (mọi client có tiếng nói ngang bằng nhau, không phụ thuộc vào hiện tượng quên).
- **Khi $\alpha = 1$:** $r_k = N \cdot \text{norm}(x_k)$ (hoàn toàn phụ thuộc vào tỷ lệ mẫu quên).
- **Khi $0 < \alpha < 1$:** Kết hợp cân bằng giữa phân bổ đều và mức độ bù trừ mẫu quên.

Tổng hệ số $r_k$ của tất cả các client luôn được bảo toàn bằng $N$:

$$
\sum_{k=1}^N r_k = \sum_{k=1}^N (1 - \alpha) + \alpha N \sum_{k=1}^N \text{norm}(x_k) = N(1 - \alpha) + \alpha N (1) = N
$$

### Bước 4 — Xác định trọng số tổng hợp và cập nhật mô hình toàn cục
Trọng số tổng hợp cuối cùng của client $k$ được chuẩn hóa bằng cách chia $r_k$ cho $N$:

$$
w_k = \frac{r_k}{N} = \frac{1 - \alpha}{N} + \alpha \cdot \text{norm}(x_k)
$$

Đảm bảo tổng trọng số luôn bằng $1$: $\sum_{k=1}^N w_k = \frac{\sum_{k=1}^N r_k}{N} = \frac{N}{N} = 1$.
- Client có mức độ quên cao hơn trung bình ($\text{norm}(x_k) > 1/N$) sẽ nhận $w_k > 1/N$.
- Client có mức độ quên thấp hơn trung bình sẽ nhận $w_k < 1/N$.

Mô hình toàn cục mới được cập nhật:

$$
w^{global} = \sum_{k=1}^N w_k w_k^{local} = \sum_{k=1}^N \frac{r_k}{N} w_k^{local}
$$

---

## 4. Ưu điểm & Nhược điểm kỹ thuật

### Ưu điểm:
1. **Chi phí truyền thông tối thiểu & Bảo toàn quyền riêng tư:** Client chỉ gửi duy nhất một giá trị số nguyên $x_k$ lên Server. Không cần gửi ma trận nhãn, gradient thô hay phân phối dữ liệu, bảo đảm tính ẩn danh hoàn hảo.
2. **Trực tiếp khắc phục Catastrophic Forgetting:** Đặc biệt hiệu quả trong các bài toán phân loại có độ lệch nhãn lớn (như SVHN, CIFAR-10 Non-IID), giúp mô hình duy trì độ chính xác cao trên các lớp thiểu số mà không làm suy giảm các lớp đa số.
3. **Độc lập kiến trúc & Không can thiệp hàm loss:** Có thể tích hợp vào bất kỳ mô hình học máy nào mà không đòi hỏi chỉnh sửa hàm mất mát hay giải các bài toán tối ưu ràng buộc phức tạp tại client.

### Nhược điểm:
1. **Phát sinh bộ nhớ theo dõi tại Client:** Mỗi client cần duy trì bộ nhớ đệm (tracking cache) để lưu trữ nhãn dự đoán của các mẫu qua từng epoch nhằm xác định trạng thái chuyển dịch từ "đúng" sang "sai".
2. **Nhạy cảm với nhãn nhiễu (Label Noise):** Nếu một client chứa nhiều mẫu bị gán nhãn sai hoặc hình ảnh bị mờ, mô hình sẽ liên tục dự đoán dao động giữa đúng và sai một cách ngẫu nhiên. Điều này làm số lượng $x_k$ tăng cao giả tạo, vô tình khiến server gán trọng số ưu tiên lớn cho một client chứa dữ liệu rác.
