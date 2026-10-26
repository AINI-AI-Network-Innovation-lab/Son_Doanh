# FedAWA (Federated Adaptive Weight Aggregation)

> **Paper**: *FedAWA: Adaptive Optimization of Aggregation Weights in Federated Learning Using Client Vectors* ([arXiv:2503.15842](https://arxiv.org/abs/2503.15842))  
> **Code**: [https://github.com/ChanglongShi/FedAWA](https://github.com/ChanglongShi/FedAWA)

---

### Ý tưởng cốt lõi

- Lấy cảm hứng từ **Task Arithmetic** trong Model Merging: sự thay đổi tham số sau local training (client vector) phản ánh trực tiếp phân phối dữ liệu của client mà không làm lộ dữ liệu thô.
- Thay vì gán trọng số tĩnh theo cỡ mẫu như FedAvg (dễ bị kéo lệch khi non-IID) hoặc cần proxy data như FedLAW, FedAWA **tự tối ưu hóa động trọng số tổng hợp $\lambda_k^t$ ở mỗi round** dựa trên độ thẳng hàng của client vector so với global vector.

---

### Các công thức chính

**1. Client vector & Global vector:**

- $\tau_k^t = w_k^t - w_g^t$: Client vector, độ lệch giữa trọng số local $w_k^t$ và global $w_g^t$ ở round $t$.
- $\tau_g^t = \sum_{k=1}^K \lambda_k^t \tau_k^t$: Cập nhật global vector, với $\lambda_k^t$ là trọng số của client $k$.

**2. Hàm mục tiêu tối ưu trọng số tại Server:**

Server giải bài toán tối ưu tìm vector trọng số $\boldsymbol{\lambda}^t$ ($\sum_{k=1}^K \lambda_k = 1, \lambda_k \ge 0$):

$$
\boldsymbol{\lambda}^t = \arg\min_{\boldsymbol{\lambda}} \sum_{k=1}^K \lambda_k \|\tau_k^t - \tau_g^t\|_2 + d\left( \sum_{k=1}^K \lambda_k w_k^t, \; w_g^t \right)
$$

- **Biểu thức đầu** ($`\sum_{k} \lambda_{k} \Vert \tau_{k}^{t} - \tau_{g}^{t} \Vert_{2}`$): Khoảng cách của global vector so với client vector, nhân với độ ảnh hưởng của mỗi client $`\to`$ Client nào có hướng cập nhật gần với hướng chung sẽ được gán trọng số cao hơn, client lệch hướng bị phạt giảm trọng số.
- **Biểu thức sau** ($`d\left(\sum_{k} \lambda_{k} w_{k}^{t},\; w_{g}^{t}\right)`$): Thành phần regularization giữ ổn định, tránh việc mô hình sau cập nhật bị trôi lệch quá xa so với global model cũ.
- **Hàm khoảng cách**: $`d(a, b) = 1 - \cos(a, b) = 1 - \dfrac{a \cdot b}{\Vert a \Vert \, \Vert b \Vert}`$.

**3. Tổng hợp mô hình toàn cục:**

$$
w_g^{t+1} = \sum_{k=1}^K \lambda_k^t w_k^t
$$

**4. Biến thể tầng mạng (FedAWA-L):**

Tối ưu riêng từng bộ trọng số $\boldsymbol{\lambda}_l^t$ cho từng layer $l$ trong mạng:

$$
w_{gl}^{t+1} = \sum_{k=1}^K \lambda_{kl}^t w_{kl}^t
$$

---

### Đánh giá nhanh

- **Ưu điểm**:
  - Không cần dataset proxy trên server (bảo mật dữ liệu tuyệt đối).
  - Trọng số thích ứng động theo từng round, chống client drift tốt hơn FedAvg/FedDisco.
  - Nhẹ hơn nhiều so với FedLAW và L-DAWA.
- **Nhược điểm**: Cần chạy vài bước gradient descent trên server để giải bài toán tối ưu $\boldsymbol{\lambda}$; biến thể FedAWA-L tốn thêm thời gian tính toán cho từng layer.
