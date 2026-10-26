# ShapFed-WA (Shapley-Driven Federated Learning with Weighted Aggregation)

> **Paper**: *Redefining Contributions: Shapley-Driven Federated Learning* ([arXiv:2406.00569](https://arxiv.org/abs/2406.00569))  
> **Code**: [https://github.com/tnurbek/shapfed](https://github.com/tnurbek/shapfed)

---

### Ý tưởng cốt lõi

- Ước lượng mức đóng góp của từng client lấy ý tưởng từ **Shapley Value** trong lý thuyết trò chơi hợp tác.
- Khắc phục hạn chế của Shapley truyền thống (đòi hỏi tập validation data trên server và bùng nổ tổ hợp $\mathcal{O}(2^n)$) và CGSV (tính cosine trên toàn bộ mạng bị nhiễu bởi các tầng feature extractor): **ShapFed chỉ trích xuất tầng phân loại cuối cùng ($\hat{w}$)** để đo đạc đóng góp riêng cho từng class (**CSSV - Class-Specific Shapley Values**).

---

### Các công thức chính

**1. Tính độ tương đồng theo từng class ở tầng cuối:**

Ở mỗi round, client gửi trọng số $w_i$ lên server. Server trích xuất tầng phân loại cuối cùng $\hat{w}_i$ và tính độ tương đồng với $\hat{w}_s$:

$$
\Gamma_i = \left[ \cos(\hat{w}_{i,1}, \hat{w}_{s,1}), \; \cos(\hat{w}_{i,2}, \hat{w}_{s,2}), \; \dots, \; \cos(\hat{w}_{i,M}, \hat{w}_{s,M}) \right]
$$

với cosine similarity:

$$
\cos(a, b) = \frac{a^T b}{\|a\|_2 \|b\|_2}
$$

**2. Làm mượt ma trận đóng góp qua các round (tránh nhiễu đột ngột):**

- $\Gamma^{(1)} = \tilde{\Gamma}^{(1)}$
- $\Gamma^{(t)} = \mu \Gamma^{(t-1)} + (1 - \mu) \tilde{\Gamma}^{(t)}$ (với $\mu \in [0, 1)$)

**3. Chuẩn hóa điểm đóng góp & Tính trọng số client:**

- Điểm đóng góp trung bình của client $i$ qua $M$ classes:

$$
\gamma_i = \frac{1}{M} \sum_{j=1}^M \frac{1 + \Gamma_{i,j}}{2} \in [0, 1]
$$

- Trọng số tổng hợp chuẩn hóa:

$$
\tilde{\gamma}_i = \frac{\gamma_i}{\sum_{k=1}^n \gamma_k} \implies \text{Trọng số của client } i \text{ khi tổng hợp}
$$

**4. Tổng hợp mô hình toàn cục (ShapFed-WA):**

$$
w_s^{t+1} = \sum_{i=1}^n \tilde{\gamma}_i w_i^t
$$

**5. Cá nhân hóa & Đảm bảo công bằng (Personalization & Collaborative Fairness):**

Server phát mô hình riêng biệt cho từng client dựa trên mức đóng góp $\gamma_i$:

$$
\bar{w}_i = \gamma_i w_s + (1 - \gamma_i) w_i
$$

- Client đóng góp nhiều ($\gamma_i \to 1$) $\to$ Nhận mô hình chất lượng cao gần $w_s$.
- Client ăn bám (free-rider) hoặc phá hoại (Byzantine) ($\gamma_i \to 0$) $\to$ Bị phạt giữ lại mô hình yếu kém của chính nó.

---

### Đánh giá nhanh

- **Ưu điểm**:
  - Đánh giá đóng góp chi tiết theo từng class, không cần tập validation trên server.
  - Chi phí tính toán nhẹ $\mathcal{O}(n + 1)$, không bị bùng nổ tổ hợp.
  - Cơ chế cá nhân hóa tự nhiên giúp thưởng công bằng và tự động triệt tiêu free-riders / Byzantine.
- **Nhược điểm**: Cần chia ma trận tầng cuối theo từng class để tính cosine tương ứng.
