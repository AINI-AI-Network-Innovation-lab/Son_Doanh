# FedNolowe (Federated Normalized Loss-based Weighted Aggregation)

> **Paper**: *FedNolowe: A normalized loss-based weighted aggregation strategy for robust federated learning in heterogeneous environments* (PLOS ONE 2025, [DOI: 10.1371/journal.pone.0322766](https://doi.org/10.1371/journal.pone.0322766))  
> **Code**: [https://github.com/dongld-2020/fednolowe](https://github.com/dongld-2020/fednolowe)

---

### Ý tưởng cốt lõi

- Gán trọng số động cho client dựa trên **training loss cục bộ** $L_k^t$ với giả định: **loss thấp $\to$ mô hình tối ưu tốt, ít bị trôi dạt (client drift) $\to$ xứng đáng được gán trọng số cao hơn**.
- Khắc phục nhược điểm của FedAsl (dùng phép chia nghịch đảo $1/L_k$ khiến trọng số bùng nổ vô hạn khi $L_k \to 0^+$): FedNolowe sử dụng **chuẩn hóa L1 hai giai đoạn kết hợp phép đảo ngược dựa trên phép trừ ($1 - \tilde{L}_k^t$)**.

---

### Các công thức chính

**1. Chuẩn hóa L1 tổn thất 2 bước (Two-Stage L1 Normalization):**

$$
\alpha_k^t = \frac{1 - \tilde{L}_k^t}{\sum_{j \in S^t} (1 - \tilde{L}_j^t)} \quad \text{where} \quad \tilde{L}_k^t = \frac{L_k^t}{\sum_{j \in S^t} L_j^t}
$$

- **Bước 1 (Chuẩn hóa L1 bên phải)**: $\tilde{L}_k^t = \frac{L_k^t}{\sum L_j^t} \in (0, 1)$ giúp thang đo loss độc lập giữa các vòng giao tiếp (scale invariance).
- **Bước 2 (Nghịch đảo phép trừ & Chuẩn hóa L1 lần 2)**: $1 - \tilde{L}_k^t$ lật ngược lại (loss càng nhỏ thì $1 - \tilde{L}_k^t$ càng lớn).
- **Rút gọn mẫu số**: Do $\sum_{j \in S^t} \tilde{L}_j^t = 1$, ta có mẫu số luôn bằng $|S^t| - 1$:

$$
\alpha_k^t = \frac{1 - \tilde{L}_k^t}{|S^t| - 1}
$$

**2. Tổng hợp mô hình toàn cục:**

$$
w^{t+1} = \sum_{k \in S^t} \alpha_k^t w_k^{t+1}
$$

---

### Điểm mạnh về mặt toán học & Hiệu năng

- **Đạo hàm độ nhạy bị chặn hoàn toàn**:

$$
\left| \frac{\partial \alpha_k^t}{\partial \tilde{L}_k^t} \right| \le \frac{1}{|S^t| - 1} < \infty
$$

  *(Không bị phân kỳ vô hạn như phép chia của FedAsl khi loss tiến về 0)*.
- **Tự động triệt tiêu outlier**: Client nào có loss quá lớn/nhiễu ($\tilde{L}_k^t \to 1$) thì $1 - \tilde{L}_k^t \to 0 \implies \alpha_k^t \to 0$ mà không cần đặt ngưỡng heuristic.
- **Siêu nhẹ**: Phía server chỉ tốn phép tính L1 trên các số float scalar loss, giảm tới 40% FLOPs so với FedMa và 19% so với FedProx.
