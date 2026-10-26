Bộ trọng số tại mỗi thiết bị được cập nhật theo bộ dữ liệu tại đó và gửi về server thực hiện **FedAvg**.

Nhược điểm của FedAvg: Cào bằng tất cả, khi truyền ngược về làm mất đi những đặc trưng cá nhân hóa của mỗi thiết bị tại biên $\rightarrow$ dẫn đến hướng nghiên cứu Personalized FL (pFL).

---

### Bài toán tối ưu toàn cục

$$
\min_{w} \; F(w) = \sum_{k=1}^{N} p_k F_k(w)
$$

với:

$$
F_k(w) = \frac{1}{n_k} \sum_{x_i \in \mathcal{D}_k} f_k(w; x_i), \qquad p_k = \frac{n_k}{n}
$$

Trong đó:
- $f_k(w; x_i)$: Là **loss trên một mẫu** dữ liệu $x_i$.
- $F_k(w)$: Là **loss cục bộ** (trung bình của $f_k$ trên toàn bộ tập dữ liệu $\mathcal{D}_k$ của client $k$).
- $F(w)$: Là **loss toàn cục** (trung bình có trọng số của các $F_k$, đây là mục tiêu cần cực tiểu hóa).
- $n_k$: Số lượng mẫu dữ liệu tại client $k$.
- $n = \sum_{k=1}^N n_k$: Tổng số mẫu của tất cả các client trong mạng.
- $N$: Tổng số lượng client tham gia.

---

### Cơ chế cập nhật và Nhược điểm của FedAvg

Client $k$ chạy $s_k$ bước SGD cục bộ trong vòng giao tiếp thứ $t$. 

Đặt độ dời tích lũy của client $k$ là:

$$
\Delta_k = w_k^{t, s_k} - w^t
$$

Trong đó $w_k^{t, s_k}$ là bộ trọng số mà client $k$ gửi lên server ở cuối vòng $t$ (coi $w_k^{t, s_k} \equiv w_k^{t+1}$). Khi đó:

$$
w_k^{t+1} = w^t + \Delta_k
$$

Tại Server, thuật toán **FedAvg** thực hiện tổng hợp:

$$
w^{t+1} = \sum_{k=1}^N p_k w_k^{t+1} = \sum_{k=1}^N p_k \big(w^t + \Delta_k\big) = w^t \underbrace{\sum_{k=1}^N p_k}_{=1} + \sum_{k=1}^N p_k \Delta_k = w^t + \sum_{k=1}^N p_k \Delta_k
$$

---

### Vấn đề thiên vị kép (Objective Inconsistency) trong FedAvg

- Càng nhiều mẫu dữ liệu $\rightarrow$ client càng chạy nhiều bước $s_k$ (do batch size cố định trên toàn hệ thống).
- Do đó, vector độ dời $\Delta_k$ càng dài:

$$
\Delta_k = s_k \cdot d_k
$$

*(với $d_k$ là độ dời trung bình trên mỗi đơn vị bước).*

Khi đó:

$$
w^{t+1} = w^t + \sum_{k=1}^N p_k \cdot (s_k \cdot d_k)
$$

$\Rightarrow$ **FedAvg đang bị thiên vị 2 lần** cho các client có nhiều dữ liệu hơn:
1. Lần 1 qua tỷ trọng mẫu dữ liệu $p_k = \frac{n_k}{n}$.
2. Lần 2 qua số bước huấn luyện tích lũy $s_k \propto n_k$.

$\rightarrow$ **FedNova** ra đời để xử lý triệt để vấn đề này bằng cách chuẩn hóa độ dời hiệu dụng:

$$
d_k = \frac{\Delta_k}{\sum_{i=1}^{s_k} \lambda_i}
$$
