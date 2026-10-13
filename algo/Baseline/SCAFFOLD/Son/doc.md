# Tài Liệu Kỹ Thuật: Thuật Toán Kết Hợp SCAFFOLD và FedNova

## 1. Tổng Quan Kiến Trúc
Hệ thống kết hợp hai cơ chế tối ưu hóa nhằm giải quyết đồng thời hai thách thức lớn nhất trong Học liên hiệp (Federated Learning):
- **Tính không đồng nhất dữ liệu (Data Heterogeneity / Non-IID):** Giải quyết bằng cơ chế Control Variates của SCAFFOLD tại Client.
- **Tính không đồng nhất hệ thống (System Heterogeneity / Unequal Local Steps):** Giải quyết bằng cơ chế Chuẩn hóa bước nhảy của FedNova tại Server.

---

## 2. Phía Client: Khử Lệch Mô Hình Với SCAFFOLD

### 2.1. Vấn đề Client Drift
Khi phân phối dữ liệu giữa các client bị lệch (Non-IID), gradient cục bộ $\nabla F_k(w)$ kéo mô hình về cực tiểu riêng của client đó thay vì cực tiểu toàn cục $f(w)$, dẫn đến hiện tượng Client Drift.

### 2.2. Cơ chế hiệu chỉnh Gradient
SCAFFOLD duy trì hai biến kiểm soát (Control Variates) có cùng số chiều với bộ trọng số mô hình:
- $c_k$: Ước lượng hướng gradient cục bộ của riêng client $k$.
- $c$: Ước lượng hướng gradient toàn cục do Server phân phối.

Tại mỗi bước tối ưu hóa SGD của mini-batch, gradient được hiệu chỉnh:
$$g_{corrected} = g_k(w) - c_k + c$$

Trọng số được cập nhật theo:
$$w \leftarrow w - \lambda \cdot g_{corrected}$$

Trong đó $\lambda$ là tốc độ học cục bộ (learning rate).

### 2.3. Cập nhật và lưu trữ Control Variate
Sau khi hoàn thành huấn luyện cục bộ:
1. Client tính biến kiểm soát mới $c_k^+$:
   $$c_k^+ = c_k - c + \frac{w_{init} - w_{final}}{a_k}$$
   với $a_k = s_k \cdot \lambda$ là tổng lượng learning rate qua $s_k$ bước.
2. Lưu $c_k^+$ vào bộ nhớ cục bộ `context.state` để tái sử dụng trong vòng huấn luyện tiếp theo của chính client đó.
3. Tính độ lệch biến kiểm soát:
   $$\Delta c_k = c_k^+ - c_k$$
   và gửi $\Delta c_k$ về Server để Server cập nhật biến toàn cục mà không cần lưu trữ trạng thái của từng client.

---

## 3. Phía Server: Chuẩn Hóa Khối Lượng Tính Toán Với FedNova

### 3.1. Vấn đề Objective Inconsistency
Khi các client thực hiện số bước huấn luyện cục bộ khác nhau ($s_k \neq s_j$), thuật toán FedAvg truyền thống sẽ làm mô hình bị thiên lệch về phía client chạy nhiều bước hơn, tối ưu hóa sai hàm mục tiêu ban đầu.

### 3.2. Chuẩn hóa độ dời tại Client
Thay vì gửi trực tiếp độ dời $\Delta_k = w_{final} - w_{init}$, client gửi độ dời đã chuẩn hóa trên mỗi bước hiệu dụng:
$$d_k = \frac{\Delta_k}{a_k}$$
với $a_k = s_k \cdot \lambda$.

### 3.3. Tổng hợp tại Server
Server thực hiện tổng hợp theo hai bước:
1. Tính hướng di chuyển trung bình toàn cục $d_{global}$:
   $$d_{global} = \sum_{k \in \mathcal{S}} p_k d_k$$
   trong đó $p_k = \frac{n_k}{\sum_{j \in \mathcal{S}} n_j}$ là trọng số dựa trên số mẫu của client.

2. Tính bước nhảy hiệu dụng $\tau_{eff}$:
   $$\tau_{eff} = \sum_{k \in \mathcal{S}} p_k a_k$$

3. Cập nhật mô hình toàn cục:
   $$w_{next} = w_{cur} + \tau_{eff} \cdot d_{global}$$

4. Cập nhật Control Variate toàn cục:
   $$c \leftarrow c + \frac{1}{N} \sum_{k \in \mathcal{S}} \Delta c_k$$
   với $N$ là tổng số client trong toàn bộ mạng liên hiệp.

---

## 4. Mô Phỏng Thực Nghiệm Tính Không Đồng Nhất

### 4.1. Phân phối dữ liệu (Data Heterogeneity)
Dữ liệu CIFAR-10 được phân chia không đồng đều giữa các client thông qua phân phối Dirichlet:
- Tham số tập trung $\alpha = 0.5$.
- Làm lệch cả tỷ lệ nhãn giữa các lớp lẫn tổng số lượng mẫu $n_k$ của từng client.

### 4.2. Khác biệt số bước chạy (System Heterogeneity)
- **Khái niệm Epoch và Step:**
  - `Epoch`: Số lượt duyệt qua toàn bộ tập dữ liệu cục bộ của client.
  - `Step` (Batch Iteration): Một lần nạp batch và cập nhật trọng số.
  - Mối quan hệ: $\text{Số steps tối đa} = \text{local\_epochs} \times \lceil n_k / \text{batch\_size} \rceil$.
- **Lý do định nghĩa theo Step thay vì Epoch:**
  - Trong thực tế, các thiết bị biên (Edge Devices) có năng lực phần cứng khác nhau hoặc bị giới hạn bởi thời gian timeout của round giao tiếp. Thiết bị yếu sẽ bị ngắt giữa chừng và chỉ kịp xử lý một số lượng batch nhất định chứ không thể hoàn thành trọn vẹn epoch.
  - Số bước $s_k$ được sinh ngẫu nhiên trong khoảng $[s_{min}, s_{max}]$ với $s_{max}$ là số batch tối đa của client đó.
  - FedNova chuẩn hóa trực tiếp trên số bước thực tế $s_k$ đã chạy thông qua $a_k = s_k \cdot \lambda$, đảm bảo tính công bằng toán học bất kể client chạy trọn vẹn hay đứt quãng.

---

## 5. Quy Trình Truyền Nhận Dữ Liệu Qua Flower

### 5.1. Cấu trúc gói tin từ Server xuống Client (`configure_train`)
- `msg.content["arrays"]`: Trọng số mô hình toàn cục $w_{cur}$ (`ArrayRecord`).
- `msg.content["control_variate"]`: Biến kiểm soát toàn cục $c$ (`ArrayRecord`).
- `msg.content["config"]["lr"]`: Tốc độ học $\lambda$ (`ConfigRecord`).

### 5.2. Cấu trúc gói tin phản hồi từ Client lên Server (`train`)
- `msg.content["arrays"]`: Vector độ dời chuẩn hóa $d_k$ (`ArrayRecord`).
- `msg.content["control_variate"]`: Độ lệch biến kiểm soát $\Delta c_k$ (`ArrayRecord`).
- `msg.content["metrics"]["num-examples"]`: Số lượng mẫu $n_k$ (`MetricRecord`).
- `msg.content["metrics"]["a_k"]`: Lượng learning rate hiệu dụng $a_k$ (`MetricRecord`).
- `msg.content["metrics"]["train_loss"]`: Giá trị hàm mất mát huấn luyện (`MetricRecord`).
