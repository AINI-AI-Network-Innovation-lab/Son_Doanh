---
tags: [federated-learning, label-skew, non-iid, flower, entropy, scaffold]
dataset: [cifar10]
framework: [pytorch, flwr]
---

# FedEntropy: Mitigating Data Heterogeneity in Federated Learning via Entropy-Weighted Aggregation

Triển khai thuật toán **FedEntropy** kết hợp biến kiểm soát **SCAFFOLD** trên nền tảng **Flower** (PyTorch) nhằm giải quyết thách thức không đồng nhất phân phối dữ liệu nhãn (Data Heterogeneity / Non-IID).

---

## 📄 Nguồn bài báo & Tài liệu tham khảo (Paper Reference)

- **Tên bài báo**: *Mitigating data heterogeneity in federated learning via entropy-weighted aggregation*
- **Tác giả**: Asier Zulueta-Doalto, Ekaitz Zulueta, Mikel Zulueta-Perez, Jose Manuel Lopez-Guede
- **Tạp chí**: *Neurocomputing*, Volume 700, Article 134398
- **DOI**: [10.1016/j.neucom.2026.134398](https://doi.org/10.1016/j.neucom.2026.134398)
- **ScienceDirect Link**: [ScienceDirect Article](https://www.sciencedirect.com/science/article/pii/S0925231226134398)
- **Tài liệu nền tảng liên quan (MEJ Algorithm)**:
  - *FedEntropy: Efficient Device Grouping for Federated Learning Using Maximum Entropy Judgment*
  - **arXiv**: [https://arxiv.org/abs/2010.15555](https://arxiv.org/abs/2010.15555)

---

## 💡 Tổng quan thuật toán (Algorithm Overview)

- FedEntropy/Performance
    - FedEntropy
        - Entropy thể hiện độ hỗn loạn/phân tán của dữ liệu giữa các lớp
            - $$H_k=-\sum_{c=1}^Cp_{k,c}\log{p_{k,c}}$ với $p_{k,c}$$ là tỷ lệ mẫu thuộc lớp c tại client k
            - Entropy càng cao thì càng cân bằng
            - $$H_k^{norm} = \frac{H_k}{\log{C}}$$ với C là tổng số lớp của bài toán
            - $$H_k^{norm}$$ gần 0 thì phân phối mất cân bằng , gần 1 thì đồng đều giữa các lớp
        - Cập nhật trọng số server side:
            - $$\alpha_k^{\text{Entropy}} = \frac{H_k^{\text{norm}} n_k}{\sum_{j=1}^K H_j^{\text{norm}} n_j}$$: Cân bằng giữa độ phân tán dữ liệu và số mẫu của client
            - $$w_{t+1} = \sum_{k=1}^K \alpha_k^{\text{Entropy}} w_{k, t+1}$$
    - FedPerformance: Giải quyết vấn đề: entropy giống nhau có thể cho ra chất lượng sai lệch do chất lượng của mẫu( dữ liệu phân phối đều nhưng có ảnh nhiễu, có ảnh nét)
        - Sử dụng training loss $\mathcal{L}_k$ để đánh giá hiệu suất
        - $$P_k=1-\frac{\mathcal{L}_k}{\sum_{j=1}^K{\mathcal{L}_j}+\epsilon}$$
            - $$\mathcal{L}_k$$: training loss của client k sau local training
            - $$\epsilon$$: hằng số để tránh chia cho 0
            - $$P_k$$: Điểm hiệu suất tương đối của client
            - Lấy 1 - biểu thức để khi loss nhỏ thì điểm hiệu suất sẽ cao hơn
        - Công thức tổng hợp trọng số ở server: $$\alpha_k^{\text{Performance}} = \frac{P_k n_k}{\sum_j P_j n_j}$$
        - Vấn đề: Hiệu suất loss ở local chỉ đánh giá trên tập dữ liệu của client, do đó với bài toán non-iid không hữu dụng lắm: thực tế chỉ ra nó ngang ngửa với fedavg :))
    - FedEntropyPerformance
        - Hybrid mode:
        - $$\alpha_k^{E+P} = \frac{P_k H_k^{\text{norm}} n_k}{\sum_j P_j H_j^{\text{norm}} n_j}$$
        - Thực tế: Chỉ lệch 0.3% so với FedEntropy
    - Kết luận: Nên dùng FedEntropy vì hiệu quả đủ tốt, không mất thêm chi phí truyền thông(không phải gửi L, chỉ gửi entropy lên sever).

---

## 📁 Cấu trúc thư mục

```text
FedEntropy/
├── __init__.py           # Package init
├── custom_strategy.py    # Chiến lược ScaffoldFedEntropy (MEJ filtering & Entropy-weighted aggregation)
├── client_app.py         # Huấn luyện cục bộ Client (SCAFFOLD control variates & tính soft labels)
├── server_app.py         # Khởi tạo ServerApp và cấu hình ScaffoldFedEntropy Strategy
├── task.py               # Mạng CNN, Dirichlet partitioner, Shannon entropy & data loaders
├── pyproject.toml        # Cấu hình siêu tham số và phụ thuộc dự án Flower
├── LICENSE               # Giấy phép mã nguồn
└── README.md             # Tài liệu hướng dẫn và nguồn bài báo
```

---

## 🚀 Hướng dẫn cài đặt & Chạy thực nghiệm

### 1. Cài đặt môi trường & dependencies
Cài đặt Flower cùng các gói phụ thuộc cần thiết:
```bash
cd /home/administrator/Son_Doanh/algo/A2_Label_Data_distribution/FedEntropy
pip install -e .
```

### 2. Cấu hình siêu tham số (`pyproject.toml`)
Các tham số có thể điều chỉnh trong phần `[tool.flwr.app.config]`:
- `num-server-rounds`: Số vòng liên kết (mặc định: `15`).
- `fraction-evaluate`: Tỷ lệ client tham gia đánh giá (mặc định: `0.5`).
- `learning-rate`: Tốc độ học cục bộ $\eta_l$ (mặc định: `0.01`).
- `batch-size`: Kích thước batch (mặc định: `32`).
- `local-epochs`: Số epoch huấn luyện cục bộ (mặc định: `2`).
- `server-lr`: Tốc độ học phía server của SCAFFOLD (mặc định: `1.0`).
- `entropy-gamma`: Hệ số lũy thừa entropy $\gamma$ cho trọng số (mặc định: `1.0`).
- `use-mej-filter`: Bật/tắt cơ chế lọc Maximum Entropy Judgment (`true`/`false`).

### 3. Chạy mô phỏng (Simulation)
Khởi chạy thử nghiệm học liên kết với Flower Simulation Engine:
```bash
flwr run .
```
