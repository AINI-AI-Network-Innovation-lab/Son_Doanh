---
tags: [federated-learning, label-skew, non-iid, flower, icml2023]
dataset: [cifar10]
framework: [pytorch, flwr]
---

# FedDisco: Federated Learning with Discrepancy-Aware Collaboration

Triển khai thuật toán **FedDisco** kết hợp **MOON** trên nền tảng **Flower** (PyTorch) nhằm giải quyết thách thức không đồng nhất phân phối dữ liệu nhãn (Label Distribution Heterogeneity / Non-IID).

---

## 📄 Nguồn bài báo & Tài liệu tham khảo (Paper Reference)

- **Tên bài báo**: *FedDisco: Federated Learning with Discrepancy-Aware Collaboration*
- **Tác giả**: Rui Ye, Mingkai Xu, Jianyu Wang, Chenxin Xu, Siheng Chen, Yanfeng Wang
- **Hội nghị**: International Conference on Machine Learning (**ICML 2023**), PMLR 202:39879-39902
- **arXiv Link**: [https://arxiv.org/abs/2305.19229](https://arxiv.org/abs/2305.19229)
- **PDF**: [https://arxiv.org/pdf/2305.19229](https://arxiv.org/pdf/2305.19229) / [PMLR](https://proceedings.mlr.press/v202/ye23f/ye23f.pdf)
---

## 💡 Tổng quan thuật toán (Algorithm Overview)

- FedDisco:
    - Vấn đề đặt ra: FedAVG,FedProx hay moon: Ở server đều gán trọng số dựa vào số lượng mẫu dữ liệu cục bộ: $w^{t+1}=\sum{p_k.F_k\{w\}}=\sum{n_k.F_k\{w\}}$. Do đó khi dữ liệu phân bố không đồng đều trên các nhãn vẫn gây ra hiện tượng client drift
    - FedDisco đề xuất tính độ  lệch phân phối dữ liệu trên mỗi client  $d_k$ và dựa vào $d_k$ cùng với số mẫu dữ liệu client để cập nhật trọng số global tại server
    - Công thức định lượng độ lệch phân phối:
        
        $$d_k = \|q_k - u\|_2 = \sqrt{\sum_{c=1}^{C} \left(q_{k,c} - \frac{1}{C}\right)^2}$$
        
        $q_{k,c}$ là tỷ lệ mẫu cục bộ của các lớp
        
        $\frac{1}{C}$ là phân phối mục tiêu toàn cục. Chỉ đúng khi dữ liệu ban đầu phân phối đều. Với các bài toán mà tập dữ liệu ban đầu không đều (như phân loại bệnh) thì phải tính lại tỉ lệ dữ liệu chuẩn.
        
        $d_k$ nhằm đảm bảo bảo mật dữ liệu, $d_k$ nên được tính ở client rồi mới gửi lên server
        
    - Hàm mục tiêu toàn cục: 
        
        $$\min_{t} \mathbb{E} \left[ \|\nabla F(w^{(t,0)})\|^2 \right]$$
        
    - $p_k \propto n_k - a \cdot d_k + b$
        - Trọng số tỷ lệ với tỷ lệ mẫu $p_k$, độ lệch phân phối $d_k$ data và phần bù $b$
        - Với client nhiều dữ liệu hơn thì được ưu tiên hơn
        - Với độ lệch lớn → trừ đi, giảm độ ưu tiên
    - Tính trọng số mới (Tại server): client gửi $d_k$, $w_{local}$ và $n_k$
        
        $$p_k = \frac{\text{ReLU}(n_k - a \cdot d_k + b)}{\sum_{m \in S_i} \text{ReLU}(n_m - a \cdot d_m + b)}$$
        
        Với $n_k$ là tỷ lệ kích thước dữ liệu của client k
        
        $d_k$ độ lệch phân phối của client k
        
        $a>0$ hyperparameter, điều chỉnh mức độ phạt đối với độ lệch
        
        $\mathcal{b}$ offset nhằm điều chỉnh nền trọng số
        
    - Kết quả implement: Với 2 node, accuracy đạt 72%

---

## 📁 Cấu trúc thư mục

```text
FedDisco/
├── __init__.py           # Package init
├── custom_strategy.py    # Chiến lược Server FedDisco tổng hợp trọng số Discrepancy-aware
├── client_app.py         # Huấn luyện cục bộ tại Client (tích hợp MOON loss & tính discrepancy)
├── server_app.py         # Khởi tạo ServerApp và cấu hình FedDisco Strategy
├── task.py               # Kiến trúc mạng CNN (Backbone + Projection Head), data loader, Dirichlet partitioner
├── pyproject.toml        # Cấu hình siêu tham số và phụ thuộc dự án Flower
├── LICENSE               # Giấy phép mã nguồn
└── README.md             # Tài liệu hướng dẫn và nguồn bài báo
```

---

## 🚀 Hướng dẫn cài đặt & Chạy thực nghiệm

### 1. Cài đặt môi trường & dependencies
Cài đặt Flower cùng các gói phụ thuộc cần thiết:
```bash
cd /home/administrator/Son_Doanh/algo/A2_Label_Data_distribution/FedDisco
pip install -e .
```

### 2. Cấu hình siêu tham số (`pyproject.toml`)
Các tham số có thể điều chỉnh trong phần `[tool.flwr.app.config]`:
- `num-server-rounds`: Số vòng liên kết (mặc định: `15`).
- `fraction-evaluate`: Tỷ lệ client tham gia đánh giá (mặc định: `0.5`).
- `learning-rate`: Tốc độ học (mặc định: `0.01`).
- `batch-size`: Kích thước batch (mặc định: `32`).
- `local-epochs`: Số epoch huấn luyện cục bộ (mặc định: `2`).
- `disco-a`: Hệ số phạt độ lệch phân phối $a$ trong FedDisco (mặc định: `0.5`).
- `disco-b`: Hệ số bù $b$ trong FedDisco (mặc định: `0.1`).
- `moon-mu`: Trọng số loss tương phản MOON (mặc định: `1.0`).
- `temperature`: Nhiệt độ $\tau$ trong hàm contrastive loss của MOON (mặc định: `0.5`).

### 3. Chạy mô phỏng (Simulation)
Khởi chạy thử nghiệm học liên kết với Flower Simulation Engine:
```bash
flwr run .
```
