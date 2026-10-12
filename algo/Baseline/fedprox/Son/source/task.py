"""Quy trình học máy thuần PyTorch cho bài toán học liên hiệp (Federated Learning).

File này định nghĩa:
1. Kiến trúc mô hình mạng nơ-ron (Net - CNN phân loại ảnh CIFAR-10).
2. Tải và phân chia tập dữ liệu CIFAR-10 cho các Client (load_data, load_centralized_dataset).
3. Vòng lặp huấn luyện cục bộ tích hợp hình phạt FedProx Proximal Term (train).
4. Vòng lặp đánh giá mô hình, tính Loss và Accuracy (test).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from datasets import load_dataset
from flwr_datasets import FederatedDataset
from flwr_datasets.partitioner import DirichletPartitioner
from torch.utils.data import DataLoader
from torchvision.transforms import Compose, Normalize, ToTensor


# ==========================================
# 1. ĐỊNH NGHĨA KIẾN TRÚC MẠNG NƠ-RON (MODEL)
# ==========================================

class Net(nn.Module):
    """Mô hình CNN đơn giản gồm 2 tầng tích chập và 3 tầng kết nối đầy đủ (dành cho CIFAR-10)."""

    def __init__(self):
        super(Net, self).__init__()
        # Tầng tích chập 1: 3 kênh đầu vào (ảnh màu RGB) -> 6 kênh đầu ra, kernel size 5x5
        self.conv1 = nn.Conv2d(3, 6, 5)
        # Tầng Max Pooling: kích thước 2x2, bước nhảy 2 (giảm kích thước ảnh đi 1/2)
        self.pool = nn.MaxPool2d(2, 2)
        # Tầng tích chập 2: 6 kênh vào -> 16 kênh ra, kernel size 5x5
        self.conv2 = nn.Conv2d(6, 16, 5)
        # Tầng kết nối đầy đủ (Fully Connected):
        # Kích thước sau 2 lần pool: 16 kênh * 5 * 5 -> 120 nodes
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        # Tầng đầu ra: 84 -> 10 classes (10 nhãn phân loại của CIFAR-10)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Lan truyền xuôi (Forward pass) qua các tầng mạng."""
        x = self.pool(F.relu(self.conv1(x)))  # Conv1 -> ReLU -> MaxPool
        x = self.pool(F.relu(self.conv2(x)))  # Conv2 -> ReLU -> MaxPool
        x = x.view(-1, 16 * 5 * 5)            # Duỗi thẳng (Flatten) tensor thành vector 1D
        x = F.relu(self.fc1(x))               # FC1 -> ReLU
        x = F.relu(self.fc2(x))               # FC2 -> ReLU
        return self.fc3(x)                    # FC3 -> Logits đầu ra (chưa qua softmax vì CrossEntropyLoss đã tích hợp)


# ==========================================
# 2. XỬ LÝ VÀ PHÂN CHIA TẬP DỮ LIỆU
# ==========================================

fds = None  # Biến toàn cục dùng để cache FederatedDataset, tránh tải lại nhiều lần

# Chuỗi tiền xử lý ảnh: Chuyển đổi thành Tensor và chuẩn hóa giá trị pixel về khoảng [-1, 1]
pytorch_transforms = Compose([ToTensor(), Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])


def apply_transforms(batch):
    """Áp dụng tiền xử lý (Transform) cho từng lô ảnh từ FederatedDataset."""
    batch["img"] = [pytorch_transforms(img) for img in batch["img"]]
    return batch


def load_data(partition_id: int, num_partitions: int, batch_size: int):
    """Tải và phân chia tập dữ liệu CIFAR-10 cho từng Client cụ thể.

    Args:
        partition_id: Số thứ tự (ID) của Client (0, 1, ..., num_partitions - 1).
        num_partitions: Tổng số phân vùng (tương ứng tổng số client mô phỏng).
        batch_size: Kích thước mỗi batch khi nạp vào DataLoader.

    Returns:
        trainloader, testloader: Hai DataLoader dành cho huấn luyện và kiểm thử tại Client đó.
    """
    global fds
    # Chỉ khởi tạo FederatedDataset một lần duy nhất để tối ưu hiệu năng
    if fds is None:
        partitioner = DirichletPartitioner(
            num_partitions=num_partitions,
            partition_by="label",
            alpha=0.5,
            min_partition_size=10,
            self_balancing=False,
            shuffle=True,
            seed=42,
        )
        fds = FederatedDataset(
            dataset="uoft-cs/cifar10",
            partitioners={"train": partitioner},
        )
        
    # Nạp đúng phân vùng dữ liệu thuộc về Client này
    partition = fds.load_partition(partition_id)
    
    # Chia phân vùng của Client: 80% để train, 20% để kiểm thử (validate)
    partition_train_test = partition.train_test_split(test_size=0.2, seed=42)
    partition_train_test = partition_train_test.with_transform(apply_transforms)
    
    # Đóng gói thành PyTorch DataLoader
    trainloader = DataLoader(
        partition_train_test["train"], batch_size=batch_size, shuffle=True
    )
    testloader = DataLoader(partition_train_test["test"], batch_size=batch_size)
    return trainloader, testloader


def load_centralized_dataset():
    """Tải toàn bộ tập kiểm tra (test set) chuẩn của CIFAR-10 để Server đánh giá mô hình toàn cục."""
    test_dataset = load_dataset("uoft-cs/cifar10", split="test")
    dataset = test_dataset.with_format("torch").with_transform(apply_transforms)
    return DataLoader(dataset, batch_size=128)


# ==========================================
# 3. HUẤN LUYỆN CỤC BỘ (VỚI HÌNH PHẠT FEDPROX)
# ==========================================

def train(net: nn.Module, trainloader: DataLoader, epochs: int, lr: float, device: torch.device, mu: float = 0.0) -> float:
    """Huấn luyện mô hình cục bộ tại Client, tích hợp hình phạt Proximal Term của thuật toán FedProx.

    Công thức hàm Loss của FedProx:
        L_total = L_cross_entropy + (mu / 2) * sum(||w - w^t||^2)
        Trong đó:
        - w: Trọng số của mô hình cục bộ đang được cập nhật qua từng batch.
        - w^t: Trọng số toàn cục ban đầu nhận từ Server ở đầu round.
        - mu: Hệ số phạt proximal term (mu = 0 tương đương thuật toán FedAvg truyền thống).

    Args:
        net: Mô hình PyTorch cần huấn luyện.
        trainloader: Dữ liệu huấn luyện cục bộ.
        epochs: Số vòng lặp qua toàn bộ dữ liệu cục bộ (local epochs).
        lr: Tốc độ học (learning rate).
        device: Thiết bị chạy (GPU hoặc CPU).
        mu: Hệ số phạt FedProx (mặc định 0.0).

    Returns:
        float: Giá trị Loss huấn luyện trung bình.
    """
    net.to(device)
    
    # 1. Lưu lại bản sao trọng số ban đầu nhận từ server (w^t) làm mốc so sánh cho Proximal Term
    # Cần dùng .detach().clone() để tạo bản sao độc lập và không tính gradient lên chính nó
    global_params = [param.detach().clone() for param in net.parameters()]

    criterion = torch.nn.CrossEntropyLoss().to(device)
    optimizer = torch.optim.SGD(net.parameters(), lr=lr, momentum=0.9)
    net.train()
    running_loss = 0.0

    # 2. Vòng lặp huấn luyện qua từng Epoch
    for _ in range(epochs):
        for batch in trainloader:
            images = batch["img"].to(device)
            labels = batch["label"].to(device)
            optimizer.zero_grad()  # Xóa gradient của batch trước

            # a. Tính hàm Loss cơ bản (Cross-Entropy Loss)
            outputs = net(images)
            loss = criterion(outputs, labels)

            # b. Tính FedProx Proximal Term: (mu / 2) * sum(||w - w^t||^2)
            # Hình phạt này ngăn mô hình Client lệch quá xa khỏi mô hình toàn cục của Server
            if mu > 0.0:
                proximal_term = 0.0
                for param, global_param in zip(net.parameters(), global_params):
                    proximal_term += torch.sum((param - global_param) ** 2)
                loss += (mu / 2.0) * proximal_term

            # c. Lan truyền ngược (Backpropagation) và cập nhật trọng số
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

    # Tính loss trung bình trên toàn bộ các batch và epoch
    avg_trainloss = running_loss / (epochs * len(trainloader))
    return avg_trainloss


# ==========================================
# 4. ĐÁNH GIÁ MÔ HÌNH (TEST / EVALUATE)
# ==========================================

def test(net: nn.Module, testloader: DataLoader, device: torch.device):
    """Đánh giá chất lượng mô hình trên tập kiểm thử (tính Loss và Accuracy).

    Args:
        net: Mô hình cần đánh giá.
        testloader: DataLoader chứa dữ liệu kiểm thử.
        device: Thiết bị chạy (GPU hoặc CPU).

    Returns:
        tuple[float, float]: (test_loss, accuracy).
    """
    net.to(device)
    criterion = torch.nn.CrossEntropyLoss()
    correct, total_loss = 0, 0.0
    
    net.eval()  # Chuyển sang chế độ evaluation (tắt dropout, cố định batchnorm nếu có)
    with torch.no_grad():  # Không tính toán gradient trong quá trình đánh giá để tiết kiệm bộ nhớ
        for batch in testloader:
            images = batch["img"].to(device)
            labels = batch["label"].to(device)
            outputs = net(images)
            
            # Cộng dồn loss
            total_loss += criterion(outputs, labels).item()
            
            # Lấy chỉ số có xác suất cao nhất làm nhãn dự đoán và so sánh với nhãn thực tế
            predicted = torch.max(outputs.data, 1)[1]
            correct += (predicted == labels).sum().item()

    # Tính độ chính xác trung bình và loss trung bình
    accuracy = correct / len(testloader.dataset)
    avg_loss = total_loss / len(testloader)
    return avg_loss, accuracy
