import torch
import torch.nn as nn
import torch.nn.functional as F
from datasets import load_dataset
from flwr_datasets import FederatedDataset
from flwr_datasets.partitioner import DirichletPartitioner
from torch.utils.data import DataLoader
from torchvision.transforms import Compose, Normalize, ToTensor


# Định nghĩa kiến trúc mạng FedAdp (chỉ CNN) 
# Định nghĩa kiến trúc mạng CNN chuẩn cho FedAdp (LeNet-5 cải tiến)
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        # 1. Trích xuất đặc trưng không gian (CNN)
        self.conv1 = nn.Conv2d(3, 6, 5)       # 3 kênh RGB -> 6 kênh
        self.pool = nn.MaxPool2d(2, 2)         # Giảm kích thước ảnh 2x2
        self.conv2 = nn.Conv2d(6, 16, 5)      # 6 kênh -> 16 kênh

        # 2. Các tầng kết nối đầy đủ (FC Layers)
        # Sau 2 lần pool: kích thước ảnh là 16 * 5 * 5 = 400
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)           # Đầu ra 10 nhãn phân loại CIFAR-10

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Qua 2 block Conv + ReLU + Pool
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = torch.flatten(x, 1)  # Duỗi phẳng thành vector 400 chiều

        # Qua các tầng Fully-Connected
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        logits = self.fc3(x)     # Trả về 1 tensor duy nhất (10 logits)
        return logits



# ==============================================================================
# XỬ LÝ VÀ PHÂN CHIA DỮ LIỆU
# ==============================================================================
fds = None  # Biến toàn cục dùng để cache FederatedDataset, tránh tải lại nhiều lần

# Chuỗi tiền xử lý ảnh: Chuyển đổi thành Tensor và chuẩn hóa giá trị pixel về khoảng [-1, 1]
pytorch_transforms = Compose([ToTensor(), Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])


def apply_transforms(batch):
    """Áp dụng tiền xử lý (Transform) cho từng lô ảnh từ FederatedDataset."""
    batch["img"] = [pytorch_transforms(img) for img in batch["img"]]
    return batch


def load_data(partition_id: int, num_partitions: int, batch_size: int):
    """Tải và phân chia tập dữ liệu CIFAR-10 cho từng Client cụ thể theo phân phối Dirichlet (Non-IID)."""
    global fds
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

def train(
    model: nn.Module,
    trainloader: DataLoader,
    s_k: int,
    lr: float,
    device: torch.device,
    epochs: int,
) -> float:
    """Huấn luyện mô hình cục bộ tại Client với hàm mất mát CrossEntropyLoss tiêu chuẩn."""
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9)
    model.train()

    total_loss = 0.0
    step = 0

    for epoch in range(epochs):
        for batch in trainloader:
            if step >= s_k:
                break
            
            images = batch["img"].to(device)
            labels = batch["label"].to(device)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            step += 1

        if step >= s_k:
            break

    return total_loss / max(step, 1)


def test(
    model: nn.Module, testloader: DataLoader, device: torch.device
) -> tuple[float, float]:
    """Kiểm thử mô hình và tính toán loss và accuracy."""
    criterion = nn.CrossEntropyLoss()
    correct = 0
    total_loss = 0.0
    model.eval()

    with torch.no_grad():
        for batch in testloader:
            images = batch["img"].to(device)
            labels = batch["label"].to(device)

            outputs = model(images)
            total_loss += criterion(outputs, labels).item()
            _, predicted = torch.max(outputs.data, 1)
            correct += (predicted == labels).sum().item()

    loss = total_loss / len(testloader)
    accuracy = correct / len(testloader.dataset)
    return loss, accuracy
