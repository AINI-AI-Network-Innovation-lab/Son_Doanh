"""pytorchexample: A Flower / PyTorch app with SCAFFOLD & FedEntropy."""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from datasets import load_dataset
from flwr_datasets import FederatedDataset
from flwr_datasets.partitioner import DirichletPartitioner
from torch.utils.data import DataLoader
from torchvision.transforms import Compose, Normalize, ToTensor


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(32, 64, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(64 * 8 * 8, 256)
        self.fc2 = nn.Linear(256, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.bn1(self.conv1(x))))  # 32x32 -> 16x16
        x = self.pool(F.relu(self.bn2(self.conv2(x))))  # 16x16 -> 8x8
        x = x.view(-1, 64 * 8 * 8)
        x = F.relu(self.fc1(x))
        return self.fc2(x)


def get_zero_control_variate(model: nn.Module) -> dict[str, torch.Tensor]:
    """Khởi tạo biến kiểm soát (control variate) bằng 0 cho các tham số khả huấn luyện."""
    return {
        name: torch.zeros_like(param.data, device="cpu")
        for name, param in model.named_parameters()
    }


def compute_shannon_entropy(probs: list[float], eps: float = 1e-12) -> float:
    """Tính Shannon Entropy H(P) = -sum(p_c * log(p_c))."""
    return -sum(p * math.log(max(p, eps)) for p in probs)


fds = None  # Cache FederatedDataset

pytorch_transforms = Compose([ToTensor(), Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])


def apply_transforms(batch):
    """Apply transforms to the partition from FederatedDataset."""
    batch["img"] = [pytorch_transforms(img) for img in batch["img"]]
    return batch


def load_data(partition_id: int, num_partitions: int, batch_size: int):
    """Load partition CIFAR10 data (Non-IID Dirichlet)."""
    global fds
    if fds is None:
        partitioner = DirichletPartitioner(
            num_partitions=num_partitions,
            partition_by="label",
            alpha=0.5,
            seed=42,
        )
        fds = FederatedDataset(
            dataset="uoft-cs/cifar10",
            partitioners={"train": partitioner},
        )
    partition = fds.load_partition(partition_id)
    partition_train_test = partition.train_test_split(test_size=0.2, seed=42)
    partition_train_test = partition_train_test.with_transform(apply_transforms)
    trainloader = DataLoader(
        partition_train_test["train"], batch_size=batch_size, shuffle=True
    )
    testloader = DataLoader(partition_train_test["test"], batch_size=batch_size)
    return trainloader, testloader


def load_centralized_dataset():
    """Load test set and return dataloader."""
    test_dataset = load_dataset("uoft-cs/cifar10", split="test")
    dataset = test_dataset.with_transform(apply_transforms)
    return DataLoader(dataset, batch_size=128)


def train(
    net: nn.Module,
    trainloader: DataLoader,
    epochs: int,
    lr: float,
    device: torch.device,
    server_c: dict[str, torch.Tensor],
    client_c: dict[str, torch.Tensor],
    num_classes: int = 10,
):
    """Huấn luyện mô hình cục bộ với SCAFFOLD và thu thập soft labels cho FedEntropy."""
    net.to(device)

    # Đưa hiệu số (c - c_i) lên thiết bị tính toán (GPU/CPU)
    c_diff = {
        name: (server_c[name] - client_c[name]).to(device)
        for name, _ in net.named_parameters()
    }

    # Bộ tích lũy đạo hàm thuần để cập nhật c_i^+ chính xác (tránh sai số do momentum/weight_decay)
    grad_accum = {
        name: torch.zeros_like(param.data, device=device)
        for name, param in net.named_parameters()
    }

    criterion = nn.CrossEntropyLoss().to(device)
    # Sử dụng SGD chuẩn không momentum theo đúng lý thuyết SCAFFOLD
    optimizer = torch.optim.SGD(net.parameters(), lr=lr, momentum=0.0)

    net.train()
    running_loss = 0.0
    num_steps = 0

    soft_label_sum = torch.zeros(num_classes, device=device)
    total_samples = 0

    for epoch in range(epochs):
        for batch in trainloader:
            images = batch["img"].to(device)
            labels = batch["label"].to(device)

            optimizer.zero_grad()
            outputs = net(images)
            loss = criterion(outputs, labels)
            loss.backward()

            with torch.no_grad():
                for name, param in net.named_parameters():
                    if param.grad is not None:
                        # 1. Lưu lại đạo hàm thực g_i(y_i) trước khi cộng biến kiểm soát
                        grad_accum[name].add_(param.grad)
                        # 2. SCAFFOLD: Hiệu chỉnh gradient g_i(y_i) <- g_i(y_i) + (c - c_i)
                        param.grad.add_(c_diff[name])

                # FedEntropy: Tích lũy soft labels ở epoch cuối cùng
                if epoch == epochs - 1:
                    probs = F.softmax(outputs.detach(), dim=-1)
                    soft_label_sum += probs.sum(dim=0)
                    total_samples += labels.size(0)

            # Chống bùng nổ đạo hàm (Gradient Clipping)
            torch.nn.utils.clip_grad_norm_(net.parameters(), max_norm=10.0)
            optimizer.step()

            running_loss += loss.item()
            num_steps += 1

    # Cập nhật c_i^+ bằng đúng trung bình đạo hàm cục bộ: c_i^+ = (1 / K) * sum(g_i)
    new_client_c = {}
    delta_c = {}

    with torch.no_grad():
        for name, _ in net.named_parameters():
            c_plus = (grad_accum[name] / max(num_steps, 1)).cpu()
            new_client_c[name] = c_plus
            delta_c[name] = c_plus - client_c[name]

    avg_soft_labels = (soft_label_sum / max(total_samples, 1)).cpu().tolist()
    avg_trainloss = running_loss / max(num_steps, 1)

    return avg_trainloss, new_client_c, delta_c, avg_soft_labels


def test(net, testloader, device):
    """Validate the model on the test set."""
    net.to(device)
    net.eval()
    criterion = torch.nn.CrossEntropyLoss()
    correct, loss = 0, 0.0
    with torch.no_grad():
        for batch in testloader:
            images = batch["img"].to(device)
            labels = batch["label"].to(device)
            outputs = net(images)
            loss += criterion(outputs, labels).item()
            correct += (torch.max(outputs.data, 1)[1] == labels).sum().item()
    accuracy = correct / len(testloader.dataset)
    loss = loss / len(testloader)
    return loss, accuracy