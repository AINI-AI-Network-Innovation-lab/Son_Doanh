"""Định nghĩa kiến trúc mô hình, phương pháp nạp dữ liệu, vòng lặp huấn luyện & hàm loss l_con & l_sup."""

import torch
import torch.nn as nn
import torch.nn.functional as F
from datasets import load_dataset
from flwr_datasets import FederatedDataset
from flwr_datasets.partitioner import DirichletPartitioner
from torch.utils.data import DataLoader
from torchvision.transforms import Compose, Normalize, ToTensor


# Định nghĩa kiến trúc mạng MOON (gồm Base Feature Extractor, Base Encoder, Projection Head & Classifier)
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        # 1. Trích xuất đặc trưng không gian (CNN)
        # Tầng tích chập 1 với 3 kênh đầu vào (ảnh màu RGB) và 6 kênh đầu ra, kích thước kernel 5x5
        self.conv1 = nn.Conv2d(3, 6, 5)
        # Max pooling 2x2, bước nhảy 2 (dùng chung cho cả 2 tầng tích chập)
        self.pool = nn.MaxPool2d(2, 2)
        # Tầng tích chập 2: 6 kênh vào -> 16 kênh ra, kích thước kernel 5x5
        self.conv2 = nn.Conv2d(6, 16, 5)

        # 2. Base Encoder: Các tầng kết nối đầy đủ (FC)
        # Kích thước sau 2 lần pool: 16 * 5 * 5 = 400 -> 120 nodes
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        # 120 nodes -> 84 nodes (đặc trưng h)
        self.fc2 = nn.Linear(120, 84)

        # 3. Projection Head: MLP 2 tầng (Tuyến tính -> ReLU -> Tuyến tính) theo bài báo MOON
        # Nhận vào 84 chiều -> tầng ẩn 256 chiều -> vector đặc trưng tương phản z 256 chiều
        self.proj = nn.Sequential(
            nn.Linear(84, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
        )

        # 4. Classification Head: Lớp phân loại 10 nhãn của CIFAR-10
        self.classifier = nn.Linear(256, 10)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        # Trích xuất đặc trưng qua CNN
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = torch.flatten(x, 1)  # Duỗi phẳng mỗi ảnh thành vector 400 chiều

        # Đi qua Base Encoder
        x = F.relu(self.fc1(x))
        h = F.relu(self.fc2(x))  # h: vector đặc trưng 84 chiều

        # Đi qua Projection Head để lấy vector đặc trưng tương phản z
        z = self.proj(h)  # z: vector 256 chiều phục vụ Contrastive Loss

        # Đi qua Classifier để lấy logits phân loại
        logits = self.classifier(z)  # logits: 10 chiều phục vụ Supervised CrossEntropy Loss

        # Trả về cả logits (cho phân loại) và z (cho so sánh tương phản)
        return logits, z


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


# ==============================================================================
# HÀM MẤT MÁT TƯƠNG PHẢN (MODEL-CONTRASTIVE LOSS)
# ==============================================================================
def model_contrastive_loss(
    z: torch.Tensor,
    z_glob: torch.Tensor,
    z_prev: torch.Tensor | None,
    temperature: float = 0.5,
) -> torch.Tensor:
    """Tính hàm mất mát Model-Contrastive của thuật toán MOON."""
    cos_sim = nn.CosineSimilarity(dim=-1)

    # 1. Đo độ tương đồng giữa model hiện tại và model toàn cục (positive pair)
    sim_pos = cos_sim(z, z_glob).reshape(-1, 1) / temperature  # [B, 1]

    # Ở round 1 chưa có prev_model, trả về 0
    if z_prev is None:
        return torch.tensor(0.0, device=z.device)

    # 2. Đo độ tương đồng giữa model hiện tại và model vòng trước (negative pair)
    sim_neg = cos_sim(z, z_prev).reshape(-1, 1) / temperature  # [B, 1]

    # 3. Ghép thành logits: cột 0 là positive, cột 1 là negative -> shape: [B, 2]
    logits = torch.cat([sim_pos, sim_neg], dim=1)

    # 4. Nhãn mục tiêu: Luôn mong muốn mô hình chọn cột 0 (tiến gần model toàn cục)
    labels = torch.zeros(logits.size(0), dtype=torch.long, device=z.device)

    # 5. Dùng CrossEntropyLoss trên logits chính là công thức InfoNCE của MOON
    criterion = nn.CrossEntropyLoss()
    return criterion(logits, labels)


# ==============================================================================
# HUẤN LUYỆN CỤC BỘ (LOCAL TRAIN)
# ==============================================================================
def train(
    model: nn.Module,
    trainloader: DataLoader,
    global_model: nn.Module,
    prev_model: nn.Module | None,
    s_k: int,
    lr: float,
    device: torch.device,
    mu: float = 1.0,
    temperature: float = 0.5,
    epochs: int = 1,
) -> float:
    """Vòng lặp huấn luyện cục bộ của MOON kết hợp số bước s_k của FedNova."""
    model.train()
    global_model.eval()
    if prev_model is not None:
        prev_model.eval()

    optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=1e-5)
    criterion = nn.CrossEntropyLoss()

    running_loss = 0.0
    total_steps = 0

    # Lặp qua các epoch nhưng giới hạn bởi số bước s_k thực tế của client (FedNova)
    while total_steps < s_k:
        for batch in trainloader:
            if total_steps >= s_k:
                break

            images = batch["img"].to(device)
            labels = batch["label"].to(device)

            # 1. Forward model hiện tại (có tính gradient)
            logits, z = model(images)

            # 2. Forward global & prev model (chặn gradient để tiết kiệm bộ nhớ)
            with torch.no_grad():
                _, z_glob = global_model(images)
                z_prev = None
                if prev_model is not None:
                    _, z_prev = prev_model(images)

            # 3. Tính loss MOON: L_total = L_ce + mu * L_con
            loss_ce = criterion(logits, labels)
            loss_con = model_contrastive_loss(z, z_glob, z_prev, temperature=temperature)
            loss_total = loss_ce + mu * loss_con

            # 4. Cập nhật trọng số
            optimizer.zero_grad()
            loss_total.backward()
            optimizer.step()

            running_loss += loss_total.item()
            total_steps += 1

    return running_loss / max(1, total_steps)


# ==============================================================================
# KIỂM THỬ (EVALUATE / TEST)
# ==============================================================================
def test(net: nn.Module, testloader: DataLoader, device: torch.device) -> tuple[float, float]:
    """Đánh giá chất lượng mô hình trên tập kiểm thử (tính Loss và Accuracy).

    Args:
        net: Mô hình cần đánh giá.
        testloader: DataLoader chứa dữ liệu kiểm thử.
        device: Thiết bị chạy (GPU hoặc CPU).

    Returns:
        tuple[float, float]: (avg_loss, accuracy).
    """
    net.to(device)
    net.eval()  # Chuyển sang chế độ đánh giá
    criterion = nn.CrossEntropyLoss()
    correct, total_loss = 0, 0.0

    with torch.no_grad():  # Tắt autograd để tiết kiệm bộ nhớ khi đánh giá
        for batch in testloader:
            images = batch["img"].to(device)
            labels = batch["label"].to(device)

            # Forward qua mô hình, chỉ lấy logits (bỏ qua z vì không cần tính contrastive loss khi test)
            logits, _ = net(images)

            # Tính loss phân loại
            total_loss += criterion(logits, labels).item()

            # Lấy chỉ số có logit cao nhất làm nhãn dự đoán và so sánh với nhãn thực tế
            predicted = torch.max(logits.data, 1)[1]
            correct += (predicted == labels).sum().item()

    num_samples = len(testloader.dataset)
    num_batches = len(testloader)

    accuracy = float(correct / num_samples) if num_samples > 0 else 0.0
    avg_loss = float(total_loss / num_batches) if num_batches > 0 else 0.0

    return avg_loss, accuracy
