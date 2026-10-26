import datasets
import math
from typing import Dict, List, Tuple
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from datasets import load_dataset
from torch.utils.data import DataLoader, Subset
from torchvision.transforms import Compose, Normalize, ToTensor

datasets.disable_caching()


# ===========================================================================
# 1. KIẾN TRÚC MÔ HÌNH MẠNG (MODEL ARCHITECTURE - 6 LỚP CNN CHUẨN)
# ===========================================================================


class ConvBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, pool: bool = False):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.pool = nn.MaxPool2d(2, 2) if pool else None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        if self.pool is not None:
            x = self.pool(x)
        return x


class Net(nn.Module):
    def __init__(self, num_classes: int = 10):
        super().__init__()
        self.layer0 = ConvBlock(3, 32, pool=False)      # Layer 0: 32x32 -> 32x32
        self.layer1 = ConvBlock(32, 64, pool=True)      # Layer 1: 32x32 -> 16x16
        self.layer2 = ConvBlock(64, 128, pool=False)    # Layer 2: 16x16 -> 16x16
        self.layer3 = ConvBlock(128, 128, pool=True)    # Layer 3: 16x16 -> 8x8
        self.layer4 = ConvBlock(128, 256, pool=False)   # Layer 4: 8x8 -> 8x8
        self.layer5 = ConvBlock(256, 256, pool=True)    # Layer 5: 8x8 -> 4x4
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.classifier = nn.Linear(256, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.layer0(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.layer5(x)
        x = self.pool(x)
        x = torch.flatten(x, 1)
        return self.classifier(x)


# ===========================================================================
# 2. TIỀN XỬ LÝ VÀ PHÂN HOẠCH DỮ LIỆU (DATASET PARTITIONING)
# ===========================================================================

pytorch_transforms = Compose([ToTensor(), Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])


def apply_transforms(batch):
    batch["img"] = [pytorch_transforms(img) for img in batch["img"]]
    return batch


_cifar_cache: Dict = {}


def _prepare_cifar_splits(
    seed: int = 42,
    alpha: float = 0.5,
    num_partitions: int = 20,
    proxy_samples: int = 2000,
    server_val_samples: int = 3000,
) -> Dict:
    cache_key = (seed, alpha, num_partitions, proxy_samples, server_val_samples)
    if cache_key in _cifar_cache:
        return _cifar_cache[cache_key]

    raw_ds = load_dataset("uoft-cs/cifar10")
    train_full = raw_ds["train"].with_format("torch").with_transform(apply_transforms)
    test_full = raw_ds["test"].with_format("torch").with_transform(apply_transforms)

    # 1. Trích xuất tập proxy U (bỏ qua nhãn) đặt tại Server
    u_dataset = Subset(train_full, range(0, proxy_samples))

    # 2. Trích xuất tập Validation đặt tại Server
    val_dataset = Subset(train_full, range(proxy_samples, proxy_samples + server_val_samples))

    # 3. Tập dữ liệu còn lại (45.000 mẫu) phân chia cho các Client theo Dirichlet
    offset = proxy_samples + server_val_samples
    labels = np.array(raw_ds["train"]["label"][offset:])
    num_classes = 10
    np_rng = np.random.RandomState(seed)

    # Phân chia Dirichlet non-IID giữa các node
    client_indices: List[List[int]] = [[] for _ in range(num_partitions)]
    for c in range(num_classes):
        c_idx = np.where(labels == c)[0]
        np_rng.shuffle(c_idx)
        proportions = np_rng.dirichlet(np.repeat(alpha, num_partitions))
        proportions = proportions / proportions.sum()
        splits = (np.cumsum(proportions) * len(c_idx)).astype(int)[:-1]
        c_splits = np.split(c_idx, splits)
        for i in range(num_partitions):
            client_indices[i].extend(c_splits[i].tolist())

    # Tạo DataLoader riêng cho từng client (chia 80% train, 20% validation cục bộ hoàn toàn in-memory)
    client_partitions = []
    for i in range(num_partitions):
        c_idx = client_indices[i]
        split_point = int(len(c_idx) * 0.8)
        train_idx = [idx + offset for idx in c_idx[:split_point]]
        val_idx = [idx + offset for idx in c_idx[split_point:]]
        client_partitions.append({
            "train": Subset(train_full, train_idx),
            "test": Subset(train_full, val_idx),
        })

    result = {
        "proxy_u": u_dataset,
        "server_val": val_dataset,
        "client_partitions": client_partitions,
        "centralized_test": test_full,
    }
    _cifar_cache[cache_key] = result
    return result


def load_data(
    partition_id: int,
    num_partitions: int,
    batch_size: int,
    alpha: float = 0.5,
    seed: int = 42,
) -> Tuple[DataLoader, DataLoader]:
    splits = _prepare_cifar_splits(seed=seed, alpha=alpha, num_partitions=num_partitions)
    client_part = splits["client_partitions"][partition_id]
    trainloader = DataLoader(client_part["train"], batch_size=batch_size, shuffle=True)
    valloader = DataLoader(client_part["test"], batch_size=batch_size, shuffle=False)
    return trainloader, valloader


def load_centralized_dataset(batch_size: int = 128) -> DataLoader:
    splits = _prepare_cifar_splits()
    return DataLoader(splits["centralized_test"], batch_size=batch_size, shuffle=False)


# ===========================================================================
# 3. VÒNG LẶP HUẤN LUYỆN VÀ KIỂM THỬ SCAFFOLD
# ===========================================================================


def train(
    net: nn.Module,
    trainloader: DataLoader,
    max_steps: int,
    lr: float,
    device: torch.device,
    c_k: Dict[str, torch.Tensor],
    c_global: Dict[str, torch.Tensor],
) -> float:
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(net.parameters(), lr=lr)
    net.train()
    total_loss = 0.0
    total_steps = 0

    while total_steps < max_steps:
        for batch in trainloader:
            if total_steps >= max_steps:
                break
            images = batch["img"].to(device)
            labels = batch["label"].to(device)
            optimizer.zero_grad()
            outputs = net(images)
            loss = criterion(outputs, labels)
            loss.backward()

            # Hiệu chỉnh gradient của SCAFFOLD: g - c_k + c_global
            with torch.no_grad():
                for name, param in net.named_parameters():
                    if param.grad is not None and name in c_global and name in c_k:
                        grad_correction = c_global[name].to(device) - c_k[name].to(device)
                        param.grad.data.add_(grad_correction)

            optimizer.step()
            total_loss += loss.item()
            total_steps += 1

    return total_loss / total_steps if total_steps > 0 else 0.0


def test(
    net: nn.Module,
    testloader: DataLoader,
    device: torch.device,
) -> Tuple[float, float]:
    net.to(device)
    net.eval()
    criterion = nn.CrossEntropyLoss()
    correct, total_loss = 0, 0.0
    num_samples = 0
    num_batches = 0

    with torch.no_grad():
        for batch in testloader:
            images = batch["img"].to(device)
            labels = batch["label"].to(device)
            outputs = net(images)
            total_loss += criterion(outputs, labels).item()
            preds = torch.max(outputs.data, 1)[1]
            correct += (preds == labels).sum().item()
            num_samples += labels.size(0)
            num_batches += 1

    accuracy = float(correct / num_samples) if num_samples > 0 else 0.0
    avg_loss = float(total_loss / num_batches) if num_batches > 0 else 0.0
    return avg_loss, accuracy