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
# 1. KIẾN TRÚC MÔ HÌNH MẠNG MOON (6 LỚP CNN + PROJECTION HEAD + CLASSIFIER)
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
        # 1. Base Feature Extractor: 6 tầng CNN chuẩn
        self.layer0 = ConvBlock(3, 32, pool=False)      # Layer 0: 32x32 -> 32x32
        self.layer1 = ConvBlock(32, 64, pool=True)      # Layer 1: 32x32 -> 16x16
        self.layer2 = ConvBlock(64, 128, pool=False)    # Layer 2: 16x16 -> 16x16
        self.layer3 = ConvBlock(128, 128, pool=True)    # Layer 3: 16x16 -> 8x8
        self.layer4 = ConvBlock(128, 256, pool=False)   # Layer 4: 8x8 -> 8x8
        self.layer5 = ConvBlock(256, 256, pool=True)    # Layer 5: 8x8 -> 4x4
        self.pool = nn.AdaptiveAvgPool2d((1, 1))

        # 2. Projection Head: MLP 2 tầng (256 -> 256 -> 256) theo bài báo MOON
        self.proj = nn.Sequential(
            nn.Linear(256, 256),
            nn.ReLU(inplace=True),
            nn.Linear(256, 256),
        )

        # 3. Classifier: 256 -> 10 classes
        self.classifier = nn.Linear(256, num_classes)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        x = self.layer0(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.layer5(x)
        x = self.pool(x)
        h = torch.flatten(x, 1)

        z = self.proj(h)
        logits = self.classifier(z)
        return logits, z


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


# ==============================================================================
# 3. HÀM MẤT MÁT TƯƠNG PHẢN (MODEL-CONTRASTIVE LOSS)
# ==============================================================================


def model_contrastive_loss(
    z: torch.Tensor,
    z_glob: torch.Tensor,
    z_prev: torch.Tensor | None,
    temperature: float = 0.5,
) -> torch.Tensor:
    cos_sim = nn.CosineSimilarity(dim=-1)
    sim_pos = cos_sim(z, z_glob).reshape(-1, 1) / temperature

    if z_prev is None:
        return torch.tensor(0.0, device=z.device)

    sim_neg = cos_sim(z, z_prev).reshape(-1, 1) / temperature
    logits = torch.cat([sim_pos, sim_neg], dim=1)
    labels = torch.zeros(logits.size(0), dtype=torch.long, device=z.device)

    criterion = nn.CrossEntropyLoss()
    return criterion(logits, labels)


# ==============================================================================
# 4. HUẤN LUYỆN CỤC BỘ VÀ KIỂM THỬ MOON
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
    model.train()
    global_model.eval()
    if prev_model is not None:
        prev_model.eval()

    optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=1e-5)
    criterion = nn.CrossEntropyLoss()

    running_loss = 0.0
    total_steps = 0

    while total_steps < s_k:
        for batch in trainloader:
            if total_steps >= s_k:
                break

            images = batch["img"].to(device)
            labels = batch["label"].to(device)

            logits, z = model(images)

            with torch.no_grad():
                _, z_glob = global_model(images)
                z_prev = None
                if prev_model is not None:
                    _, z_prev = prev_model(images)

            loss_ce = criterion(logits, labels)
            loss_con = model_contrastive_loss(z, z_glob, z_prev, temperature=temperature)
            loss_total = loss_ce + mu * loss_con

            optimizer.zero_grad()
            loss_total.backward()
            optimizer.step()

            running_loss += loss_total.item()
            total_steps += 1

    return running_loss / max(1, total_steps)


def test(net: nn.Module, testloader: DataLoader, device: torch.device) -> tuple[float, float]:
    net.to(device)
    net.eval()
    criterion = nn.CrossEntropyLoss()
    correct, total_loss = 0, 0.0

    with torch.no_grad():
        for batch in testloader:
            images = batch["img"].to(device)
            labels = batch["label"].to(device)

            logits, _ = net(images)
            total_loss += criterion(logits, labels).item()

            predicted = torch.max(logits.data, 1)[1]
            correct += (predicted == labels).sum().item()

    num_samples = len(testloader.dataset)
    num_batches = len(testloader)

    accuracy = float(correct / num_samples) if num_samples > 0 else 0.0
    avg_loss = float(total_loss / num_batches) if num_batches > 0 else 0.0
    return avg_loss, accuracy
