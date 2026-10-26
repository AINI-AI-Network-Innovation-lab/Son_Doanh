import math
from typing import Dict, List, Tuple
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import datasets
from datasets import load_dataset
from torch.utils.data import DataLoader, Subset
from torchvision.transforms import Compose, Normalize, ToTensor

datasets.disable_caching()


# ===========================================================================
# 1. KIẾN TRÚC MÔ HÌNH MẠNG (MODEL ARCHITECTURE)
# ===========================================================================


# Khối tích chập đơn vị (ConvBlock): gồm Conv2d -> BatchNorm2d -> ReLU -> (tùy chọn MaxPool2d)
class ConvBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, pool: bool = False):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.pool = nn.MaxPool2d(2, 2) if pool else None

    # Lan truyền xuôi qua khối tích chập
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        if self.pool is not None:
            x = self.pool(x)
        return x


# Mô hình CNN 6 tầng tích chập chuẩn theo bài báo FedSL (Net) cho CIFAR-10
# Gồm 6 tầng tích chập độc lập (layer0 đến layer5) và 1 tầng phân loại tuyến tính (classifier)
class Net(nn.Module):
    def __init__(self, num_classes: int = 10):
        super().__init__()
        # 6 tầng layer được chia tách riêng biệt để phục vụ cơ chế Split Layer của FedSL
        self.layer0 = ConvBlock(3, 32, pool=False)      # Layer 0: ảnh 32x32 -> 32x32
        self.layer1 = ConvBlock(32, 64, pool=True)      # Layer 1: 32x32 -> 16x16
        self.layer2 = ConvBlock(64, 128, pool=False)    # Layer 2: 16x16 -> 16x16
        self.layer3 = ConvBlock(128, 128, pool=True)    # Layer 3: 16x16 -> 8x8
        self.layer4 = ConvBlock(128, 256, pool=False)   # Layer 4: 8x8 -> 8x8
        self.layer5 = ConvBlock(256, 256, pool=True)    # Layer 5: 8x8 -> 4x4
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.classifier = nn.Linear(256, num_classes)   # Bộ phân loại tách riêng, gộp chung toàn hệ thống

    # Lan truyền xuôi qua 6 tầng conv và bộ phân loại
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



# 2. CÔNG CỤ BÓC TÁCH LAYER, TÍNH KHOẢNG CÁCH VÀ DUNG LƯỢNG MẠNG



# Hàm bóc tách cấu trúc mạng thành các nhóm tầng layer độc lập và bộ phân loại:
# - split_layer_keys: Dict ánh xạ layer_idx (0..5) -> danh sách toàn bộ key của tầng đó (dùng cho vòng lặp gộp tầng)
# - classifier_keys: Danh sách key thuộc bộ phân loại cuối (Linear)
def get_layer_groups(
    model: nn.Module,
) -> Tuple[Dict[int, List[str]], List[str]]:
    state_keys = list(model.state_dict().keys())
    split_layer_keys: Dict[int, List[str]] = {}
    for l in range(6):
        prefix = f"layer{l}."
        split_layer_keys[l] = [k for k in state_keys if k.startswith(prefix)]
    classifier_keys = [k for k in state_keys if k.startswith("classifier.")]
    return split_layer_keys, classifier_keys


# Hàm tính khoảng cách Euclid (chuẩn L2) đo độ biến thiên của từng tầng mạng sau khi train cục bộ:
# Công thức FedASL: c_{l,k}^t = ||w_{l,k}^t - w_{l,k}^{t-1}||_2
# Chỉ tính trên các trọng số học được (weight và bias), tự động bỏ qua các buffer
# Đầu ra: dict {layer_idx: khoảng_cách_euclid}
def compute_layer_distances(
    initial_state: Dict[str, torch.Tensor],
    final_state: Dict[str, torch.Tensor],
    split_layer_keys: Dict[int, List[str]],
) -> Dict[int, float]:
    distances: Dict[int, float] = {}
    for l, keys in split_layer_keys.items():
        sq_sum = 0.0
        for k in keys:
            if ("weight" in k or "bias" in k) and k in initial_state and k in final_state:
                diff = (final_state[k].float() - initial_state[k].float()).view(-1)
                sq_sum += float(torch.sum(diff * diff).item())
        distances[l] = float(math.sqrt(sq_sum))
    return distances


# Hàm tính tổng dung lượng truyền tải mạng (tính bằng byte) của các layer được chọn gửi lên server:
# Giả định kiểu dữ liệu float32 tương ứng 4 bytes cho mỗi tham số
def calculate_param_bytes(state_dict: Dict[str, torch.Tensor], keys: List[str]) -> int:
    total_elements = 0
    for k in keys:
        if k in state_dict:
            total_elements += state_dict[k].numel()
    return total_elements * 4



# 3. TIỀN XỬ LÝ VÀ PHÂN HOẠCH DỮ LIỆU (DATASET PARTITIONING)


# Chuẩn hóa ảnh CIFAR-10 về khoảng [-1, 1]
pytorch_transforms = Compose([ToTensor(), Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])


# Áp dụng hàm biến đổi transform cho từng lô ảnh
def apply_transforms(batch):
    batch["img"] = [pytorch_transforms(img) for img in batch["img"]]
    return batch


# Bộ nhớ đệm phân vùng CIFAR-10 tránh tải lại nhiều lần
_cifar_cache: Dict = {}


# Hàm chuẩn bị và phân hoạch dữ liệu theo đúng kế hoạch myAlgo.md:
# 1. 2.000 ảnh làm tập proxy không nhãn U ở server
# 2. 3.000 ảnh làm tập validation đánh giá server
# 3. 45.000 ảnh chia cho các client theo phân phối Dirichlet non-IID (alpha)
# 4. 10.000 ảnh test chuẩn để kiểm thử toàn cục
def _prepare_cifar_splits(
    seed: int = 42,
    alpha: float = 0.5,
    num_partitions: int = 10,
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


# Hàm tải DataLoader tập train và test cục bộ của một Client cụ thể (nạp và chia train test dữ liệu được chia theo partition_id)
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


# DataLoader load data ko nhãn lên server
def load_unlabeled_proxy_dataset(
    batch_size: int = 64,
    proxy_samples: int = 2000,
    seed: int = 42,
) -> DataLoader:
    splits = _prepare_cifar_splits(seed=seed, proxy_samples=proxy_samples)
    return DataLoader(splits["proxy_u"], batch_size=batch_size, shuffle=False)


# DataLoader load data validation tại server
def load_server_val_dataset(batch_size: int = 128, seed: int = 42) -> DataLoader:
    splits = _prepare_cifar_splits(seed=seed)
    return DataLoader(splits["server_val"], batch_size=batch_size, shuffle=False)


# Hàm tải DataLoader tập kiểm thử toàn cục chuẩn CIFAR-10 (10.000 mẫu)
def load_centralized_dataset(batch_size: int = 128) -> DataLoader:
    splits = _prepare_cifar_splits()
    return DataLoader(splits["centralized_test"], batch_size=batch_size, shuffle=False)


# 4. VÒNG LẶP HUẤN LUYỆN VÀ KIỂM THỬ (TRAIN & TEST)


# Hàm huấn luyện mô hình cục bộ tại Client bằng SGD với momentum = 0.9
def train(
    net: nn.Module,
    trainloader: DataLoader,
    epochs: int,
    lr: float,
    device: torch.device,
) -> float:
    net.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(net.parameters(), lr=lr, momentum=0.9)
    net.train()

    total_loss = 0.0
    total_steps = 0
    for _ in range(epochs):
        for batch in trainloader:
            images = batch["img"].to(device)
            labels = batch["label"].to(device)
            optimizer.zero_grad()
            outputs = net(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            total_steps += 1

    return total_loss / total_steps if total_steps > 0 else 0.0


# Hàm đánh giá chất lượng mô hình trên DataLoader (trả về avg_loss và accuracy)
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
