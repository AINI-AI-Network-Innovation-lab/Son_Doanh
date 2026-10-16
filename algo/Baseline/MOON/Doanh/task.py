"""pytorchexample: A Flower / PyTorch app with MOON & FedDisco."""

from collections import Counter
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from datasets import load_dataset
from flwr_datasets import FederatedDataset
from flwr_datasets.partitioner import DirichletPartitioner
from torch.utils.data import DataLoader
from torchvision.transforms import Compose, Normalize, ToTensor


class Net(nn.Module):
    def __init__(self, num_classes: int = 10, proj_dim: int = 128):
        super(Net, self).__init__()
        # Backbone dựa trên thiết kế CNN ban đầu
        self.conv1 = nn.Conv2d(3, 32, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(32, 64, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(64 * 8 * 8, 256)

        # Projection head cho MOON
        self.proj_head = nn.Sequential(
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, proj_dim),
        )
        # Tầng phân loại cuối (Classifier)
        self.fc2 = nn.Linear(proj_dim, num_classes)

    def forward(self, x):
        x = self.pool(F.relu(self.bn1(self.conv1(x))))  # 32x32 -> 16x16
        x = self.pool(F.relu(self.bn2(self.conv2(x))))  # 16x16 -> 8x8
        x = x.view(-1, 64 * 8 * 8)
        h = F.relu(self.fc1(x))

        z = self.proj_head(h)
        logits = self.fc2(z)
        return logits, z


fds = None  # Cache FederatedDataset

pytorch_transforms = Compose([ToTensor(), Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])


def apply_transforms(batch):
    """Apply transforms to the partition from FederatedDataset."""
    batch["img"] = [pytorch_transforms(img) for img in batch["img"]]
    return batch


def compute_discrepancy(dataset, num_classes: int = 10) -> float:
    """Tính độ lệch phân phối L1 (d_k) giữa Client và phân phối đều toàn cục."""
    labels = dataset["label"]
    counts = Counter(labels)
    total = len(labels)
    local_dist = np.array([counts.get(i, 0) / total for i in range(num_classes)])
    global_dist = np.full(num_classes, 1.0 / num_classes)
    # Khoảng cách L1 giữa phân phối cục bộ và toàn cục
    return float(np.sum(np.abs(local_dist - global_dist)))


def load_data(partition_id: int, num_partitions: int, batch_size: int):
    """Load partition CIFAR10 data và tính discrepancy cho FedDisco."""
    global fds
    if fds is None:
        # Sử dụng DirichletPartitioner để mô phỏng Non-IID (có thể đổi lại IidPartitioner)
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

    # Tính độ lệch phân phối trên tập train cục bộ trước khi bọc transform
    discrepancy = compute_discrepancy(partition_train_test["train"], num_classes=10)

    partition_train_test = partition_train_test.with_transform(apply_transforms)
    trainloader = DataLoader(
        partition_train_test["train"], batch_size=batch_size, shuffle=True
    )
    testloader = DataLoader(partition_train_test["test"], batch_size=batch_size)
    return trainloader, testloader, discrepancy


def load_centralized_dataset():
    """Load test set and return dataloader."""
    test_dataset = load_dataset("uoft-cs/cifar10", split="test")
    dataset = test_dataset.with_transform(apply_transforms)
    return DataLoader(dataset, batch_size=128)


def train(
    net: nn.Module,
    global_net: nn.Module,
    prev_net: nn.Module,
    trainloader: DataLoader,
    epochs: int,
    lr: float,
    device: torch.device,
    moon_mu: float = 1.0,
    temperature: float = 0.5,
):
    """Huấn luyện mô hình cục bộ sử dụng MOON (Model-Contrastive Loss)."""
    net.to(device)
    global_net.to(device)
    prev_net.to(device)

    global_net.eval()
    prev_net.eval()
    for param in global_net.parameters():
        param.requires_grad = False
    for param in prev_net.parameters():
        param.requires_grad = False

    criterion = nn.CrossEntropyLoss().to(device)
    cos = nn.CosineSimilarity(dim=-1).to(device)
    optimizer = torch.optim.SGD(net.parameters(), lr=lr, momentum=0.9, weight_decay=1e-5)

    net.train()
    running_loss = 0.0

    for _ in range(epochs):
        for batch in trainloader:
            images = batch["img"].to(device)
            labels = batch["label"].to(device)

            optimizer.zero_grad()

            logits, z = net(images)
            task_loss = criterion(logits, labels)

            # Tính Contrastive Loss của MOON
            if moon_mu > 0:
                with torch.no_grad():
                    _, z_glob = global_net(images)
                    _, z_prev = prev_net(images)

                sim_glob = cos(z, z_glob) / temperature
                sim_prev = cos(z, z_prev) / temperature

                logits_con = torch.cat(
                    [sim_glob.unsqueeze(1), sim_prev.unsqueeze(1)], dim=1
                )
                labels_con = torch.zeros(
                    images.size(0), dtype=torch.long, device=device
                )
                loss_con = criterion(logits_con, labels_con)
                loss = task_loss + moon_mu * loss_con
            else:
                loss = task_loss

            loss.backward()
            optimizer.step()

            running_loss += loss.item()

    avg_trainloss = running_loss / (epochs * len(trainloader))
    return avg_trainloss


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
            logits, _ = net(images)
            loss += criterion(logits, labels).item()
            correct += (torch.max(logits.data, 1)[1] == labels).sum().item()
    accuracy = correct / len(testloader.dataset)
    loss = loss / len(testloader)
    return loss, accuracy