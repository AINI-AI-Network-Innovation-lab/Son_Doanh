"""
Template FedAvg với Flower (simulation chạy local) trên CIFAR-10.

Cài đặt:
    pip install "flwr[simulation]" flwr-datasets[vision] torch torchvision

Chạy:
    python fedavg_flower_cifar10.py
"""
from collections import OrderedDict

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix

import torch
import torch.nn as nn
import torch.nn.functional as F  # các công cụ xây dựng mạng neuron, hàm kích hoạt, hàm mất mát, lớp tuyến tính
from torch.utils.data import DataLoader
from torchvision.transforms import Compose, Normalize, ToTensor #xử lysy ảnh và nạp dữ liêu, xử lý dữ liệu

from flwr.client import ClientApp, NumPyClient  #các thư viện của flower cung cấp khả năng điều phối giao tiếp giữa client và server, cung cấp thuật toán fedavg
from flwr.common import Context, ndarrays_to_parameters
from flwr.server import ServerApp, ServerAppComponents, ServerConfig
from flwr.server.strategy import FedAvg
from flwr.simulation import run_simulation
from flwr_datasets import FederatedDataset  #thư viện cung cấp bộ chia dữ liệu
from flwr_datasets.partitioner import DirichletPartitioner, IidPartitioner

# ----------------------------- Cấu hình -----------------------------
NUM_CLIENTS = 10          # tổng số client
NUM_ROUNDS = 10           # số round T
LOCAL_EPOCHS = 2          # số epoch cục bộ mỗi round
BATCH_SIZE = 32
LR = 0.01
FRACTION_FIT = 0.5        # tỉ lệ client được chọn mỗi round
PARTITION = "dirichlet"         # "iid" hoặc "dirichlet"
DIRICHLET_ALPHA = 0.5     # nhỏ hơn -> non-IID mạnh hơn
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
NUM_WORKERS = 0           # để 0 trên Windows cho đỡ lỗi multiprocessing


# ----------------------------- Model -----------------------------
class Net(nn.Module):   #mạng cnn kiểu LeNet gồm 2 lớp tích chập + max pool + 3 lớp fully connected
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 16 * 5 * 5)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.fc3(x)


def get_weights(model):  #xuất trọng số từ pytorch tensor của model dưới dạng numpy array
    return [v.cpu().numpy() for v in model.state_dict().values()]


def set_weights(model, weights):  #nhận numpy từ server, chuyển thành pytorch tensor và set trọng số cho model
    keys = model.state_dict().keys()
    state = OrderedDict({k: torch.tensor(v) for k, v in zip(keys, weights)})
    model.load_state_dict(state, strict=True)


# ----------------------------- Dữ liệu -----------------------------
_fds = None  # cache để không phải khởi tạo lại dataset ở mỗi client


def get_fds():  #tải dataset CIFAR-10 từ federated dataset và chia cho các client theo iid hoặc dirichlet
    global _fds
    if _fds is None:
        if PARTITION == "iid":
            partitioner = IidPartitioner(num_partitions=NUM_CLIENTS)
        else:
            partitioner = DirichletPartitioner(
                num_partitions=NUM_CLIENTS,
                partition_by="label",
                alpha=DIRICHLET_ALPHA,
                min_partition_size=10,
                self_balancing=True,
            )
        _fds = FederatedDataset(
            dataset="uoft-cs/cifar10",
            partitioners={"train": partitioner},
        )
    return _fds


_tf = Compose([ToTensor(), Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]) #chuẩn hóa dữ liệu và chuyển về torch tensor


def apply_transforms(batch): #áp dụng transforms cho batch từ ảnh thành tensor và chuẩn hóa về dải -1 => 1 
    batch["img"] = [_tf(img) for img in batch["img"]]
    return batch


def load_client_data(partition_id: int): # tải dữ liệu vào client với partition_id cụ thể và chia 80% cho train và 20% cho tập valid
    partition = get_fds().load_partition(partition_id)
    split = partition.train_test_split(test_size=0.2, seed=42)
    split = split.with_transform(apply_transforms)
    trainloader = DataLoader(split["train"], batch_size=BATCH_SIZE, shuffle=True,
                             num_workers=NUM_WORKERS)
    valloader = DataLoader(split["test"], batch_size=BATCH_SIZE,
                           num_workers=NUM_WORKERS)
    return trainloader, valloader


def load_centralized_testset(): #tải tập test cho server
    testset = get_fds().load_split("test").with_transform(apply_transforms)
    return DataLoader(testset, batch_size=128, num_workers=NUM_WORKERS)

# ----------------------------- Trực quan hóa dữ liệu -----------------------------
def plot_label_distribution():
    print("Đang vẽ biểu đồ phân phối nhãn...")
    fds = get_fds()
    label_counts = np.zeros((NUM_CLIENTS, 10))
    
    for i in range(NUM_CLIENTS):
        partition = fds.load_partition(i)
        for row in partition:
            label_counts[i, row["label"]] += 1
            
    plt.figure(figsize=(10, 6))
    sns.heatmap(label_counts, annot=True, fmt="g", cmap="Blues")
    plt.title("Phân phối nhãn (Label Distribution) trên các Client")
    plt.xlabel("Lớp (Class 0-9)")
    plt.ylabel("Client ID")
    plt.tight_layout()
    plt.savefig("label_distribution.png")
    plt.close()
    print("Đã lưu label_distribution.png")

# ----------------------------- Train / Test -----------------------------
def train(model, loader, epochs, lr):
    model.to(DEVICE).train()
    opt = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9)
    for _ in range(epochs):
        for batch in loader:
            x, y = batch["img"].to(DEVICE), batch["label"].to(DEVICE)
            opt.zero_grad()
            F.cross_entropy(model(x), y).backward()
            opt.step()


@torch.no_grad() # đang test nên không cần tính và lưu trữ gradient để chạy backward
def test(model, loader): # hàm tính loss và accuracy
    model.to(DEVICE).eval() # chuyển sang chế độ evaluation, ko dùng dropout tắt neuron bất kì nữa
    loss, correct, n = 0.0, 0, 0
    y_true = []
    y_pred = []
    for batch in loader:
        x, y = batch["img"].to(DEVICE), batch["label"].to(DEVICE)
        out = model(x)
        loss += F.cross_entropy(out, y, reduction="sum").item()
        preds = out.argmax(1)
        correct += (preds == y).sum().item()
        n += y.size(0)
        y_true.extend(y.cpu().numpy())
        y_pred.extend(preds.cpu().numpy())
    return loss / n, correct / n, y_true, y_pred


# ----------------------------- Client -----------------------------
class FlowerClient(NumPyClient):
    def __init__(self, trainloader, valloader, partition_id=-1):
        self.model = Net()
        self.trainloader = trainloader
        self.valloader = valloader
        self.partition_id = partition_id

    def fit(self, parameters, config):
        set_weights(self.model, parameters)  # nhận w^t từ server
        train(self.model, self.trainloader,
              epochs=config["local_epochs"], lr=config["lr"])
        return get_weights(self.model), len(self.trainloader.dataset), {}

    def evaluate(self, parameters, config): #eval trên tập valid set của client
        set_weights(self.model, parameters)
        loss, acc, _, _ = test(self.model, self.valloader)
        return loss, len(self.valloader.dataset), {"accuracy": acc, "partition_id": self.partition_id}


def client_fn(context: Context): #tạo client
    partition_id = context.node_config["partition-id"]
    trainloader, valloader = load_client_data(partition_id)
    return FlowerClient(trainloader, valloader, partition_id).to_client()


client_app = ClientApp(client_fn=client_fn) #không cần cấu hình giao tiếp, bọc tất cả client vào trong 1 cái app để nó tự nói chuyện với server


# ----------------------------- Server -----------------------------
client_metrics_history = {i: [] for i in range(NUM_CLIENTS)}
global_metrics_history = []  # list of (round, loss, acc)
global_model_final = Net()

def fit_config(server_round: int): #config cho mỗi round, gửi siêu tham số từ server xuống client tại đầu mỗi round
    return {"local_epochs": LOCAL_EPOCHS, "lr": LR}


def weighted_average(metrics): #tính độ chính xác trung bình dựa trên số mẫu của từng client, nếu client nào có nhiều mẫu hơn thì có trọng số cao hơn
    total = sum(n for n, _ in metrics)
    for n, m in metrics:
        if "partition_id" in m and "accuracy" in m and m["partition_id"] != -1:
            client_metrics_history[m["partition_id"]].append(m["accuracy"])
    return {"accuracy": sum(n * m["accuracy"] for n, m in metrics) / total}


def centralized_evaluate(server_round, parameters_ndarrays, config): #eval trên tập test của server
    """Đánh giá global model trên test set CIFAR-10 đầy đủ."""
    set_weights(global_model_final, parameters_ndarrays)
    loss, acc, _, _ = test(global_model_final, load_centralized_testset())
    global_metrics_history.append((server_round, loss, acc))
    print(f"[Round {server_round}] global test loss={loss:.4f} acc={acc:.4f}")
    return loss, {"centralized_accuracy": acc}


def server_fn(context: Context): #tạo server và thuật toán FedAvg
    init_params = ndarrays_to_parameters(get_weights(Net()))
    strategy = FedAvg(
        fraction_fit=FRACTION_FIT,
        fraction_evaluate=FRACTION_FIT,
        min_fit_clients=2,
        min_evaluate_clients=2,
        min_available_clients=NUM_CLIENTS,
        initial_parameters=init_params,
        on_fit_config_fn=fit_config,
        evaluate_fn=centralized_evaluate,
        evaluate_metrics_aggregation_fn=weighted_average,
    )
    return ServerAppComponents(strategy=strategy,
                               config=ServerConfig(num_rounds=NUM_ROUNDS))


server_app = ServerApp(server_fn=server_fn)


# ----------------------------- Trực quan hóa kết quả -----------------------------
def plot_results():
    print("Đang vẽ biểu đồ kết quả...")
    
    # 1. Vẽ Loss & Accuracy của Server (Global Model)
    if len(global_metrics_history) > 0:
        rounds = [t[0] for t in global_metrics_history]
        losses = [t[1] for t in global_metrics_history]
        accuracies = [t[2] for t in global_metrics_history]
        
        plt.figure(figsize=(12, 5))
        plt.subplot(1, 2, 1)
        plt.plot(rounds, losses, marker='o')
        plt.title("Global Test Loss")
        plt.xlabel("Round")
        plt.ylabel("Loss")
        plt.grid(True)
        
        plt.subplot(1, 2, 2)
        plt.plot(rounds, accuracies, marker='o', color='orange')
        plt.title("Global Test Accuracy")
        plt.xlabel("Round")
        plt.ylabel("Accuracy")
        plt.grid(True)
        plt.tight_layout()
        plt.savefig("global_metrics.png")
        plt.close()
        print("Đã lưu global_metrics.png")
    else:
        print("Không có dữ liệu global metrics để vẽ.")
    
    # 2. Vẽ Accuracy của từng Client
    has_client_data = any(len(v) > 0 for v in client_metrics_history.values())
    if has_client_data:
        plt.figure(figsize=(10, 6))
        for cid, acc_list in client_metrics_history.items():
            if len(acc_list) > 0:
                plt.plot(acc_list, label=f"Client {cid}")
        plt.title("Validation Accuracy của từng Client")
        plt.xlabel("Lần evaluate")
        plt.ylabel("Accuracy")
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.grid(True)
        plt.tight_layout()
        plt.savefig("client_accuracies.png")
        plt.close()
        print("Đã lưu client_accuracies.png")
    else:
        print("Không có dữ liệu client metrics để vẽ.")
    
    # 3. Vẽ Confusion Matrix của Global Model cuối cùng
    if len(global_metrics_history) > 0:
        _, _, y_true, y_pred = test(global_model_final, load_centralized_testset())
        cm = confusion_matrix(y_true, y_pred)
        
        classes = ['plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']
        plt.figure(figsize=(10, 8))
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=classes, yticklabels=classes)
        plt.title("Confusion Matrix - Final Global Model")
        plt.xlabel("Predicted Label")
        plt.ylabel("True Label")
        plt.tight_layout()
        plt.savefig("confusion_matrix.png")
        plt.close()
        print("Đã lưu confusion_matrix.png")
    else:
        print("Không có global model để vẽ confusion matrix.")

# ----------------------------- Chạy simulation -----------------------------
if __name__ == "__main__":  # bắt buộc trên Windows
    plot_label_distribution()
    
    run_simulation(
        server_app=server_app,
        client_app=client_app,
        num_supernodes=NUM_CLIENTS,
        backend_config={
            "client_resources": {
                "num_cpus": 1,
                "num_gpus": 0.1 if torch.cuda.is_available() else 0.0,
            }
        },
    )
    
    plot_results()