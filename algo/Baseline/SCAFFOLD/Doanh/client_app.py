"""pytorchexample: A Flower / PyTorch app."""

import torch
from flwr.app import ArrayRecord, Context, Message, MetricRecord, RecordDict
from flwr.clientapp import ClientApp

from task import (
    Net,
    compute_shannon_entropy,
    get_zero_control_variate,
    load_data,
)
from task import test as test_fn
from task import train as train_fn

# Flower ClientApp
app = ClientApp()


@app.train()
def train(msg: Message, context: Context):
    """Train the model on local data using SCAFFOLD and compute soft labels for FedEntropy."""

    # 1. Khởi tạo mô hình từ trọng số Server gửi xuống
    model = Net()
    model.load_state_dict(msg.content["arrays"].to_torch_state_dict())
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model.to(device)

    # 2. Đọc biến kiểm soát server_c từ Message và khôi phục client_c từ context.state
    server_c = msg.content["server_c"].to_torch_state_dict()
    if "client_c" in context.state.array_records:
        client_c = context.state.array_records["client_c"].to_torch_state_dict()
    else:
        client_c = get_zero_control_variate(model)

    # 3. Tải dữ liệu phân mảnh cục bộ
    partition_id = context.node_config["partition-id"]
    num_partitions = context.node_config["num-partitions"]
    batch_size = context.run_config["batch-size"]
    trainloader, _ = load_data(partition_id, num_partitions, batch_size)

    # 4. Huấn luyện mô hình cục bộ với SCAFFOLD
    train_loss, new_client_c, delta_c, avg_soft_labels = train_fn(
        net=model,
        trainloader=trainloader,
        epochs=context.run_config["local-epochs"],
        lr=float(msg.content["config"]["lr"]),
        device=device,
        server_c=server_c,
        client_c=client_c,
    )

    # 5. Lưu client_c mới vào context.state cho vòng huấn luyện tiếp theo
    context.state.array_records["client_c"] = ArrayRecord(new_client_c)

    # 6. Đóng gói mô hình, delta_c (SCAFFOLD) và soft_labels (FedEntropy) gửi về Server
    model_record = ArrayRecord(model.state_dict())
    delta_c_record = ArrayRecord(delta_c)
    local_entropy = compute_shannon_entropy(avg_soft_labels)

    metrics = {
        "train_loss": train_loss,
        "num-examples": len(trainloader.dataset),
        "soft_labels": avg_soft_labels,
        "local_entropy": local_entropy,
    }
    metric_record = MetricRecord(metrics)
    content = RecordDict(
        {
            "arrays": model_record,
            "delta_c": delta_c_record,
            "metrics": metric_record,
        }
    )
    return Message(content=content, reply_to=msg)


@app.evaluate()
def evaluate(msg: Message, context: Context):
    """Evaluate the model on local data."""

    model = Net()
    model.load_state_dict(msg.content["arrays"].to_torch_state_dict())
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model.to(device)

    partition_id = context.node_config["partition-id"]
    num_partitions = context.node_config["num-partitions"]
    batch_size = context.run_config["batch-size"]
    _, valloader = load_data(partition_id, num_partitions, batch_size)

    eval_loss, eval_acc = test_fn(
        model,
        valloader,
        device,
    )

    metrics = {
        "eval_loss": eval_loss,
        "eval_acc": eval_acc,
        "num-examples": len(valloader.dataset),
    }
    metric_record = MetricRecord(metrics)
    content = RecordDict({"metrics": metric_record})
    return Message(content=content, reply_to=msg)