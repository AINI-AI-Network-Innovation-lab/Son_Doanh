"""pytorchexample: A Flower / PyTorch app."""

import copy
import torch
from flwr.app import ArrayRecord, Context, Message, MetricRecord, RecordDict
from flwr.clientapp import ClientApp

from task import Net, load_data
from task import test as test_fn
from task import train as train_fn

# Flower ClientApp
app = ClientApp()


@app.train()
def train(msg: Message, context: Context):
    """Train the model on local data using MOON and report discrepancy for FedDisco."""

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # 1. Khởi tạo mô hình hiện tại và mô hình global từ trọng số Server gửi xuống
    global_state_dict = msg.content["arrays"].to_torch_state_dict()

    model = Net()
    model.load_state_dict(global_state_dict)

    global_model = copy.deepcopy(model)

    # 2. Khôi phục mô hình vòng trước (prev_model) từ context.state của Client
    prev_model = Net()
    if "prev_arrays" in context.state.array_records:
        prev_state_dict = context.state.array_records["prev_arrays"].to_torch_state_dict()
        prev_model.load_state_dict(prev_state_dict)
    else:
        # Ở vòng đầu tiên, khởi tạo prev_model bằng global_model
        prev_model.load_state_dict(global_state_dict)

    # 3. Tải dữ liệu cục bộ và độ lệch phân phối (discrepancy)
    partition_id = context.node_config["partition-id"]
    num_partitions = context.node_config["num-partitions"]
    batch_size = context.run_config["batch-size"]
    trainloader, _, discrepancy = load_data(partition_id, num_partitions, batch_size)

    # 4. Đọc cấu hình MOON từ Message config và huấn luyện
    moon_mu = float(msg.content["config"]["moon-mu"])
    temperature = float(msg.content["config"]["temperature"])

    train_loss = train_fn(
        net=model,
        global_net=global_model,
        prev_net=prev_model,
        trainloader=trainloader,
        epochs=context.run_config["local-epochs"],
        lr=float(msg.content["config"]["lr"]),
        device=device,
        moon_mu=moon_mu,
        temperature=temperature,
    )

    # 5. Lưu lại trọng số mô hình vừa huấn luyện vào context.state cho vòng kế tiếp
    updated_model_record = ArrayRecord(model.state_dict())
    context.state.array_records["prev_arrays"] = updated_model_record

    # 6. Đóng gói kết quả kèm chỉ số discrepancy gửi về Server (FedDisco)
    metrics = {
        "train_loss": train_loss,
        "num-examples": len(trainloader.dataset),
        "discrepancy": discrepancy,
    }
    metric_record = MetricRecord(metrics)
    content = RecordDict({"arrays": updated_model_record, "metrics": metric_record})
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
    _, valloader, _ = load_data(partition_id, num_partitions, batch_size)

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