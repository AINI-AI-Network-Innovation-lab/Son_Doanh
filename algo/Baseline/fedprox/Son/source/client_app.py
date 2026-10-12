"""ClientApp cho thuật toán FedProx trên nền tảng Flower.

File này định nghĩa:
1. Quy trình huấn luyện cục bộ tại Client (@app.train): nhận trọng số server, train với hàm loss có FedProx proximal term.
2. Quy trình đánh giá cục bộ tại Client (@app.evaluate): đánh giá mô hình trên dữ liệu kiểm thử riêng của Client.
"""

import torch
from flwr.app import ArrayRecord, Context, Message, MetricRecord, RecordDict
from flwr.clientapp import ClientApp

# Import kiến trúc mạng, hàm load dữ liệu và các hàm train/test từ task.py
from source.task import Net, load_data
from source.task import test as test_fn    # Đổi tên thành test_fn để không trùng với hàm evaluate của ClientApp
from source.task import train as train_fn  # Đổi tên thành train_fn để không trùng với hàm train của ClientApp

# Khởi tạo đối tượng ClientApp đại diện cho ứng dụng chạy trên mỗi Client/Node
app = ClientApp()


@app.train()
def train(msg: Message, context: Context) -> Message:
    """Hàm callback được gọi khi Server yêu cầu Client thực hiện một vòng huấn luyện cục bộ (local training).

    Args:
        msg: Tin nhắn từ Server gửi xuống, chứa:
             - msg.content["arrays"]: Trọng số toàn cục hiện tại của Server (Global Weights).
             - msg.content["config"]: Các cấu hình bổ sung (như learning rate 'lr', 'proximal-mu').
        context: Chứa thông tin cấu hình runtime của Node/Client:
             - context.node_config: Thông tin node (ví dụ: partition-id của dữ liệu).
             - context.run_config: Các thông số chung từ pyproject.toml (epochs, batch-size, mu...).

    Returns:
        Message: Tin nhắn phản hồi gửi ngược lại cho Server chứa trọng số mới và các chỉ số đo lường (metrics).
    """
    # 1. Khởi tạo mô hình và nạp trọng số toàn cục nhận được từ Server
    model = Net()
    model.load_state_dict(msg.content["arrays"].to_torch_state_dict())
    
    # Ưu tiên sử dụng GPU (CUDA) nếu máy hỗ trợ, ngược lại chạy trên CPU
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model.to(device)

    # 2. Tải phần dữ liệu (partition) dành riêng cho Client này
    partition_id = context.node_config["partition-id"]      # ID của client (0, 1, 2, ...)
    num_partitions = context.node_config["num-partitions"]  # Tổng số client tham gia chia dữ liệu
    batch_size = context.run_config["batch-size"]           # Kích thước batch đọc từ pyproject.toml
    trainloader, _ = load_data(partition_id, num_partitions, batch_size)

    # 3. Lấy hệ số phạt proximal term (mu) của FedProx:
    # Ưu tiên lấy từ tin nhắn Server gửi xuống ('proximal-mu'), nếu không có thì lấy từ run_config, mặc định 0.01
    mu = msg.content.get("config", {}).get("proximal-mu", context.run_config.get("mu", 0.01))

    # 4. Thực hiện huấn luyện cục bộ bằng hàm train_fn (được định nghĩa trong task.py)
    # Hàm này sẽ tính loss bao gồm: CrossEntropyLoss + Proximal Term phạt độ lệch trọng số
    train_loss = train_fn(
        model,
        trainloader,
        context.run_config["local-epochs"],  # Số epoch chạy ở mỗi vòng
        msg.content["config"]["lr"],          # Learning rate do Server cấu hình gửi xuống
        device,
        mu=mu,                               # Hệ số phạt của FedProx
    )

    # 5. Đóng gói kết quả gửi trả về cho Server:
    # - Arrays: Trọng số mô hình sau khi Client đã huấn luyện xong (State Dict -> ArrayRecord)
    model_record = ArrayRecord(model.state_dict())
    
    # - Metrics: Các số liệu thống kê tại Client để Server tổng hợp (train loss, số lượng mẫu train)
    metrics = {
        "train_loss": train_loss,
        "num-examples": len(trainloader.dataset),  # Dùng để Server tính trung bình có trọng số (Weighted Average)
    }
    metric_record = MetricRecord(metrics)
    
    content = RecordDict({"arrays": model_record, "metrics": metric_record})
    return Message(content=content, reply_to=msg)


@app.evaluate()
def evaluate(msg: Message, context: Context) -> Message:
    """Hàm callback được gọi khi Server yêu cầu Client tự đánh giá trên tập kiểm thử cục bộ (local validation).

    Args:
        msg: Tin nhắn từ Server chứa trọng số toàn cục cần đánh giá.
        context: Cấu hình runtime của Node.

    Returns:
        Message: Chứa loss và độ chính xác (eval_loss, eval_acc) tại Client này.
    """
    # 1. Khởi tạo mô hình và nạp trọng số nhận từ Server
    model = Net()
    model.load_state_dict(msg.content["arrays"].to_torch_state_dict())
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model.to(device)

    # 2. Tải tập dữ liệu kiểm thử (validation set) của Client
    partition_id = context.node_config["partition-id"]
    num_partitions = context.node_config["num-partitions"]
    batch_size = context.run_config["batch-size"]
    _, valloader = load_data(partition_id, num_partitions, batch_size)

    # 3. Chạy hàm kiểm thử test_fn (định nghĩa trong task.py)
    eval_loss, eval_acc = test_fn(
        model,
        valloader,
        device,
    )

    # 4. Đóng gói kết quả đánh giá gửi về cho Server
    metrics = {
        "eval_loss": eval_loss,
        "eval_acc": eval_acc,
        "num-examples": len(valloader.dataset),
    }
    metric_record = MetricRecord(metrics)
    content = RecordDict({"metrics": metric_record})
    return Message(content=content, reply_to=msg)