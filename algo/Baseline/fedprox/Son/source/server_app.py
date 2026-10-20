"""ServerApp cho thuật toán FedProx trên nền tảng Flower.

File này định nghĩa:
1. Quy trình điều phối của Server (lựa chọn client, gửi trọng số, nhận và tổng hợp trọng số).
2. Chiến lược tổng hợp (Strategy): FedProx (với hệ số phạt proximal_mu).
3. Đánh giá mô hình toàn cục sau mỗi vòng (round) trên tập dữ liệu kiểm tra tập trung.
"""

import torch
from flwr.app import ArrayRecord, ConfigRecord, Context, MetricRecord
from flwr.serverapp import Grid, ServerApp
from flwr.serverapp.strategy import FedProx  # Chiến lược FedProx chuẩn của Flower ServerApp

from source.task import Net, load_centralized_dataset, test

# Khởi tạo đối tượng ServerApp đại diện cho ứng dụng phía Server
app = ServerApp()
#nếu muốn định nghĩa lại cách xử lý của server khi nhận trọng số / gửi trọng số (HOẶC TẠO FILE STRATEGY.PY VÀ IMPORT VÀO)
# class MyCustomStrategy(FedAvg):
    # def __init__(self, my_param: float = 0.5, **kwargs):
    #     super().__init__(**kwargs)
    #     self.my_param = my_param
    # def aggregate_train(self, server_round, replies):  hàm tổng hợp ở server 
    #     # Tự viết công thức tổng hợp ở Server tại đây
    #     print(f"Server đang tổng hợp với tham số: {self.my_param}")
    #     ...
    #     return new_arrays, metrics
    # def configure_train() : hàm config chuẩn bị các gói tin gửi cho client ở mỗi round
    # def select_clients(): hàm chọn client tham gia ở mỗi round
    # def configure_evaluate() : hàm config các tham số gửi cho client ở mỗi round
#rồi gọi strategy = MyCustomStrategy(...) thay vì FedProx(...) và chạy vòng lặp với strategy đó với strategy.start(params)

@app.main()
def main(grid: Grid, context: Context) -> None:
    """Hàm chính (entry point) của Server, điều phối toàn bộ quá trình học liên hiệp.

    Args:
        grid: Đối tượng quản lý mạng lưới các client/nodes tham gia học.
        context: Chứa cấu hình chạy (run_config) được đọc từ file pyproject.toml.
    """
    # 1. Đọc các siêu tham số từ mục [tool.flwr.app.config] trong pyproject.toml
    fraction_evaluate: float = context.run_config["fraction-evaluate"]  # Tỷ lệ client tham gia đánh giá
    num_rounds: int = context.run_config["num-server-rounds"]           # Tổng số vòng học liên hiệp
    lr: float = context.run_config["learning-rate"]                     # Learning rate cho client
    mu: float = context.run_config.get("mu", 0.01)                     # Hệ số phạt proximal term của FedProx

    # 2. Khởi tạo mô hình toàn cục ban đầu (Global Model)
    global_model = Net()
    
    # Đóng gói trọng số PyTorch (state_dict) sang ArrayRecord để gửi qua mạng cho các client
    arrays = ArrayRecord(global_model.state_dict())

    # 3. Khởi tạo chiến lược FedProx (Strategy)
    # - FedProx mở rộng từ FedAvg: tổng hợp trọng số theo trung bình có trọng số (weighted average).
    # - Đồng thời tự động chèn 'proximal_mu' vào cấu hình gửi xuống cho mỗi Client ở từng round.
    fraction_train: float = context.run_config.get("fraction-train", 0.5)
    strategy = FedProx(
        fraction_train=fraction_train,
        fraction_evaluate=fraction_evaluate,
        proximal_mu=mu,
    )

    # 4. Bắt đầu quy trình huấn luyện Federated Learning qua num_rounds vòng
    result = strategy.start(
        grid=grid,                                  # Mạng lưới các client
        initial_arrays=arrays,                      # Trọng số khởi tạo gửi cho round 1
        train_config=ConfigRecord({"lr": lr}),      # Cấu hình gửi xuống client (learning rate)
        num_rounds=num_rounds,                      # Số vòng huấn luyện
        evaluate_fn=global_evaluate,                # Hàm đánh giá toàn cục tại Server sau mỗi round
    )

    # 5. Lưu mô hình cuối cùng ra file nếu save-model = true trong pyproject.toml
    if context.run_config.get("save-model", False):
        print("\nĐang lưu mô hình cuối cùng ra file disk...")
        state_dict = result.arrays.to_torch_state_dict()
        torch.save(state_dict, "final_model.pt")
        print("Đã lưu thành công: final_model.pt")


def global_evaluate(server_round: int, arrays: ArrayRecord) -> MetricRecord:
    """Đánh giá mô hình toàn cục tại Server sau mỗi round trên tập dữ liệu kiểm tra tập trung.

    Args:
        server_round: Vòng học hiện tại (0 là trước khi train, 1, 2, ...).
        arrays: Trọng số mô hình toàn cục hiện tại dưới dạng ArrayRecord.

    Returns:
        MetricRecord chứa độ chính xác (accuracy) và loss toàn cục.
    """
    # Khởi tạo mô hình và nạp trọng số toàn cục vừa tổng hợp được
    model = Net()
    model.load_state_dict(arrays.to_torch_state_dict())
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model.to(device)

    # Tải tập dữ liệu kiểm tra tập trung (Centralized Test Set)
    test_dataloader = load_centralized_dataset()

    # Chạy hàm kiểm thử test() từ file task.py
    test_loss, test_acc = test(model, test_dataloader, device)

    print(f"[Server Round {server_round}] Test Acc: {test_acc:.4f} | Test Loss: {test_loss:.4f}")

    # Đóng gói kết quả trả về cho Server lưu trữ / hiển thị
    return MetricRecord({"accuracy": test_acc, "loss": test_loss})
