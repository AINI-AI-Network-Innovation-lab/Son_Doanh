import torch
from flwr.app import ArrayRecord, ConfigRecord, Context, MetricRecord, Message
from flwr.serverapp import Grid, ServerApp
from flwr.serverapp.strategy import FedAvg #SCAFFOLD vẫn dùng fedAvg, nhưng giờ muốn dùng fednova để chuẩn hóa lại vì fedavg ưu tiên bình phương cho các client có nhiều trọng số
from collections.abc import Iterable 
from source.task import Net, load_centralized_dataset, test

app = ServerApp()

class FedNovaScaffold(FedAvg):
    def __init__(self, num_total_clients: int,
        efficient_step_length: float | None = None, **kwargs):
        super().__init__(**kwargs)
        self.num_total_clients = num_total_clients
        self.efficient_step_length = efficient_step_length
        self.current_arrays: ArrayRecord | None = None
        self.c_global: dict[str, torch.Tensor] | None = None

    #1. CONFIGURE TRAIN: Chuẩn bị gửi trọng số mới và c_global cho Client
    def configure_train(
        self,
        server_round: int,
        arrays: ArrayRecord,
        config: ConfigRecord,
        grid: Grid,
    ) : 
        # Lưu lại trọng số hiện tại (w_cur) để dùng cho bước cập nhật ở aggregate_train
        self.current_arrays = arrays
        # Gọi hàm của FedAvg gốc để tự động chọn Client và đóng gói Message cơ bản
        messages = list(super().configure_train(server_round, arrays, config, grid))
        # Khởi tạo c_global toàn số 0 ở round 1 (nếu chưa có)
        if self.c_global is None:
            w_dict = arrays.to_torch_state_dict()
            self.c_global = {k: torch.zeros_like(v) for k, v in w_dict.items()}
        # Nhét c_global vào từng Message gửi cho các Client
        c_record = ArrayRecord(self.c_global)
        for msg in messages:
            msg.content["control_variate"] = c_record
        return messages

    #2. AGGREGATE TRAIN: Tính trọng số mới (w_next) và cập nhật c_global
    def aggregate_train(
        self, server_round: int, replies: Iterable[Message]
    ):
        # 1. Lọc các gói tin hợp lệ gửi về từ client
        valid_replies = [msg for msg in replies if msg.has_content()]
        if not valid_replies:
            return None, None
        # 2. Đọc dữ liệu từ từng Client
        client_data = []
        n_total = 0
        for msg in valid_replies:
            metrics = msg.content["metrics"]
            n_k = metrics["num-examples"]
            a_k = metrics["a_k"]
            train_loss = metrics.get("train_loss", 0.0)
            d_k = msg.content["arrays"].to_torch_state_dict()
            delta_c_k = msg.content["control_variate"].to_torch_state_dict()
            client_data.append({
                "n_k": n_k,
                "a_k": a_k,
                "d_k": d_k,
                "delta_c_k": delta_c_k,
                "train_loss": train_loss,
            })
            n_total += n_k
        # 3. Khởi tạo d_global và tổng delta_c_k
        sample_dict = client_data[0]["d_k"]
        d_global = {k: torch.zeros_like(v, dtype=torch.float32) for k, v in sample_dict.items()}
        sum_delta_c = {k: torch.zeros_like(v, dtype=torch.float32) for k, v in sample_dict.items()}
        tau_eff_auto = 0.0
        weighted_loss = 0.0
        # 4. Tính d_global = sum(p_k * d_k) và tau_eff = sum(p_k * a_k)
        for client in client_data:
            p_k = client["n_k"] / n_total
            a_k = client["a_k"]
            d_k = client["d_k"]
            delta_c = client["delta_c_k"]
            tau_eff_auto += p_k * a_k
            weighted_loss += p_k * client["train_loss"]
            for k in d_global:
                d_global[k] += p_k * d_k[k].to(torch.float32)
            for k in sum_delta_c:
                sum_delta_c[k] += delta_c[k].to(torch.float32)
        # Chọn bước nhảy hiệu dụng (FedNova)
        tau_eff = (
            self.efficient_step_length
            if self.efficient_step_length is not None
            else tau_eff_auto
        )
        # 5. Cập nhật mô hình mới: w_next = w_cur + tau_eff * d_global
        w_cur = self.current_arrays.to_torch_state_dict()
        w_next = {}
        for k in w_cur:
            w_next[k] = w_cur[k] + tau_eff * d_global[k]
        # 6. Cập nhật c_global: c = c + (1 / N) * sum(delta_c_k) (SCAFFOLD)
        for k in self.c_global:
            self.c_global[k] += sum_delta_c[k] / self.num_total_clients
        # 7. Trả về đúng 1 cặp tuple: (Mô hình mới, Chỉ số đo lường)
        new_arrays = ArrayRecord(w_next)
        new_metrics = MetricRecord({
            "train_loss": float(weighted_loss),
            "tau_eff": float(tau_eff),
        })
        return new_arrays, new_metrics

@app.main()
def main(grid: Grid, context: Context) -> None:
    """Entry point chính của ServerApp."""
    fraction_evaluate: float = context.run_config["fraction-evaluate"]
    num_rounds: int = context.run_config["num-server-rounds"]
    lr: float = context.run_config["learning-rate"]
    
    fraction_train: float = context.run_config.get("fraction-train", 0.5)
    num_total_clients: int = context.run_config.get("num-partitions", 20)
    # 1. Khởi tạo mô hình ban đầu
    global_model = Net()
    arrays = ArrayRecord(global_model.state_dict())
    # 2. Khởi tạo chiến lược tùy chỉnh
    strategy = FedNovaScaffold(
        num_total_clients=num_total_clients,
        fraction_train=fraction_train,
        fraction_evaluate=fraction_evaluate,
        efficient_step_length=None,       # None = tự động tính tau_eff theo FedNova
    )
    # 3. Chạy quá trình Federated Learning
    result = strategy.start(
        grid=grid,  #grid là hệ thống liên lạc giữa server và client
        initial_arrays=arrays,
        train_config=ConfigRecord({"lr": lr}),
        num_rounds=num_rounds,
        evaluate_fn=global_evaluate,
    )
    if context.run_config.get("save-model", False):
        state_dict = result.arrays.to_torch_state_dict()
        torch.save(state_dict, "final_model.pt")

def global_evaluate(server_round: int, arrays: ArrayRecord) -> MetricRecord:
    """Đánh giá mô hình toàn cục trên tập test tập trung sau mỗi round."""
    model = Net()
    model.load_state_dict(arrays.to_torch_state_dict())
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model.to(device)
    test_dataloader = load_centralized_dataset()
    test_loss, test_acc = test(model, test_dataloader, device)
    print(f"[Server Round {server_round}] Test Acc: {test_acc:.4f} | Test Loss: {test_loss:.4f}")
    return MetricRecord({"accuracy": float(test_acc), "loss": float(test_loss)})
