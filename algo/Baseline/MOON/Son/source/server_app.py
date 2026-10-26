import torch
from flwr.app import ArrayRecord, ConfigRecord, Context, MetricRecord, Message
from flwr.serverapp import Grid, ServerApp
from flwr.serverapp.strategy import FedAvg
from collections.abc import Iterable 
from source.task import Net, load_centralized_dataset, test

# Khởi tạo instance ServerApp cho ứng dụng Flower Server
app = ServerApp()


class FedNovaMOON(FedAvg):
    def __init__(
        self,
        num_total_clients: int,
        efficient_step_length: float | None = None,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.num_total_clients = num_total_clients
        self.efficient_step_length = efficient_step_length
        self.current_arrays: ArrayRecord | None = None  # Lưu trữ model toàn cục hiện tại trước khi huấn luyện

    # Cấu hình nội dung gửi cho client ở mỗi round: gửi model toàn cục và learning rate
    def configure_train(
        self, server_round: int, arrays: ArrayRecord, config: ConfigRecord, grid: Grid
    ) -> Iterable[Message]:
        # 1. Lưu lại mô hình toàn cục hiện tại để dùng cho tính toán ở aggregate_train
        self.current_arrays = arrays
        # 2. Gọi FedAvg đóng gói Message cơ bản gồm model + config và chọn tập client ngẫu nhiên
        messages = list(super().configure_train(server_round, arrays, config, grid))
        return messages

    def aggregate_train(
        self, server_round: int, replies: Iterable[Message]
    ) -> tuple[ArrayRecord | None, MetricRecord | None]:
        # 1. Lọc các phản hồi có nội dung hợp lệ từ client
        valid_replies = [msg for msg in replies if msg.has_content()]
        if not valid_replies or self.current_arrays is None:
            return None, None

        # 2. Đọc dữ liệu từ client và lưu tạm vào danh sách
        client_data = []  # Danh sách tạm để chứa dữ liệu các client cho đến khi tính xong tổng số mẫu n_total
        n_total = 0
        for msg in valid_replies:
            # Lấy các chỉ số vô hướng từ metrics của client
            metrics = msg.content["metrics"]
            n_k = metrics["num-examples"]
            a_k = metrics["a_k"]
            train_loss = metrics["train_loss"]
            # Giải nén tensor độ dời chuẩn hóa d_k
            d_k = msg.content["arrays"].to_torch_state_dict()

            # Cộng dồn số mẫu và lưu bản ghi của client k
            n_total += n_k
            client_data.append({
                "n_k": n_k,
                "a_k": a_k,
                "d_k": d_k,
                "train_loss": train_loss,
            })

        # 3. Khởi tạo các biến tích lũy toàn cục tại server
        # Lấy d_k của client đầu tiên làm mẫu để xác định tên các layer và kích thước tensor
        sample_dict = client_data[0]["d_k"]
        # Khởi tạo d_global chứa toàn tensor 0 có cùng cấu trúc với mô hình
        d_global = {k: torch.zeros_like(v, dtype=torch.float32) for k, v in sample_dict.items()}
        # Biến cộng dồn số bước hiệu quả toàn cục
        tau_eff_auto = 0.0
        # Biến cộng dồn loss trung bình có trọng số toàn mạng
        weighted_loss = 0.0

        # 4. Tính d_global, tau_eff và weighted_loss theo thuật toán FedNova
        for client in client_data:
            p_k = client["n_k"] / n_total  # Trọng số dữ liệu của client k
            a_k = client["a_k"]
            d_k = client["d_k"]
            tau_eff_auto += p_k * a_k  # Cộng dồn số bước hiệu quả có trọng số
            weighted_loss += p_k * client["train_loss"]  # Cộng dồn loss có trọng số

            for key in d_k:
                d_global[key] += p_k * d_k[key]  # Cộng dồn độ dời theo từng layer của mô hình

        # 5. Chọn bước nhảy hiệu dụng (FedNova)
        tau_eff = (
            self.efficient_step_length
            if self.efficient_step_length is not None
            else tau_eff_auto  # Dùng giá trị tự động tính nếu không cấu hình cố định
        )

        # 6. Cập nhật mô hình mới: w_next = w_cur + tau_eff * d_global
        w_cur = self.current_arrays.to_torch_state_dict()
        w_next = {}
        for key in w_cur:
            w_next[key] = w_cur[key] + (tau_eff * d_global[key]).to(w_cur[key].dtype)

        # 7. Đóng gói mô hình mới và chỉ số đo lường trả về cho ServerApp
        new_arrays = ArrayRecord(w_next)
        new_metrics = MetricRecord({
            "train_loss": float(weighted_loss),
            "tau_eff": float(tau_eff),
        })
        return new_arrays, new_metrics


@app.main()
def main(grid: Grid, context: Context) -> None:
   
    # 1. Đọc tỷ lệ client được chọn để đánh giá từ config (mặc định 1.0 = 100%)
    fraction_evaluate: float = context.run_config.get("fraction-evaluate", 1.0)

    # 2. Đọc tổng số vòng huấn luyện FL (mặc định 10 rounds)
    num_rounds: int = context.run_config.get("num-server-rounds", 10)

    # 3. Đọc tốc độ học learning rate (mặc định 0.01)
    lr: float = context.run_config.get("learning-rate", 0.01)

    # 4. Đọc tổng số client trong mạng mô phỏng từ số partition dữ liệu (mặc định 20)
    num_total_clients: int = context.run_config.get("num-partitions", 20)
    fraction_train: float = context.run_config.get("fraction-train", 0.5)

    # 5. Khởi tạo kiến trúc mô hình toàn cục ban đầu tại Server
    global_model = Net()    #Net được cấu hình tại task.py

    # 6. Đóng gói state_dict của mô hình thành ArrayRecord để phát cho client ở round 1
    arrays = ArrayRecord(global_model.state_dict())

    # 7. Khởi tạo chiến lược tổng hợp tùy chỉnh FedNovaMOON
    strategy = FedNovaMOON(
        num_total_clients=num_total_clients,
        fraction_train=fraction_train,
        fraction_evaluate=fraction_evaluate,
        efficient_step_length=None,            # None để tự động tính tau_eff theo dữ liệu thực tế từng round
    )

    # 8. Bắt đầu toàn bộ quy trình huấn luyện liên kết, start là bắt đầu các vòng lặp, 
    # trong mỗi round, start tự gọi configure train để lấy message và gửi cho client qua grid
    # sau khi nhận reply từ client, start sẽ gọi aggregate train để tổng hợp kết quả
    result = strategy.start(
        grid=grid,                              # Grid: hạ tầng mạng kết nối server và các client
        initial_arrays=arrays,                  # Trọng số mô hình xuất phát ban đầu
        train_config=ConfigRecord({"lr": lr}),  # Config chứa learning rate gửi cho client mỗi round
        num_rounds=num_rounds,                  # Số vòng FL cần chạy
        evaluate_fn=global_evaluate,            # Hàm đánh giá tập trung tại server sau mỗi round
    )

    # 9. Kiểm tra nếu có cờ cấu hình lưu mô hình thì lưu trọng số ra file .pt
    if context.run_config.get("save-model", False):
        state_dict = result.arrays.to_torch_state_dict()
        torch.save(state_dict, "final_model.pt")


def global_evaluate(server_round: int, arrays: ArrayRecord) -> MetricRecord:
    """Đánh giá mô hình toàn cục trên tập test tập trung sau mỗi round."""
    # 1. Khởi tạo một đối tượng mô hình mới để nạp trọng số đánh giá
    model = Net()

    # 2. Giải nén ArrayRecord từ server thành state_dict PyTorch và nạp vào mô hình
    model.load_state_dict(arrays.to_torch_state_dict())

    # 3. Chọn thiết bị chạy (ưu tiên GPU cuda nếu có, ngược lại chạy trên CPU)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model.to(device)

    # 4. Tải DataLoader của tập dữ liệu kiểm thử tập trung (testset)
    test_dataloader = load_centralized_dataset()

    # 5. Chạy hàm test để tính toán độ mất mát (loss) và độ chính xác (accuracy)
    test_loss, test_acc = test(model, test_dataloader, device)
    print(f"[Server Round {server_round}] Test Acc: {test_acc:.4f} | Test Loss: {test_loss:.4f}")

    # 6. Đóng gói kết quả thành MetricRecord trả về cho Flower ghi nhận log
    return MetricRecord({"accuracy": float(test_acc), "loss": float(test_loss)})