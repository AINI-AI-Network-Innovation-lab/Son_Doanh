import torch
from flwr.app import ArrayRecord, ConfigRecord, Context, MetricRecord, Message
from flwr.serverapp import Grid, ServerApp
from flwr.serverapp.strategy import FedAvg
from collections.abc import Iterable 
from source.task import Net, load_centralized_dataset, test
import math 
# Khởi tạo instance ServerApp cho ứng dụng Flower Server
app = ServerApp()

class FedAdp(FedAvg):
    def __init__(
        self, 
        decrease_speed : int | None = 1, 
        **kwargs, 
    ): 
        super().__init__(**kwargs) 
        self.decrease_speed = decrease_speed 
        self.current_arrays: ArrayRecord | None = None #lưu trữ model toàn cục hiện tại
        self.smooth_angles: dict[int | str, float] = {} #khởi tạo dict lưu trữ id client và góc làm mượt cũ
        self.logits : dict[int | str, float] = {}   #khởi tạo dict logits
    #nội dung gửi cho client
    # Cấu hình nội dung gửi cho client ở mỗi round: gửi model toàn cục và learning rate
    def configure_train(
        self, server_round: int, arrays: ArrayRecord, config: ConfigRecord, grid: Grid
    ) -> Iterable[Message]:
        # 1. Lưu lại mô hình toàn cục hiện tại để dùng cho tính toán ở aggregate_train
        self.current_arrays = arrays
        # 2. Gọi FedAvg đóng gói Message cơ bản gồm model + config và chọn tập client ngẫu nhiên
        messages = list(super().configure_train(server_round, arrays, config, grid))
        return messages
    
    #cấu hình train tại server
    def aggregate_train(self, server_round : int, replies : Iterable[Message]): #nhận vào danh sách các reply message từ client
        #lọc phản hồi
        valid_replies = [msg for msg in replies if msg.has_content()]
        if not valid_replies or self.current_arrays is None : 
            return None, None
        client_data = []
        n_total = 0
        
        #đọc reply
        for msg in valid_replies : 
            metrics = msg.content["metrics"]
            lr = metrics["lr"]
            train_loss = metrics["train_loss"]
            num_examples = metrics["num_examples"]
            delta_w = msg.content["arrays"].to_torch_state_dict() 
            
            # 1. Nối tất cả các layer thành 1 vector 1D duy nhất để tính góc cho dễ
            delta_vec = torch.cat([v.view(-1) for v in delta_w.values()])
            # 2. Chia trực tiếp vector 1D cho số float
            grad_vec = -delta_vec / lr  # Hoặc -delta_vec / lr tùy công thức gradient của bạn
            n_total+=num_examples

            client_data.append({
                "id": msg.metadata.src_node_id,
                "delta_vec" : delta_vec,
                "num_examples" : num_examples,
                "train_loss" : train_loss,
                "grad_vec" : grad_vec,
                "delta_w" : delta_w
            })
        grad_vec_global = torch.zeros_like(client_data[0]["grad_vec"])
        #tính gradient losss toàn cục
        for client in client_data: 
            grad_vec_global += (client["num_examples"] / n_total) * client["grad_vec"]
        
        #tính góc và góc làm mượt
        for client in client_data : 
            id = client["id"]
            grad_vec = client["grad_vec"]
            #tính góc
            # 1. Tính tích vô hướng (dot product) và norm (độ dài vector)
            dot_prod = torch.dot(grad_vec, grad_vec_global)
            norm_local = torch.norm(grad_vec)
            norm_global = torch.norm(grad_vec_global)
            # 2. Tính cosin của góc (thêm 1e-8 ở mẫu để tránh chia cho 0)
            cos_val = dot_prod / (norm_local * norm_global + 1e-8)
            # 3. KẸP giá trị trong đoạn [-1.0, 1.0] (RẤT QUAN TRỌNG: để tránh sai số float gây ra lỗi NaN ở hàm acos)
            cos_val = torch.clamp(cos_val, -1.0, 1.0)
            # 4. Tính góc theta (đơn vị: radian, giá trị từ 0 đến pi)
            angle = torch.acos(cos_val).item() # .item() để lấy ra số float thuần

           # Cập nhật góc làm mượt an toàn:
            if id not in self.smooth_angles:
                self.smooth_angles[id] = angle
            else:
                self.smooth_angles[id] = (
                    self.smooth_angles[id] * (server_round - 1) + angle
                ) / server_round
            theta_smooth = self.smooth_angles[id]
            # Tính điểm hàm f:
            alpha = self.decrease_speed
            inner_exp = math.exp(-alpha * (theta_smooth - 1.0))
            score_f = alpha * (1.0 - math.exp(-inner_exp))
            
            self.logits[id] = score_f
            client["score_f"] = score_f
        
        #tính softmax logit
        softmax_logits : dict[int, float] = {}
        total_exp = 0.0
        #tính tổng mẫu
        for client in client_data: 
            total_exp += math.exp(client["score_f"])
        
        for client in client_data:
            id = client["id"]
            softmax_logits[id] = math.exp(client["score_f"]) / total_exp
        
        #tính model mới 
        current_w = self.current_arrays.to_torch_state_dict()
        new_w = {k : v.clone() for k, v in current_w.items()}

        #cộng dồn độ dời 
        for client in client_data:
            c_id = client["id"]
            p_k = softmax_logits[c_id]
            delta_dict = client["delta_w"]
            for key in new_w:
                new_w[key] += p_k * delta_dict[key]
        # 3. Tính loss trung bình để server theo dõi
        avg_loss = sum(client["train_loss"] * (client["num_examples"] / n_total) for client in client_data)
        # 4. Trả về cho ServerApp
        return ArrayRecord(new_w), MetricRecord({"train_loss": float(avg_loss)})

@app.main()
def main(grid: Grid, context: Context) -> None:
    # 1. Đọc các tham số cấu hình từ pyproject.toml
    num_rounds: int = context.run_config.get("num-server-rounds", 10)
    lr: float = context.run_config.get("learning-rate", 0.01)
    decrease_speed: float = context.run_config.get("decrease-speed", 1.0)
    fraction_train: float = context.run_config.get("fraction-train", 1.0)
    fraction_evaluate: float = context.run_config.get("fraction-evaluate", 1.0)

    # 2. Khởi tạo mô hình toàn cục ban đầu tại Server
    global_model = Net()
    arrays = ArrayRecord(global_model.state_dict())

    # 3. Khởi tạo Strategy FedAdp
    strategy = FedAdp(
        decrease_speed=decrease_speed,
        fraction_train=fraction_train,
        fraction_evaluate=fraction_evaluate,
    )

    # 4. Bắt đầu toàn bộ quy trình huấn luyện liên kết
    result = strategy.start(
        grid=grid,
        initial_arrays=arrays,
        train_config=ConfigRecord({"lr": lr}),
        num_rounds=num_rounds,
        evaluate_fn=global_evaluate, # Đánh giá tập trung tại Server sau mỗi round
    )

    # 5. Lưu model nếu có cấu hình save-model
    if context.run_config.get("save-model", False):
        state_dict = result.arrays.to_torch_state_dict()
        torch.save(state_dict, "final_model.pt")


def global_evaluate(server_round: int, arrays: ArrayRecord) -> MetricRecord:
    """Đánh giá mô hình toàn cục trên tập testset tập trung sau mỗi round."""
    model = Net()
    model.load_state_dict(arrays.to_torch_state_dict())
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model.to(device)

    # Tải tập test tập trung từ task.py
    test_dataloader = load_centralized_dataset()
    test_loss, test_acc = test(model, test_dataloader, device)

    return MetricRecord({"accuracy": float(test_acc), "loss": float(test_loss)})
            
                

            

            

            