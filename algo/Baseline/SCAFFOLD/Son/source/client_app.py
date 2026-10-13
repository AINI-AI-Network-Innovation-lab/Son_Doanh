import torch
from flwr.app import ArrayRecord, Context, Message, MetricRecord, RecordDict
from flwr.clientapp import ClientApp
#dữ liệu dạng tensor : đóng vào ArrayRecord
#dữ liệu dạng số: đóng vào Metric Record
# Import kiến trúc mạng, hàm load dữ liệu và các hàm train/test từ task.py
from source.task import Net, load_data
from source.task import test as test_fn    # Đổi tên thành test_fn để không trùng với hàm evaluate của ClientApp
from source.task import train as train_fn  # Đổi tên thành train_fn để không trùng với hàm train của ClientApp

# Khởi tạo đối tượng ClientApp đại diện cho ứng dụng chạy trên mỗi Client/Node
app = ClientApp()

@app.train()
def train(msg: Message, context: Context):
    # 1. Khởi tạo mô hình và nạp trọng số toàn cục ban đầu vào client 
    #nhận trọng số từ server thông qua Message -> ArrayRecord
    model = Net()
    w_cur = msg.content["arrays"].to_torch_state_dict()
    model.load_state_dict(w_cur)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model.to(device)
    #lưu lại w_cur để tính độ dời sau khi train
    w_init = {k: v.clone().to(device) for k, v in w_cur.items()}

    #lấy c_k lưu lại tại cuối vòng trước và c_global nhận đầu vòng
    # Nếu là vòng đầu tiên client này được chọn, khởi tạo c_k toàn số 0
    if "control_variate" in context.state:
        c_k = {k: v.to(device) for k, v in context.state["control_variate"].to_torch_state_dict().items()}
    else:
        c_k = {k: torch.zeros_like(v, device=device) for k, v in w_init.items()}
    # --- LẤY c TOÀN CỤC TỪ SERVER GỬI XUỐNG ---
    if "control_variate" in msg.content:
        c_global = {k: v.to(device) for k, v in msg.content["control_variate"].to_torch_state_dict().items()}
    else:
        c_global = {k: torch.zeros_like(v, device=device) for k, v in w_init.items()}

    # LOAD DỮ LIỆU
    partition_id = context.node_config["partition-id"]
    num_partitions = context.node_config["num-partitions"]
    batch_size = context.run_config["batch-size"]
    trainloader, _ = load_data(partition_id, num_partitions, batch_size)

    lr = msg.content["config"]["lr"]
    local_epochs = context.run_config["local-epochs"]

    #  TÍNH VÀ BỐC RANDOM SỐ BƯỚC s_k CHO CLIENT NÀY:
    max_steps = len(trainloader) * local_epochs  # Tổng số batch tối đa (số mẫu / batch_size * epochs)
    
    if context.run_config.get("random-steps", False):
        import random
        min_ratio = context.run_config.get("min-steps-ratio", 0.2)
        min_steps = max(1, int(max_steps * min_ratio))
        # Bốc ngẫu nhiên số bước thực tế client này chạy:
        s_k = random.randint(min_steps, max_steps)
    else:
        s_k = max_steps

    # 3. HUẤN LUYỆN LOCAL VỚI SỐ BƯỚC s_k
    train_loss = train_fn(model, trainloader, s_k, lr, device, c_k, c_global)

    # 4. TÍNH a_k DỰA TRÊN SỐ BƯỚC s_k THỰC TẾ ĐÃ CHẠY:
    a_k = float(s_k * lr)


    # tính độ dời và c_k ở cuối vòng giao tiếp và tính delta c_k (độ lệch của c_k của client so với vòng giao tiếp trước)
    #thay vì phải gửi cả c_k+1 thì ta chỉ cần gửi delta_c_k = c_k+1-c_global
    #như vậy server sẽ k phải lưu trữ tất cả để tính trung bình cho ra c_global mới, mà chỉ cần 
    # c_global mới = c_global cũ + average(delta_c_k)
    # Delta_k = w_final - w_init
    # d_k = Delta_k / a_k (độ dời trung bình trên 1 bước)
    w_final = model.state_dict()
    d_k = {}
    c_k_plus = {}
    delta_c_k = {}

    for key in w_final:
        w_f = w_final[key].to(device)
        w_i = w_init[key]

        # 1. Tính d_k = (w_final - w_init) / a_k (cho FedNova)
        delta_k = w_f - w_i
        d_k[key] = (delta_k / a_k).cpu()
    
        # 2. Tính c_k+ theo công thức: c_k+ = c_k - c + (w_init - w_final) / a_k
        c_k_new = c_k[key] - c_global[key] + (w_i - w_f) / a_k
        c_k_plus[key] = c_k_new.cpu()

        # 3. Tính delta_c_k gửi về server: delta_c_k = c_k+ - c_k
        delta_c_k[key] = (c_k_new - c_k[key]).cpu()
        

    # 4. LƯU c_k+ VÀO CONTEXT để dùng cho vòng sau
    # Lưu c_k+ vào context để dùng cho vòng giao tiếp tiếp theo
    context.state["control_variate"] = ArrayRecord(c_k_plus)

    # Số lượng mẫu của client k (n_k)
    n_k = len(trainloader.dataset)

    # Đóng gói gửi về Server
    # Các thông số vô hướng -> đóng vào MetricRecord
    metrics_record = MetricRecord({
        "num-examples": n_k,
        "a_k": a_k,
        "train_loss": float(train_loss),
    })

    # Gộp vào RecordDict và gửi đi
    content = RecordDict({
        "arrays": ArrayRecord(d_k),       # Server sẽ đọc d_k ở đây
        "metrics": metrics_record,  # Server sẽ đọc n_k và a_k ở đây
        "control_variate": ArrayRecord(delta_c_k), # Gửi delta_c_k về server để tính c_global mới
    })

    return Message(content=content, reply_to=msg)

@app.evaluate()
def evaluate(msg: Message, context: Context):
   

    # 1. Khởi tạo mô hình và nạp trọng số toàn cục nhận được từ Server
    model = Net()
    model.load_state_dict(msg.content["arrays"].to_torch_state_dict())
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model.to(device)

    # 2. Tải tập dữ liệu kiểm thử (valloader / testloader) của Client này
    partition_id = context.node_config["partition-id"]
    num_partitions = context.node_config["num-partitions"]
    batch_size = context.run_config["batch-size"]
    _, valloader = load_data(partition_id, num_partitions, batch_size)

    # 3. Chạy hàm kiểm thử test_fn (từ file task.py)
    eval_loss, eval_acc = test_fn(
        model,
        valloader,
        device,
    )

    # 4. Đóng gói kết quả đánh giá vào MetricRecord gửi về Server
    metrics = {
        "eval_loss": float(eval_loss),
        "eval_acc": float(eval_acc),
        "num-examples": len(valloader.dataset),
    }
    metric_record = MetricRecord(metrics)
    
    # Evaluate không thay đổi trọng số, nên chỉ cần gửi kết quả metrics
    content = RecordDict({"metrics": metric_record})
    return Message(content=content, reply_to=msg)
