import torch
from flwr.app import ArrayRecord, Context, Message, MetricRecord, RecordDict
from flwr.clientapp import ClientApp
#dữ liệu dạng tensor : đóng vào ArrayRecord
#dữ liệu dạng số: đóng vào Metric Record
# Import kiến trúc mạng, hàm load dữ liệu và các hàm train/test từ task.py
from source.task import Net, load_data
from source.task import test as test_fn    # Đổi tên thành test_fn để không trùng với hàm evaluate của ClientApp
from source.task import train as train_fn  # Đổi tên thành train_fn để không trùng với hàm train của ClientApp

app = ClientApp()

@app.train()
def train( msg: Message, context: Context) -> Message : 
    model = Net()
    w_cur = msg.content["arrays"].to_torch_state_dict()  #lấy bộ trọng số được phát 
    model.load_state_dict(w_cur) #nạp trọng số vào model
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model.to(device)
    w_init = {k : v.clone() for k, v in w_cur.items()} #lưu trọng số đầu vòng để tính độ dời

    #load dữ liệu
    partition_id = context.node_config["partition-id"]
    num_partitions = context.node_config["num-partitions"]
    batch_size = context.run_config["batch-size"]
    trainloader, _ = load_data(partition_id, num_partitions, batch_size)
    
    # Lấy hyperparams
    lr = msg.content["config"]["lr"]
    local_epochs = context.run_config["local-epochs"]
    mu = context.run_config.get("mu", 1.0)
    temperature = context.run_config.get("temperature", 0.5)

    # TÍNH VÀ BỐC RANDOM SỐ BƯỚC s_k CHO CLIENT NÀY:
    max_steps = len(trainloader) * local_epochs  # Tổng số batch tối đa (số mẫu / batch_size * epochs)
    
    if context.run_config.get("random-steps", False):
        import random
        min_ratio = context.run_config.get("min-steps-ratio", 0.2)
        min_steps = max(1, int(max_steps * min_ratio))
        # Bốc ngẫu nhiên số bước thực tế client này chạy:
        s_k = random.randint(min_steps, max_steps)
    else:
        s_k = max_steps

    #train dữ liệu tại client, train cơ bản không có gì đặc biệt
    train_loss = train_fn(model = model , trainloader= trainloader, s_k = s_k , lr = lr, device = device, epochs = local_epochs)

    #lấy model sau khi train 
    w_final = model.state_dict()
    delta_w = {}
    #tính độ dời để gửi cho server, ko dùng - mà phải for key
    for key in w_final : 
        w_f = w_final[key].cpu()
        w_i = w_init[key].cpu()
        d_w = w_f - w_i 
        delta_w[key] = d_w
    
    #số dữ liệu tại client
    n_k = len(trainloader.dataset)    

    #đóng gói metrics
    metrics_record = MetricRecord({
        "train_loss" : float(train_loss), 
        "lr" : float(lr), 
        "num_examples" : n_k 
    })

    #cho vào record dict để gửi
    content = RecordDict({
        "arrays" : ArrayRecord(delta_w),
        "metrics" : metrics_record
    })

    return Message(content = content, reply_to=msg)

@app.evaluate()
def evaluate(msg: Message, context: Context):
    model = Net()
    model.load_state_dict(msg.content["arrays"].to_torch_state_dict())
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model.to(device)

    partition_id = context.node_config["partition-id"]
    num_partitions = context.node_config["num-partitions"]
    batch_size = context.run_config["batch-size"]
    _, valloader = load_data(partition_id, num_partitions, batch_size)

    eval_loss, eval_acc = test_fn(model, valloader, device)

    metrics = {
        "eval_loss": float(eval_loss),
        "eval_acc": float(eval_acc),
        "num-examples": len(valloader.dataset),
    }
    return Message(content=RecordDict({"metrics": MetricRecord(metrics)}), reply_to=msg)


    

