
import torch
from flwr.app import ArrayRecord, Context, Message, MetricRecord, RecordDict
from flwr.clientapp import ClientApp
#dữ liệu dạng tensor : đóng vào ArrayRecord
#dữ liệu dạng số: đóng vào Metric Record
# Import kiến trúc mạng, hàm load dữ liệu và các hàm train/test từ task.py
from source.task import Net, load_data
from source.task import test as test_fn    # Đổi tên thành test_fn để không trùng với hàm evaluate của ClientApp
from source.task import train as train_fn  # Đổi tên thành train_fn để không trùng với hàm train của ClientApp

app= ClientApp()

@app.train()
def train(msg : Message, context : Context):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    #lấy model từ vòng trước ra, đây là prev_model
    prev_model = None
    if "prev_model" in context.state:
        prev_model = Net()#khởi tạo mạng mới trong RAM
        #nạp trọng số từ vòng trước vào mạng prev_model và đưa lên gpu
        prev_model.load_state_dict(context.state["prev_model"].to_torch_state_dict())
        prev_model.to(device)

    #nhận trọng số từ server, model này để train
    model = Net()
    w_cur = msg.content["arrays"].to_torch_state_dict()
    model.load_state_dict(w_cur)
    model.to(device)

    #lưu lại w ban đầu ở đầu vòng để chạy fednova và tính constrast với MOON
    global_model = Net()
    global_model.load_state_dict(w_cur)
    global_model.to(device)
    global_model.eval()  # Đóng băng không cho cập nhật

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

    train_loss = train_fn(
        model=model,
        trainloader=trainloader,
        global_model=global_model,
        prev_model=prev_model,
        s_k=s_k,
        lr=lr,
        device=device,
        mu=mu,
        temperature=temperature,
        epochs=local_epochs,
    )

    #tính a_k dựa trên số bước s_k, với lr cố định 
    a_k = float (s_k * lr)

    #tính độ dời w từ đầu vòng đến cuối vòng
    w_final = model.state_dict()
    w_init = global_model.state_dict() #dùng chung w_global để tính đồ dời cho FedNova và MOON
    d_k = {}

    for key in w_final: #key là tên các layer trong model cuối cùng
        w_f = w_final[key].to(device)
        w_i = w_init[key]

        #tính độ dời trung bình trên 1 đơn vị bước
        delta_k = w_f - w_i
        d_k[key] = (delta_k/ a_k).cpu() #chuyển về cpu để gửi 
    
    #số lượng mẫu của client k 
    n_k = len(trainloader.dataset)

    #đóng gói và gửi về server 
    metrics_record = MetricRecord({
        "num-examples" :n_k , 
        "train_loss" : float(train_loss),
        "a_k" : a_k
    })

    #không cần đóng gói model 
    
    #gộp vào RecordDict và gửi đi
    content = RecordDict({
        "arrays" : ArrayRecord(d_k),
        "metrics" : metrics_record 
    })

    #cất model vòng này để dùng cho vòng sau 
    context.state["prev_model"] = ArrayRecord(w_final)

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

    



    

    