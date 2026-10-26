---
tags: [quickstart]
dataset: []
framework: [numpy]
---

# Federated Learning with NumPy and Flower (Demo)

This example of Flower uses a dummy `NumPy` model as well as dummy training and evaluation steps in the `ClientApp` to showcase the core functionality of Flower apps. This app does not use a dataset.

## Set up the project

### Fetch the app

Install Flower:

```shell
pip install flwr
```

## 💡 Tổng quan thuật toán (Algorithm Overview)

- FedEntropy/Performance
    - FedEntropy
        - Entropy thể hiện độ hỗn loạn/phân tán của dữ liệu giữa các lớp
            - $H_k = -\sum_{c=1}^C p_{k,c} \log{p_{k,c}}$ với $p_{k,c}$ là tỷ lệ mẫu thuộc lớp $c$ tại client $k$
            - Entropy càng cao thì càng cân bằng
            - $H_k^{norm} = \frac{H_k}{\log{C}}$ với $C$ là tổng số lớp của bài toán
            - $H_k^{norm}$ gần 0 thì phân phối mất cân bằng, gần 1 thì đồng đều giữa các lớp
        - Cập nhật trọng số server side:
            - $\alpha_k^{\text{Entropy}} = \frac{H_k^{\text{norm}} n_k}{\sum_{j=1}^K H_j^{\text{norm}} n_j}$: Cân bằng giữa độ phân tán dữ liệu và số mẫu của client
            - $w_{t+1} = \sum_{k=1}^K \alpha_k^{\text{Entropy}} w_{k, t+1}$
    - FedPerformance: Giải quyết vấn đề: entropy giống nhau có thể cho ra chất lượng sai lệch do chất lượng của mẫu (dữ liệu phân phối đều nhưng có ảnh nhiễu, có ảnh nét)
        - Sử dụng training loss $\mathcal{L}_k$ để đánh giá hiệu suất
        - $P_k = 1 - \frac{\mathcal{L}_k}{\sum_{j=1}^K \mathcal{L}_j + \epsilon}$
            - $\mathcal{L}_k$: training loss của client $k$ sau local training
            - $\epsilon$: hằng số để tránh chia cho 0
            - $P_k$: Điểm hiệu suất tương đối của client
            - Lấy $1 - \text{biểu thức}$ để khi loss nhỏ thì điểm hiệu suất sẽ cao hơn
        - Công thức tổng hợp trọng số ở server: $\alpha_k^{\text{Performance}} = \frac{P_k n_k}{\sum_j P_j n_j}$
        - Vấn đề: Hiệu suất loss ở local chỉ đánh giá trên tập dữ liệu của client, do đó với bài toán non-iid không hữu dụng lắm: thực tế chỉ ra nó ngang ngửa với fedavg :))
    - FedEntropyPerformance
        - Hybrid mode:
        - $\alpha_k^{E+P} = \frac{P_k H_k^{\text{norm}} n_k}{\sum_j P_j H_j^{\text{norm}} n_j}$
        - Thực tế: Chỉ lệch 0.3% so với FedEntropy
    - Kết luận: Nên dùng FedEntropy vì hiệu quả đủ tốt, không mất thêm chi phí truyền thông(không phải gửi L, chỉ gửi entropy lên sever).
Fetch the app:

```shell
flwr new @flwrlabs/demo
```

```shell
demo
├── quickstart_numpy
│   ├── __init__.py
│   ├── client_app.py   # Defines your ClientApp
│   ├── server_app.py   # Defines your ServerApp
│   └── task.py         # Defines model creation
├── pyproject.toml      # Project metadata like dependencies and configs
└── README.md
```

### Install dependencies and project

Install the dependencies defined in `pyproject.toml` as well as the `quickstart_numpy` package.

```bash
pip install -e .
```

> **Tip:** Your `pyproject.toml` file can define more than just the dependencies of your Flower app. You can also use it to specify hyperparameters for your runs and control which Flower Runtime is used. By default, it uses the Simulation Runtime, but you can switch to the Deployment Runtime when needed.
> Learn more in the [TOML configuration guide](https://flower.ai/docs/framework/how-to-configure-pyproject-toml.html).

## Run with the Simulation Engine

In the `quickstart-numpy` directory, use `flwr run` to run a local simulation:

```bash
flwr run .
```

Refer to the [How to Run Simulations](https://flower.ai/docs/framework/how-to-run-simulations.html) guide in the documentation for advice on how to optimize your simulations.

## Run with the Deployment Engine

Follow this [how-to guide](https://flower.ai/docs/framework/how-to-run-flower-with-deployment-engine.html) to run the same app in this example but with Flower's Deployment Engine. After that, you might be interested in setting up [secure TLS-enabled communications](https://flower.ai/docs/framework/how-to-enable-tls-connections.html) and [SuperNode authentication](https://flower.ai/docs/framework/how-to-authenticate-supernodes.html) in your federation.

You can run Flower on Docker too! Check out the [Flower with Docker](https://flower.ai/docs/framework/docker/index.html) documentation.

## Resources

- Flower website: [flower.ai](https://flower.ai/)
- Check the documentation: [flower.ai/docs](https://flower.ai/docs/)
- Give Flower a ⭐️ on GitHub: [GitHub](https://github.com/adap/flower)
- Join the Flower community!
  - [Flower Slack](https://flower.ai/join-slack/)
  - [Flower Discuss](https://discuss.flower.ai/)
