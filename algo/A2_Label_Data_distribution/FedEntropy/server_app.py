"""pytorchexample: A Flower / PyTorch app."""

import torch
from flwr.app import ArrayRecord, ConfigRecord, Context, MetricRecord
from flwr.serverapp import Grid, ServerApp

from custom_strategy import ScaffoldFedEntropy
from task import Net, get_zero_control_variate, load_centralized_dataset, test

app = ServerApp()


@app.main()
def main(grid: Grid, context: Context) -> None:
    """Main entry point for the ServerApp."""

    # Read run config safely
    fraction_evaluate: float = context.run_config.get("fraction-evaluate", 0.5)
    num_rounds: int = context.run_config.get("num-server-rounds", 3)
    lr: float = context.run_config.get("learning-rate", 0.01)

    # Đọc siêu tham số của SCAFFOLD và FedEntropy
    server_lr: float = context.run_config.get("server-lr", 1.0)
    entropy_gamma: float = context.run_config.get("entropy-gamma", 1.0)
    use_mej_filter: bool = context.run_config.get("use-mej-filter", True)

    # Khởi tạo mô hình toàn cục và biến kiểm soát server_c ban đầu
    global_model = Net()
    arrays = ArrayRecord(global_model.state_dict())
    initial_server_c = ArrayRecord(get_zero_control_variate(global_model))

    # Khởi tạo chiến lược kết hợp SCAFFOLD + FedEntropy
    strategy = ScaffoldFedEntropy(
        initial_server_c=initial_server_c,
        fraction_evaluate=fraction_evaluate,
        server_lr=server_lr,
        entropy_gamma=entropy_gamma,
        use_mej_filter=use_mej_filter,
    )

    # Start strategy
    result = strategy.start(
        grid=grid,
        initial_arrays=arrays,
        train_config=ConfigRecord({"lr": lr}),
        num_rounds=num_rounds,
        evaluate_fn=global_evaluate,
    )

    if context.run_config.get("save-model", False):
        print("\nSaving final model to disk...")
        state_dict = result.arrays.to_torch_state_dict()
        torch.save(state_dict, "final_model.pt")


def global_evaluate(server_round: int, arrays: ArrayRecord) -> MetricRecord:
    """Evaluate model on central data."""
    model = Net()
    model.load_state_dict(arrays.to_torch_state_dict())
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model.to(device)

    test_dataloader = load_centralized_dataset()
    test_loss, test_acc = test(model, test_dataloader, device)

    return MetricRecord({"accuracy": test_acc, "loss": test_loss})