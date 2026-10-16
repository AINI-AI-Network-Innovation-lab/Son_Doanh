"""pytorchexample: A Flower / PyTorch app."""

import torch
from flwr.app import ArrayRecord, ConfigRecord, Context, MetricRecord
from flwr.serverapp import Grid, ServerApp

from custom_strategy import FedDisco
from task import Net, load_centralized_dataset, test

app = ServerApp()


@app.main()
def main(grid: Grid, context: Context) -> None:
    """Main entry point for the ServerApp."""

    # Read run config safely
    fraction_evaluate: float = context.run_config.get("fraction-evaluate", 0.5)
    num_rounds: int = context.run_config.get("num-server-rounds", 3)
    lr: float = context.run_config.get("learning-rate", 0.01)

    # Đọc siêu tham số của MOON và FedDisco từ pyproject.toml
    moon_mu: float = context.run_config.get("moon-mu", 1.0)
    temperature: float = context.run_config.get("temperature", 0.5)
    disco_a: float = context.run_config.get("disco-a", 0.5)
    disco_b: float = context.run_config.get("disco-b", 0.1)

    # Load global model
    global_model = Net()
    arrays = ArrayRecord(global_model.state_dict())

    # Initialize FedDisco + MOON strategy
    strategy = FedDisco(
        fraction_evaluate=fraction_evaluate,
        moon_mu=moon_mu,
        temperature=temperature,
        disco_a=disco_a,
        disco_b=disco_b,
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