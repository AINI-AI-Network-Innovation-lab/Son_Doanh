"""custom_strategy: FedDisco strategy combined with MOON config."""

from typing import Iterable, Optional
import torch
from flwr.app import ArrayRecord, ConfigRecord, Message, MetricRecord
from flwr.serverapp import Grid
from flwr.serverapp.strategy import FedAvg


class FedDisco(FedAvg):
    """Chiến lược FedDisco kết hợp cấu hình siêu tham số cho MOON."""

    def __init__(
        self,
        moon_mu: float = 1.0,
        temperature: float = 0.5,
        disco_a: float = 0.5,
        disco_b: float = 0.1,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.moon_mu = moon_mu
        self.temperature = temperature
        self.disco_a = disco_a
        self.disco_b = disco_b

    def configure_train(
        self,
        server_round: int,
        arrays: ArrayRecord,
        config: ConfigRecord,
        grid: Grid,
    ) -> Iterable[Message]:
        """Gắn thêm siêu tham số MOON vào config trước khi gửi tới các Client."""
        config["moon-mu"] = self.moon_mu
        config["temperature"] = self.temperature
        return super().configure_train(server_round, arrays, config, grid)

    def aggregate_train(
        self,
        server_round: int,
        replies: Iterable[Message],
    ) -> tuple[Optional[ArrayRecord], Optional[MetricRecord]]:
        """Tổng hợp mô hình sử dụng trọng số Discrepancy-aware (FedDisco)."""
        replies_list = list(replies)
        valid_replies = [msg for msg in replies_list if not msg.has_error()]
        if not valid_replies:
            return None, None

        # 1. Thu thập số lượng mẫu (n_k), độ lệch (d_k) và bộ trọng số của từng Client
        num_examples_list = []
        discrepancies = []
        state_dicts = []

        for msg in valid_replies:
            metrics = msg.content["metrics"]
            arrays = msg.content["arrays"]

            num_examples_list.append(int(metrics["num-examples"]))
            discrepancies.append(float(metrics["discrepancy"]))
            state_dicts.append(arrays.to_torch_state_dict())

        total_examples = sum(num_examples_list)

        # 2. Tính trọng số FedDisco: ReLU(n_k / N - a * d_k + b)
        raw_weights = []
        for n_k, d_k in zip(num_examples_list, discrepancies):
            base_weight = n_k / total_examples
            disco_weight = max(0.0, base_weight - self.disco_a * d_k + self.disco_b)
            raw_weights.append(disco_weight)

        # Dự phòng trường hợp tất cả trọng số bị cắt về 0
        sum_weights = sum(raw_weights)
        if sum_weights == 0.0:
            normalized_weights = [n_k / total_examples for n_k in num_examples_list]
        else:
            normalized_weights = [w / sum_weights for w in raw_weights]

        # 3. Cộng có trọng số các tensor trong state_dict
        aggregated_state_dict = {}
        reference_keys = state_dicts[0].keys()

        for key in reference_keys:
            first_tensor = state_dicts[0][key]
            # Với các biến đếm nguyên (ví dụ: num_batches_tracked của BatchNorm)
            if not first_tensor.is_floating_point():
                aggregated_state_dict[key] = first_tensor.clone()
            else:
                agg_tensor = torch.zeros_like(first_tensor, dtype=torch.float32)
                for weight, sd in zip(normalized_weights, state_dicts):
                    agg_tensor += weight * sd[key].float()
                aggregated_state_dict[key] = agg_tensor.to(first_tensor.dtype)

        # 4. Tính trung bình có trọng số cho train_loss để log trên Server
        avg_loss = sum(
            w * float(msg.content["metrics"]["train_loss"])
            for w, msg in zip(normalized_weights, valid_replies)
        )

        return (
            ArrayRecord(aggregated_state_dict),
            MetricRecord({"train_loss": avg_loss}),
        )