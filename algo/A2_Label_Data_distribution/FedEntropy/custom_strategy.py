"""custom_strategy: Combination of SCAFFOLD and FedEntropy."""

from typing import Iterable, Optional
import numpy as np
import torch
from flwr.app import ArrayRecord, ConfigRecord, Message, MetricRecord
from flwr.serverapp import Grid
from flwr.serverapp.strategy import FedAvg


class ScaffoldFedEntropy(FedAvg):
    """Chiến lược kết hợp biến kiểm soát SCAFFOLD và lọc/trọng số FedEntropy."""

    def __init__(
        self,
        initial_server_c: ArrayRecord,
        server_lr: float = 1.0,
        entropy_gamma: float = 1.0,
        use_mej_filter: bool = True,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.server_c = initial_server_c
        self.server_lr = server_lr
        self.entropy_gamma = entropy_gamma
        self.use_mej_filter = use_mej_filter
        self.current_global_state_dict: Optional[dict[str, torch.Tensor]] = None

    @staticmethod
    def _shannon_entropy(dist: np.ndarray, eps: float = 1e-12) -> float:
        """Tính Shannon Entropy của một vector phân phối xác suất."""
        clipped = np.clip(dist, eps, 1.0)
        return float(-np.sum(clipped * np.log(clipped)))

    def _maximum_entropy_judgment(
        self, soft_labels: list[np.ndarray], num_examples: list[int]
    ) -> list[int]:
        """Thuật toán Maximum Entropy Judgment (MEJ) của FedEntropy."""
        active_indices = list(range(len(soft_labels)))
        if not self.use_mej_filter or len(active_indices) <= 2:
            return active_indices

        def group_entropy(indices: list[int]) -> float:
            total_n = sum(num_examples[i] for i in indices)
            agg_dist = (
                sum(num_examples[i] * soft_labels[i] for i in indices) / total_n
            )
            return self._shannon_entropy(agg_dist)

        while len(active_indices) > 2:
            current_ent = group_entropy(active_indices)
            best_ent = current_ent
            worst_idx = None

            for idx in active_indices:
                candidate = [i for i in active_indices if i != idx]
                cand_ent = group_entropy(candidate)
                if cand_ent > best_ent:
                    best_ent = cand_ent
                    worst_idx = idx

            if worst_idx is not None:
                active_indices.remove(worst_idx)
            else:
                break

        return active_indices

    def configure_train(
        self,
        server_round: int,
        arrays: ArrayRecord,
        config: ConfigRecord,
        grid: Grid,
    ) -> Iterable[Message]:
        """Lưu mô hình global hiện tại và đính kèm biến kiểm soát server_c xuống Client."""
        self.current_global_state_dict = arrays.to_torch_state_dict()
        messages = list(super().configure_train(server_round, arrays, config, grid))
        for msg in messages:
            msg.content["server_c"] = self.server_c
        return messages

    def aggregate_train(
        self,
        server_round: int,
        replies: Iterable[Message],
    ) -> tuple[Optional[ArrayRecord], Optional[MetricRecord]]:
        """Lọc Client bằng FedEntropy và cập nhật (x, c) theo SCAFFOLD."""
        valid_replies = [msg for msg in replies if not msg.has_error()]
        if not valid_replies:
            return None, None

        num_received = len(valid_replies)
        num_examples_list = []
        soft_labels_list = []
        local_entropies = []
        state_dicts = []
        delta_c_dicts = []

        for msg in valid_replies:
            metrics = msg.content["metrics"]
            num_examples_list.append(int(metrics["num-examples"]))

            sl = np.array(metrics["soft_labels"], dtype=np.float64)
            soft_labels_list.append(sl)
            local_entropies.append(float(metrics["local_entropy"]))

            state_dicts.append(msg.content["arrays"].to_torch_state_dict())
            delta_c_dicts.append(msg.content["delta_c"].to_torch_state_dict())

        # 1. FedEntropy: Chọn tập Client tối đa hóa Entropy (MEJ)
        selected_indices = self._maximum_entropy_judgment(
            soft_labels_list, num_examples_list
        )

        # 2. Tính trọng số tổng hợp mô hình theo FedEntropy
        raw_weights = []
        for idx in selected_indices:
            n_k = num_examples_list[idx]
            ent_k = max(local_entropies[idx], 1e-6)
            weight_k = n_k * (ent_k ** self.entropy_gamma)
            raw_weights.append(weight_k)

        total_weight = sum(raw_weights)
        norm_weights = [w / total_weight for w in raw_weights]

        # 3. SCAFFOLD: Cập nhật tham số mô hình toàn cục
        new_global_state_dict = {}
        reference_keys = state_dicts[0].keys()
        server_c_dict = self.server_c.to_torch_state_dict()

        for key in reference_keys:
            ref_tensor = state_dicts[0][key]
            if not ref_tensor.is_floating_point():
                new_global_state_dict[key] = state_dicts[selected_indices[0]][key].clone()
            elif key not in server_c_dict:
                # Với các buffer của BatchNorm (running_mean, running_var): lấy trung bình có trọng số trực tiếp
                agg_buffer = torch.zeros_like(ref_tensor, dtype=torch.float32)
                for w, idx in zip(norm_weights, selected_indices):
                    agg_buffer += w * state_dicts[idx][key].float()
                new_global_state_dict[key] = agg_buffer.to(ref_tensor.dtype)
            else:
                # Với các tham số khả huấn luyện: x^(t+1) = x^(t) + server_lr * sum(w_k * (y_k - x^(t)))
                x_old = self.current_global_state_dict[key].float()
                delta_y = torch.zeros_like(x_old, dtype=torch.float32)
                for w, idx in zip(norm_weights, selected_indices):
                    y_k = state_dicts[idx][key].float()
                    delta_y += w * (y_k - x_old)

                x_new = x_old + self.server_lr * delta_y
                new_global_state_dict[key] = x_new.to(ref_tensor.dtype)

        # 4. SCAFFOLD: Cập nhật biến kiểm soát toàn cục c^(t+1) = c^(t) + (1 / N) * sum_{k in A} delta_c_k
        # Dùng hệ số đều 1/num_received để đảm bảo c_k cũ triệt tiêu hoàn toàn qua các vòng
        inv_n = 1.0 / max(num_received, 1)
        for key in server_c_dict.keys():
            sum_delta_c = torch.zeros_like(server_c_dict[key], dtype=torch.float32)
            for idx in selected_indices:
                sum_delta_c += delta_c_dicts[idx][key].float()
            server_c_dict[key] = (
                server_c_dict[key].float() + inv_n * sum_delta_c
            ).to(server_c_dict[key].dtype)

        self.server_c = ArrayRecord(server_c_dict)

        # 5. Ghi nhận metrics trung bình của vòng huấn luyện
        avg_loss = sum(
            w * float(valid_replies[idx].content["metrics"]["train_loss"])
            for w, idx in zip(norm_weights, selected_indices)
        )
        agg_soft_dist = sum(
            w * soft_labels_list[idx] for w, idx in zip(norm_weights, selected_indices)
        )
        group_ent = self._shannon_entropy(agg_soft_dist)

        return (
            ArrayRecord(new_global_state_dict),
            MetricRecord(
                {
                    "train_loss": avg_loss,
                    "selected_clients": len(selected_indices),
                    "group_entropy": group_ent,
                }
            ),
        )