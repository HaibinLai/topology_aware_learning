from __future__ import annotations

import sys
import time
import os
from torch import nn
from torch.utils.data import Dataset
from datetime import datetime
from sklearn.metrics import classification_report, roc_auc_score
import torch
from torch.utils.data import DataLoader
from torch.nn import functional as F  # noqa: N812

# TODO (MS): update all of these clients to decentralized clients
from src.decentralized_client import DecentralClient
from src.types import Result

from parsl.app.app import python_app

intel_xpu_count = torch.xpu.device_count()
if intel_xpu_count > 0:
    import intel_extension_for_pytorch as ipex


NUM_DATA_WORKERS = 0
_FAST_CLIENT_TENSORS: dict[int, tuple[torch.Tensor, torch.Tensor]] = {}


def register_fast_client_tensors(
    client: DecentralClient, inputs: torch.Tensor, targets: torch.Tensor
) -> None:
    """Cache deterministic image tensors for in-process client training."""
    _FAST_CLIENT_TENSORS[id(client)] = (inputs, targets)


def clear_fast_client_tensors() -> None:
    _FAST_CLIENT_TENSORS.clear()


def _validate_classification_targets(
    targets: torch.Tensor,
    logits: torch.Tensor,
    *,
    client_idx: int,
    round_idx: int,
    batch_idx: int,
    source_target_range: tuple[int, int] | None = None,
) -> None:
    """Fail on the CPU before CUDA cross entropy can trigger a device assert."""
    if targets.numel() == 0:
        raise ValueError(f"empty target batch client={client_idx} round={round_idx} batch={batch_idx}")
    target_min = int(targets.min().item())
    target_max = int(targets.max().item())
    num_classes = int(logits.shape[-1])
    if target_min < 0 or target_max >= num_classes:
        raise ValueError(
            f"invalid target labels client={client_idx} round={round_idx} batch={batch_idx} "
            f"min={target_min} max={target_max} num_classes={num_classes} "
            f"source_cpu_range={source_target_range} device={targets.device}"
        )


def _loader_kwargs() -> dict:
    """DataLoader settings for overlapping CPU input work with GPU compute."""
    pin_memory = torch.cuda.is_available() and os.environ.get("TOPO_FL_PIN_MEMORY", "1") != "0"
    return {
        "num_workers": NUM_DATA_WORKERS,
        "pin_memory": pin_memory,
        "persistent_workers": NUM_DATA_WORKERS > 0,
    }


def _move_batch_tensor(tensor: torch.Tensor, device: torch.device) -> torch.Tensor:
    # Non-blocking copies are only safe when the CPU source is pinned. This
    # avoids asynchronous reads from short-lived pageable DataLoader batches.
    return tensor.to(
        device,
        non_blocking=(device.type == "cuda" and tensor.device.type == "cpu" and tensor.is_pinned()),
    )


def accuracy(inputs, logits):
    # Shift so that tokens < n predict n
    shift_labels = inputs[..., 1:].contiguous()
    shift_logits = logits[..., :-1, :].contiguous()

    # converts logits to predictions
    predictions = torch.argmax(shift_logits, axis=-1)

    # Now compute accuracy
    N = torch.numel(predictions)
    accuracy = (shift_labels == predictions).sum() / N

    return accuracy


@python_app(executors=["decentral_train"])
def no_local_train(
    future: tuple(list[Result], DecentralClient),
    round_idx: int,
    epochs: int,
    batch_size: int,
    lr: float,
    momentum: float,
    prox_coeff: float,
    # device: torch.device,
    seed: int,
    backdoor: bool = False,
    dataset: DataChoices = None,
    optimizer: str = "sgd",
    weight_decay: float = 5e-4,
    beta_1: float = 0.9,
    beta_2: float = 0.98,
    *neighbor_futures: list[(list[Result], DecentralClient)],
) -> tuple(list[Result], DecentralClient):
    # NOTE(MS): assign device once task has been fired off, rather than before via a function arg
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device("xpu" if torch.xpu.is_available() else device)
    dataset_name = dataset.value.lower()

    if seed is not None:
        # torch.manual_seed will seed cuda GPUs
        torch.manual_seed(seed)
        torch.xpu.manual_seed(seed)

    client = future[1]
    if torch.cuda.is_available():
        visible = os.environ.get("CUDA_VISIBLE_DEVICES", "")
        if visible and "," not in visible:
            device = torch.device("cuda:0")
        else:
            device = torch.device(f"cuda:{client.idx % torch.cuda.device_count()}")
    results: list[Result] = []
    client.model.train()
    client.model = client.model.to(device)
    if optimizer == "sgd":
        optimizer = torch.optim.SGD(
            client.model.parameters(),
            lr=lr,
            momentum=momentum,
            weight_decay=weight_decay,
        )
    if optimizer == "adam":
        optimizer = torch.optim.Adam(
            client.model.parameters(), lr=lr, weight_decay=weight_decay
        )
    if optimizer == "adamw":
        optimizer = torch.optim.AdamW(
            client.model.parameters(),
            lr=lr,
            weight_decay=weight_decay,
            betas=(beta_1, beta_2),
        )
    loader = DataLoader(client.train_data, batch_size=batch_size, **_loader_kwargs())

    if prox_coeff > 0:
        for neighbor_future in neighbor_futures:
            neighbor_future[1].model.to(device)
    avg_time_per_epoch = 0
    for epoch in range(epochs):
        start_time = time.time()

        epoch_results = []
        running_loss = torch.zeros((), device=device)
        running_perp = torch.zeros((), device=device)
        running_acc = torch.zeros((), device=device)

        client.model.train()
        client.model = client.model.to(device)
        if intel_xpu_count > 0:
            client.model, optimizer = ipex.optimize(client.model, optimizer=optimizer)

        for batch_idx, batch in enumerate(loader):
            inputs, targets = batch
            source_target_range = (
                (int(targets.min().item()), int(targets.max().item()))
                if os.environ.get("TOPO_FL_VALIDATE_TARGET_TRANSFER", "0") == "1"
                else None
            )
            inputs = _move_batch_tensor(inputs, device)
            targets = _move_batch_tensor(targets, device)
            if "tiny_mem" in dataset_name:
                model_output = client.model(inputs, labels=inputs)
                loss = model_output.loss
                running_perp += torch.exp(loss).detach()
                running_acc += accuracy(inputs, model_output.logits)

            else:
                preds = client.model(inputs)
                _validate_classification_targets(
                    targets,
                    preds,
                    client_idx=client.idx,
                    round_idx=round_idx,
                    batch_idx=batch_idx,
                    source_target_range=source_target_range,
                )
                loss = F.cross_entropy(preds, targets)

            running_loss += loss.detach()

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()

        end_time = time.time()
        avg_time_per_epoch += end_time - start_time

        # The result table for final-only runs needs metrics from the final
        # aggregated models, not from every local epoch/round.  Skip these
        # repeated test-set passes; DecentrallearnApp schedules one evaluation
        # after the last aggregation instead.
        if os.environ.get("TOPO_FL_FINAL_EVAL_ONLY", "0") == "1":
            continue

        # Test client on local test set TODO

        # Test client on global test set
        global_test_result = test_model(
            client.model,
            client.global_test_data,
            round_idx,
            batch_size,
            seed,
            dataset,
        )

        if backdoor:
            # Test client on global backdoor test set
            global_backdoor_test_result = test_model(
                client.model,
                client.global_backdoor_test_data,
                round_idx,
                batch_size,
                seed,
                dataset,
            )

            global_test_result = global_test_result | {
                "backdoor_acc": global_backdoor_test_result["test_acc"],
                "backdoor_loss": global_backdoor_test_result["test_loss"],
                "backdoor_auc_macro": global_backdoor_test_result["clean_auc_macro"],
                "backdoor_auc_weighted": global_backdoor_test_result["clean_auc_weighted"],
            }
        epoch_results.append(
            {
                "avg_time_per_epoch": avg_time_per_epoch / epochs,
                # "avg_time_per_epoch": (start_time - end_time).total_seconds(),
                "date_time": datetime.now(),
                "client_idx": client.idx,
                "neighbors": client.neighbors,
                "round_idx": round_idx,
                "epoch": epoch,
                "data_size": len(client.train_data),
                "train_loss": (running_loss / len(loader)).item(),
                "train_perp": (running_perp / len(loader)).item(),
                "train_acc": (running_acc / len(loader)).item(),
            }
            | global_test_result,
        )

        results.extend(epoch_results)
        print(f"TRAIN_EPOCH_DONE client={client.idx} round={round_idx} epoch={epoch}", flush=True)

    # need to send model back to cpu since all coordination is happening from a non-gpu node
    client.model.to("cpu")
    return results, client


def local_train_impl(
    future: tuple(list[Result], DecentralClient),
    round_idx: int,
    epochs: int,
    batch_size: int,
    lr: float,
    momentum: float,
    prox_coeff: float,
    # device: torch.device,
    seed: int,
    backdoor: bool = False,
    dataset: DataChoices = None,
    optimizer: str = "sgd",
    weight_decay: float = 5e-4,
    beta_1: float = 0.9,
    beta_2: float = 0.98,
    *neighbor_futures: list[(list[Result], DecentralClient)],
) -> tuple(list[Result], DecentralClient):
    """Local training job.

    Args:
        client: The client to train.
        round_idx: The current round number.
        epochs: Number of epochs.
        batch_size: Batch size when iterating through data.
        lr: Learning rate.
        device: Backend hardware to train with.

    Returns:
        List of results that record the training history.
    """
    client = future[1]
    print(f"TRAIN_START client={client.idx} round={round_idx} epochs={epochs}", flush=True)
    inprocess_execution = os.environ.get("TOPO_FL_INPROCESS", "0") == "1"
    if inprocess_execution:
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        device = torch.device("xpu" if torch.xpu.is_available() else device)
    dataset_name = dataset.value.lower()

    # The in-process path has no stochastic model layers or shuffled loaders.
    # Avoid resetting PyTorch's process-global CPU/CUDA RNG concurrently from
    # several client threads.
    if seed is not None and not inprocess_execution:
        # torch.manual_seed will seed cuda GPUs
        torch.manual_seed(seed)
        torch.xpu.manual_seed(seed)

    if torch.cuda.is_available() and not inprocess_execution:
        visible = os.environ.get("CUDA_VISIBLE_DEVICES", "")
        if visible and "," not in visible:
            device = torch.device("cuda:0")
        else:
            device = torch.device(f"cuda:{client.idx % torch.cuda.device_count()}")
    results: list[Result] = []
    client.model.train()
    client.model = client.model.to(device)
    if optimizer == "sgd":
        optimizer = torch.optim.SGD(
            client.model.parameters(),
            lr=lr,
            momentum=momentum,
            weight_decay=weight_decay,
        )
    if optimizer == "adam":
        optimizer = torch.optim.Adam(
            client.model.parameters(), lr=lr, weight_decay=weight_decay
        )
    if optimizer == "adamw":
        optimizer = torch.optim.AdamW(
            client.model.parameters(),
            lr=lr,
            weight_decay=weight_decay,
            betas=(beta_1, beta_2),
        )
    fast_tensors = _FAST_CLIENT_TENSORS.get(id(client)) if inprocess_execution else None
    if fast_tensors is None:
        loader = DataLoader(client.train_data, batch_size=batch_size, **_loader_kwargs())
        num_batches = len(loader)
    else:
        loader = None
        num_batches = (len(fast_tensors[1]) + batch_size - 1) // batch_size

    if prox_coeff > 0:
        for neighbor_future in neighbor_futures:
            neighbor_future[1].model.to(device)

    avg_time_per_epoch = 0
    for epoch in range(epochs):
        start_time = time.time()

        epoch_results = []
        running_loss = torch.zeros((), device=device)
        running_perp = torch.zeros((), device=device)
        running_acc = torch.zeros((), device=device)

        client.model.train()
        client.model = client.model.to(device)
        if intel_xpu_count > 0:
            client.model, optimizer = ipex.optimize(client.model, optimizer=optimizer)

        if fast_tensors is None:
            batches = enumerate(loader)
        else:
            cached_inputs, cached_targets = fast_tensors
            batches = (
                (
                    batch_idx,
                    (
                        cached_inputs[batch_idx * batch_size : (batch_idx + 1) * batch_size],
                        cached_targets[batch_idx * batch_size : (batch_idx + 1) * batch_size],
                    ),
                )
                for batch_idx in range(num_batches)
            )
        for batch_idx, batch in batches:
            inputs, targets = batch
            if fast_tensors is None:
                inputs = _move_batch_tensor(inputs, device)
                targets = _move_batch_tensor(targets, device)
            if "tiny_mem" in dataset_name:
                model_output = client.model(inputs, labels=inputs)
                loss = model_output.loss
                running_perp += torch.exp(loss).detach()
                running_acc += accuracy(inputs, model_output.logits)

            else:
                preds = client.model(inputs)
                if fast_tensors is None:
                    _validate_classification_targets(
                        targets,
                        preds,
                        client_idx=client.idx,
                        round_idx=round_idx,
                        batch_idx=batch_idx,
                    )
                loss = F.cross_entropy(preds, targets)

            running_loss += loss.detach()

            # Append proximal term
            # Inspired by: https://github.com/ki-ljl/FedProx-PyTorch/blob/main/client.py#L62
            if prox_coeff > 0:
                proximal_term = 0.0
                for neighbor_future in neighbor_futures:
                    neighbor = neighbor_future[1]
                    for w, w_t in zip(
                        client.model.parameters(), neighbor.model.parameters()
                    ):
                        proximal_term += (w - w_t).norm(2)
                loss += (prox_coeff / 2) * proximal_term

            loss.backward()
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)

        end_time = time.time()
        avg_time_per_epoch += end_time - start_time

        # Final-only experiment runs evaluate the post-aggregation models once
        # at the end; avoid repeating full global-test inference each epoch.
        if os.environ.get("TOPO_FL_FINAL_EVAL_ONLY", "0") == "1":
            continue

        # Test client on local test set TODO

        # Test client on global test set
        global_test_result = test_model(
            client.model,
            client.global_test_data,
            round_idx,
            batch_size,
            seed,
            dataset,
        )

        if backdoor:
            # Test client on global backdoor test set
            global_backdoor_test_result = test_model(
                client.model,
                client.global_backdoor_test_data,
                round_idx,
                batch_size,
                seed,
                dataset,
            )

            global_test_result = global_test_result | {
                "backdoor_acc": global_backdoor_test_result["test_acc"],
                "backdoor_loss": global_backdoor_test_result["test_loss"],
                "backdoor_auc_macro": global_backdoor_test_result["clean_auc_macro"],
                "backdoor_auc_weighted": global_backdoor_test_result["clean_auc_weighted"],
            }
        epoch_results.append(
            {
                "avg_time_per_epoch": avg_time_per_epoch / epochs,
                # "avg_time_per_epoch": (start_time - end_time).total_seconds(),
                "date_time": datetime.now(),
                "client_idx": client.idx,
                "neighbors": client.neighbors,
                "round_idx": round_idx,
                "epoch": epoch,
                "data_size": len(client.train_data),
                "train_loss": (running_loss / num_batches).item(),
                "train_perp": (running_perp / num_batches).item(),
                "train_acc": (running_acc / num_batches).item(),
            }
            | global_test_result,
        )

        print(global_test_result)
        results.extend(epoch_results)

    # The in-process execution path keeps client weights resident on the GPU
    # across rounds, avoiding a GPU->CPU->GPU transfer for every client.
    if os.environ.get("TOPO_FL_INPROCESS", "0") != "1":
        client.model.to("cpu")
    return results, client


local_train = python_app(executors=["decentral_train"])(local_train_impl)


def evaluate_final_client_impl(
    client_future: tuple(list[Result], DecentralClient),
    round_idx: int,
    batch_size: int,
    seed: int,
    dataset: DataChoices,
    backdoor: bool = False,
    evaluate_noisy: bool = False,
) -> tuple(list[Result], DecentralClient):
    """Evaluate a client's model after the final round aggregation."""
    client = client_future[1]
    result = test_model(
        client.model,
        client.global_test_data,
        round_idx,
        batch_size,
        seed,
        dataset,
    )
    if backdoor:
        backdoor_result = test_model(
            client.model,
            client.global_backdoor_test_data,
            round_idx,
            batch_size,
            seed,
            dataset,
        )
        result.update(
            {
                "backdoor_acc": backdoor_result["test_acc"],
                "backdoor_loss": backdoor_result["test_loss"],
                "backdoor_auc_macro": backdoor_result["clean_auc_macro"],
                "backdoor_auc_weighted": backdoor_result["clean_auc_weighted"],
            }
        )
    if evaluate_noisy and client.global_noisy_test_data is not None:
        noisy_result = test_model(
            client.model, client.global_noisy_test_data, round_idx, batch_size, seed, dataset
        )
        result.update({
            "ood_acc": noisy_result["test_acc"],
            "ood_auc_macro": noisy_result["clean_auc_macro"],
        })
    result.update(
        {
            "client_idx": client.idx,
            "round_idx": round_idx,
            "epoch": 0,
            "data_size": len(client.train_data),
            "is_final_evaluation": True,
        }
    )
    client.model.to("cpu")
    print(f"FINAL_EVAL_DONE client={client.idx} round={round_idx}", flush=True)
    return [result], client


evaluate_final_client = python_app(executors=["decentral_train"])(evaluate_final_client_impl)


def test_model(
    model: nn.Module,
    data: Dataset,
    round_idx: int,
    batch_size: int,
    # device: torch.device,
    seed: int,
    dataset: DataChoices = None,
) -> Result:
    """Evaluate a model."""

    # NOTE(MS): assign device once task has been fired off, rather than before via a function arg
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device("xpu" if torch.xpu.is_available() else device)
    dataset_name = dataset.value.lower()

    if seed is not None:
        # torch.manual_seed will seed cuda GPUs
        torch.manual_seed(seed)
        torch.xpu.manual_seed(seed)

    if "tiny_mem" in dataset_name:
        model.eval()
        total_loss = torch.zeros((), device=device)
        total_perp = torch.zeros((), device=device)
        total_acc = torch.zeros((), device=device)
        with torch.no_grad():
            model.to(device)
            loader = DataLoader(data, batch_size=batch_size, **_loader_kwargs())
            for batch in loader:
                inputs, targets = batch
                inputs = inputs.to(device, non_blocking=True)
                targets = targets.to(device, non_blocking=True)
                model_outputs = model(inputs, labels=inputs)
                loss = model_outputs.loss
                total_loss += loss.detach()
                total_perp += torch.exp(loss).detach()
                total_acc += accuracy(inputs, model_outputs.logits)
        res: Result = {
            "test_loss": (total_loss / len(loader)).item(),
            "test_perp": (total_perp / len(loader)).item(),
            "test_acc": (total_acc / len(loader)).item(),
        }
        return res

    model.eval()
    total_loss = torch.zeros((), device=device)
    total_acc = torch.zeros((), device=device)
    n_batches = 0
    y_true = []
    y_pred = []
    y_prob = []
    with torch.no_grad():
        model.to(device)
        loader = DataLoader(data, batch_size=batch_size, **_loader_kwargs())
        for batch in loader:
            inputs, targets = batch
            inputs = inputs.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)
            preds = model(inputs)
            _validate_classification_targets(
                targets, preds, client_idx=-1, round_idx=round_idx, batch_idx=n_batches
            )
            y_true.append(targets.detach())
            y_prob.append(torch.softmax(preds, dim=1).detach())
            loss = F.cross_entropy(preds, targets)
            total_loss += loss.detach()

            # Accuracy calculations
            top_probabilities, top_preds = torch.topk(preds, k=1, dim=1)
            top_preds = torch.squeeze(top_preds)
            y_pred.append(top_preds.detach())
            acc = torch.sum(top_preds == targets) / targets.shape[0]
            total_acc += acc.detach()

            n_batches += 1

    # print("test acc: ", total_acc / n_batches)
    y_true_tensor = torch.cat(y_true).cpu()
    y_prob_tensor = torch.cat(y_prob).cpu()
    y_true = y_true_tensor.tolist()
    y_pred = torch.cat(y_pred).cpu().tolist()
    report = classification_report(
        y_true, y_pred, digits=4, output_dict=True, zero_division=0
    )

    stats_dict = {}
    for k in report:
        if dict == type(report[k]):
            # print(k, report[k])
            for header in report[k]:
                stats_dict[f"{k}_{header}"] = report[k][header]
                # print(f'{k}_{header}: {report[k][header]}')
    # print(stats_dict)

    try:
        clean_auc_macro = float(
            roc_auc_score(
                y_true_tensor.numpy(), y_prob_tensor.numpy(),
                multi_class="ovr", average="macro"
            )
        )
        clean_auc_weighted = float(
            roc_auc_score(
                y_true_tensor.numpy(), y_prob_tensor.numpy(),
                multi_class="ovr", average="weighted"
            )
        )
    except ValueError:
        clean_auc_macro = float("nan")
        clean_auc_weighted = float("nan")

    res: Result = {
        # "time": datetime.now(),
        # "round_idx": round_idx,
        "test_loss": (total_loss / n_batches).item(),
        "test_acc": (total_acc / n_batches).item(),
        "clean_auc_macro": clean_auc_macro,
        "clean_auc_weighted": clean_auc_weighted,
    } | stats_dict
    return res
