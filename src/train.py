import os
import json
import torch
from time import time
from tqdm import tqdm
from typing import Any, Dict

from loss import BeardLoss
from model import get_model
from utils import create_dir
from test import get_metric_scores
from dataset import create_dataloader
from visualize import plot_performance_curves


def train_model(config: Dict[Any, Any]) -> torch.nn.Module:
    output_path = config["training"]["output_path"]
    create_dir(output_path)

    with open(os.path.join(output_path, "ExperimentSummary.json"), "w") as f:
        json.dump(config, f, indent=2)

    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = (
            "1"  # fallback to cpu if an mps-incompatible op is tried
        )
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    print(f"[INFO] Running on {device}")

    dataloaders = {
        "train": create_dataloader(
            data_path=os.path.join(config["data"]["data_path"], "train"),
            transform=config["data"]["transforms"],
            shuffle=True,
            **config["training"]["dataloader_args"],
        ),
        "test": create_dataloader(
            data_path=os.path.join(config["data"]["data_path"], "test"),
            shuffle=True,
            **config["training"]["dataloader_args"],
        ),
    }

    model = get_model(**config["model"]).to(device)
    criterion = BeardLoss(config["training"]["criterion_args"]).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), **config["training"]["optimizer_args"])
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, **config["training"]["scheduler_args"]
    )

    tick = time()
    best_epoch = -1
    best_error = 999999
    phases = ["train", "test"]

    metric_list = ["Loss"]
    if config["training"]["metric_list"]:
        metric_list += config["training"]["metric_list"]

    metrics = {metric: {phase: [] for phase in phases} for metric in metric_list}
    ckpt_path = os.path.join(output_path, "checkpoint.pt")
    for epoch in range(config["training"]["num_epochs"]):
        print("-" * 20)
        print(f"Epoch {epoch + 1} / {config['training']['num_epochs']}")
        for phase in phases:
            if phase == "train":
                model.train()
            else:
                model.eval()
            running_metrics = {metric: 0 for metric in metric_list}
            with torch.set_grad_enabled(phase == "train"):
                for data, target in tqdm(
                    dataloaders[phase], total=len(dataloaders[phase]), ncols=94
                ):
                    data, target = data.to(device), target.to(device)
                    optimizer.zero_grad()
                    output = model(data)
                    loss = criterion(output, target)
                    running_metrics["Loss"] += loss.item()

                    metric_scores = get_metric_scores(metric_list[1:], output, target)
                    for key, score in metric_scores.items():
                        running_metrics[key] += score

                    if phase == "train":
                        loss.backward()
                        optimizer.step()

            for key, score in running_metrics.items():
                score /= len(dataloaders[phase])
                print(f"{key}: {score:.3f}")
                metrics[key][phase].append(score)

            if phase == "test":
                running_error = running_metrics["Loss"]
                scheduler.step(running_error)
                if running_error < best_error:
                    best_error = running_error
                    best_epoch = epoch
                    print(f"+ Saving the model to {ckpt_path}...")
                    torch.save(model.state_dict(), ckpt_path)
        # If no validation improvement has been recorded for "early_stop" number of epochs
        # stop the training.
        if epoch - best_epoch >= config["training"]["early_stop_patience"]:
            print(f"No improvements in {config['training']['early_stop_patience']} epochs, stop!")
            break

    total_time = time() - tick
    m, s = divmod(total_time, 60)
    h, m = divmod(m, 60)
    print(f"Training took {int(h):d} hours {int(m):d} minutes {s:.2f} seconds.")

    best_model_state = torch.load(ckpt_path, map_location=device, weights_only=True)
    model.load_state_dict(best_model_state)
    plot_performance_curves(metrics, output_path)
    return model
