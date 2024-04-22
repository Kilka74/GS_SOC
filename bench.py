import logging
import os
import time
import wandb
import numpy as np
import torch
import torch.nn as nn
import hydra
from preactresnet import *
from train_standard import init_model
from utils import (
    upper_limit,
    lower_limit,
    cifar10_mean,
    cifar10_std,
    clamp,
    get_loaders,
    attack_pgd,
    evaluate_pgd,
    evaluate_standard,
)
import torch.utils.benchmark as benchmark

logger = logging.getLogger(__name__)
criterion = nn.CrossEntropyLoss()


def fwd(model, opt, X, y):
    output = model(X)
    ce_loss = criterion(output, y)
    opt.zero_grad(set_to_none=True)
    ce_loss.bacward()
    opt.step()


def warmup():
    linear = nn.Linear(10000, 512, device="cuda")
    x = torch.randn(64, 10000, device="cuda")
    t0 = benchmark.Timer(stmt="linear(x)", globals={"x": x, "linear": linear})
    res = t0.timeit(1000).mean
    del x
    del linear
    torch.cuda.empty_cache()



@hydra.main(config_path="conf", config_name="config", version_base=None)
def main(args):

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)

    train_loader, test_loader = get_loaders(
        args.data_dir, args.batch_size, args.dataset
    )
    for i in range(5):
        warmup()

    groups = [1, 2, 4, 8, 16, 32]

    for conv in ["soc", "monarch_soc"]:
        for group in groups:
            args.groups = group
            args.conv_layer = conv
            model = init_model(args).cuda()
            model.train()
            conv_params = []
            activation_params = []
            other_params = []
            for name, param in model.named_parameters():
                if param.requires_grad:
                    if "activation" in name:
                        activation_params.append(param)
                    elif "conv" in name:
                        conv_params.append(param)
                    else:
                        other_params.append(param)

            opt = torch.optim.SGD(
                [
                    {"params": activation_params, "weight_decay": 0.0},
                    {
                        "params": (conv_params + other_params),
                        "weight_decay": args.weight_decay,
                    },
                ],
                lr=args.lr_max,
                momentum=args.momentum,
            )

            lr_steps = args.epochs * len(train_loader)
            
            X, y = next(train_loader)
            X, y = X.cuda(), y.cuda()

            t0 = benchmark.Timer(stmt="bench(model, opt, X, y)", globals={
                "bench": fwd,
                "model" : model,
                "opt": opt,
                "X": X,
                "y": y
            })
            del model
            del opt
            del X
            del y
            torch.cuda.empty_cache()

            time = t0.timeit(100).mean
            print(f"Conv: {args.conv_layer}, groups: {args.groups}, time: {time}")


if __name__ == "main":
    main()

