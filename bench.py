import logging
import numpy as np
import torch
import torch.nn as nn
import hydra
from time import sleep
import wandb
from preactresnet import *
from train_standard import init_model as init_model_standard
from train_robust import init_model as init_model_robust
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

def warmup():
    linear = nn.Linear(10000, 512, device="cuda")
    x = torch.randn(64, 10000, device="cuda")
    t0 = benchmark.Timer(stmt="linear(x)", globals={"x": x, "linear": linear})
    res = t0.timeit(1000).mean
    del x
    del linear
    torch.cuda.empty_cache()


def train_epoch_bench(model, loader, loss, opt):
    model.train()
    train_loss = 0
    # train_acc = 0
    # train_n = 0
    for _, (X, y) in enumerate(loader):
        X, y = X.cuda(), y.cuda()

        output = model(X)
        ce_loss = loss(output, y)
        opt.zero_grad(set_to_none=True)
        ce_loss.backward()
        opt.step()

        train_loss += ce_loss.item() * y.size(0)
        # train_acc += (output.max(1)[1] == y).sum().item()
        # train_n += y.size(0)

@torch.no_grad()
def eval_epoch(model, loader, loss):
    model.eval()
    test_loss = 0
    # test_acc = 0
    # test_n = 0
    with torch.no_grad():
        for _, (X, y) in enumerate(loader):
            X, y = X.cuda(), y.cuda()

            output = model(X)
            ce_loss = loss(output, y)

            test_loss += ce_loss.item() * y.size(0)
            # test_acc += (output.max(1)[1] == y).sum().item()
            # test_n += y.size(0)

@hydra.main(config_path="conf", config_name="config_benchmark", version_base=None)
def main(args):

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)

    train_loader, test_loader = get_loaders(
        args.data_dir, args.batch_size, args.dataset
    )
    for _ in range(10):
        warmup()

    if (isinstance(args.groups, int)) or (args.conv_layer == "monarch_soc"):
        wandb.login(key=args.wandb_key, relogin=True)
        wandb.init(
            entity="kilka74",
            project="MonarchSOC",
            tags=[args.dataset, "benchmark"],
            name=f"benchmark epoch time {args.model_name}, {args.conv_layer}, {args.dataset}, groups={args.groups}, wd={args.weight_decay}",
            config={
                "batch_size": args.batch_size,
                "model_name": args.model_name,
                "dataset": args.dataset,
                "seed": args.seed,
                "activation": args.activation,
                "conv_layer": args.conv_layer,
                "max_lr": args.lr_max,
                "weight_decay": args.weight_decay,
                "momentum": args.momentum,
                "groups": args.groups,
            }
        )

        block_sizes = list(map(int, args.block_size.split()))

        for block_size in block_sizes:
            torch.cuda.empty_cache()
            args.block_size = block_size
            model = init_model_robust(args).cuda()
                
            opt = torch.optim.SGD(
                model.parameters(),
                weight_decay=args.weight_decay,
                lr=args.lr_max,
                momentum=args.momentum
            )
            criterion = nn.CrossEntropyLoss()

            t0 = benchmark.Timer(stmt="bench(model, loader, loss, opt)", globals={
                "bench": train_epoch_bench,
                "model" : model,
                "loader": train_loader,
                "loss": criterion,
                "opt": opt
            })
            torch.cuda.empty_cache()

            train_time = t0.timeit(5).mean
            torch.cuda.empty_cache()
            t1 = benchmark.Timer(stmt="bench(model, loader, loss)", globals={
                "bench": eval_epoch,
                "model": model,
                "loader": test_loader,
                "loss": criterion
            })
            test_time = t1.timeit(5).mean

            wandb.log({
                "lipconvnet": block_size * 5,
                "train_time": train_time,
                "test_time": test_time,
                "number of parameters with grad": sum(p.numel() for p in model.parameters() if p.requires_grad),
                "number of all parameters": sum(p.numel() for p in model.parameters()),
            })
        wandb.finish()


if __name__ == "__main__":
    main()

