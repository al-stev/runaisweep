import signal
import sys
import wandb
from types import FrameType
import time
import logging
import torch
import torch.optim as optim
import torch.nn.functional as F
import torch.nn as nn
from torchvision import datasets, transforms
import os
import argparse

root = logging.getLogger()
root.setLevel(logging.DEBUG)

handler = logging.StreamHandler(sys.stdout)
handler.setLevel(logging.DEBUG)
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
handler.setFormatter(formatter)
root.addHandler(handler)
wandb.login(key=os.environ['WANDB_API_KEY'],relogin=True, force=True)

base_config= {'batch_size': 32,
                'dropout': 0.4,
                'epochs': 200,
                'fc_layer_size': 128,
                'learning_rate': 0.1,
                'optimizer': 'adam'}

def shutdown_handler(signal: int, frame: FrameType) -> None:
    global current_run
    print('Going into shutdown handler')
    root.warning("Signal received, safely shutting down.")
    if current_run:
        print(f"Marking sweep ({current_run.name}) as preempted.")
        current_run.mark_preempting()
    root.warning("Exiting process")
    sys.exit(1)

def train(build_network,build_optimizer,build_dataset,train_epoch):
    # Initialize a new wandb run
    global current_run
    with wandb.init(config=None, resume=True) as run:
        current_run = run
        # If called by wandb.agent, as below,
        # this config will be set by Sweep Controller
        config = wandb.config

        loader = build_dataset(run.config.batch_size)
        network = build_network(run.config.fc_layer_size, run.config.dropout)
        optimizer = build_optimizer(network,run.config.optimizer, run.config.learning_rate)

        for epoch in range(config.epochs):
            avg_loss = train_epoch(network, loader, optimizer)
            wandb.log({"loss": avg_loss, "epoch": epoch})    

def build_dataset(batch_size):
   
    transform = transforms.Compose(
        [transforms.ToTensor(),
         transforms.Normalize((0.1307,), (0.3081,))])
    # download MNIST training dataset
    dataset = datasets.MNIST(".", train=True, download=True,
                             transform=transform)
    sub_dataset = torch.utils.data.Subset(
        dataset, indices=range(0, len(dataset), 5))
    loader = torch.utils.data.DataLoader(sub_dataset, batch_size=batch_size)

    return loader


def build_network(fc_layer_size, dropout):
    network = nn.Sequential(  # fully-connected, single hidden layer
        nn.Flatten(),
        nn.Linear(784, fc_layer_size), nn.ReLU(),
        nn.Dropout(dropout),
        nn.Linear(fc_layer_size, 10),
        nn.LogSoftmax(dim=1))

    return network.to(device)
        

def build_optimizer(network, optimizer, learning_rate):
    if optimizer == "sgd":
        optimizer = optim.SGD(network.parameters(),
                              lr=learning_rate, momentum=0.9)
    elif optimizer == "adam":
        optimizer = optim.Adam(network.parameters(),
                               lr=learning_rate)
    return optimizer


def train_epoch(network, loader, optimizer):
    cumu_loss = 0
    for _, (data, target) in enumerate(loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()

        # ➡ Forward pass
        loss = F.nll_loss(network(data), target)
        cumu_loss += loss.item()
        # ⬅ Backward pass + weight update
        loss.backward()
        optimizer.step()

        wandb.log({"batch loss": loss.item()})

    return cumu_loss / len(loader)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Fun description")
    parser.add_argument('--sweepid', type=str, required=True,
                        help='The sweep ID to use for W&B sweeps.')
    parser.add_argument('--count', type=int, required=False,
                        help='The number of sweeps to run.')
    
    # Step 3: Parse the command line arguments
    args = parser.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    signal.signal(signal.SIGTERM, shutdown_handler)
    root.info(f"Running sweep {args.sweepid} count={args.count}")
    agent = wandb.agent(sweep_id=args.sweepid, function=lambda: train(
        build_network=build_network,
        build_optimizer=build_optimizer,
        build_dataset=build_dataset,
        train_epoch=train_epoch),
        count=args.count)
