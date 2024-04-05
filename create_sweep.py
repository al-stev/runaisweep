# not used in example but for avoidance of you can create a sweep in 3 ways -> UI, Python, CLI
api = wandb.Api()
project = api.project(f"{os.environ['WANDB_ENTITY']}/{os.environ['WANDB_PROJECT']}")

# Step 4: List all sweeps in the project

### create a sweep
sweep_config = {
    "name": "My Sweep",
    "method": "grid",
    "parameters": {
        "batch_size": {"values": [32, 64]},
        "dropout": {"values": [0.2, 0.4]},
        "fc_layer_size": {"values": [128, 256]},
        "learning_rate": {"values": [0.1, 0.01]},
        "optimizer": {"values": ["adam", "sgd"]},
        "epochs": {"values": [10, 20]}
    }
}
sweep_id = wandb.sweep(sweep_config, project=os.environ['WANDB_PROJECT'], entity=os.environ['WANDB_ENTITY'])
print(sweep_id)