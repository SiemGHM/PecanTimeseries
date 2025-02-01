import os
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.nn as nn

# Define the ToyModel class
class ToyModel(nn.Module):
    def __init__(self):
        super(ToyModel, self).__init__()
        self.net1 = nn.Linear(10, 10)
        self.relu = nn.ReLU()
        self.net2 = nn.Linear(10, 5)

    def forward(self, x):
        return self.net2(self.relu(self.net1(x)))

# Define the main training function
def demo_basic():
    # Initialize the process group
    dist.init_process_group("nccl")

    # Get rank and local rank
    rank = int(os.environ["RANK"])  # Global rank
    local_rank = int(os.environ["LOCAL_RANK"])  # Local rank on the current node
    world_size = dist.get_world_size()

    # Assign the GPU based on local rank
    device_id = local_rank  # Each local rank gets a unique GPU
    torch.cuda.set_device(device_id)

    print(f"I am rank {rank}, local rank {local_rank}, using GPU {device_id} on host {os.uname()[1]}.")

    # Create the model and move it to the assigned device
    model = ToyModel().to(device_id)

    # Wrap the model with DDP
    ddp_model = DDP(model, device_ids=[device_id])

    # Create a dummy input tensor and perform a forward pass
    input_tensor = torch.randn(20, 10).to(device_id)
    output = ddp_model(input_tensor)

    print(f"Rank {rank}: Forward pass output shape: {output.shape}")

    # Destroy the process group on completion
    dist.destroy_process_group()

if __name__ == "__main__":
    demo_basic()

