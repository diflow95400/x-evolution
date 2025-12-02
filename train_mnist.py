import torch
from torch import tensor, nn
import torch.nn.functional as F
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

# model

model = nn.Sequential(
    nn.Flatten(),
    nn.Linear(784, 512),
    nn.ReLU(),
    nn.Linear(512, 256),
    nn.ReLU(),
    nn.Linear(256, 128),
    nn.ReLU(),
    nn.Linear(128, 64),
    nn.ReLU(),
    nn.LayerNorm(64, bias = False),
    nn.Linear(64, 10),
)

# data

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
])

dataset = datasets.MNIST('./data', train = True, download = True, transform = transform)

# fitness as inverse of loss

def loss_mnist(model):
    device = next(model.parameters()).device
    
    dataloader = DataLoader(dataset, batch_size = 16, shuffle = True)
    data_iterator = iter(dataloader)
    data, target = next(data_iterator)

    data, target = data.to(device), target.to(device)

    with torch.no_grad():
        logits = model(data)
        loss = F.cross_entropy(logits, target)

    return -loss

# evo

from x_evolution import EvoStrategy

evo_strat = EvoStrategy(
    model,
    environment = loss_mnist,
    noise_population_size = 100,
    noise_scale = 1e-2,
    noise_low_rank = 2,
    num_generations = 10_000,
    learning_rate = 1e-5
)

evo_strat()
