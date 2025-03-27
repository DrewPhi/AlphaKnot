import torch
import torch.nn as nn
from torchviz import make_dot

# Define the KnotNNet model
class KnotNNet(nn.Module):
    def __init__(self, input_size, action_size):
        super(KnotNNet, self).__init__()
        self.fc1 = nn.Linear(input_size, 64)
        self.fc2 = nn.Linear(64, 64)
        self.fc_policy = nn.Linear(64, action_size)
        self.fc_value = nn.Linear(64, 1)

    def forward(self, x):
        x = x.float()
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        policy = self.fc_policy(x)
        value = torch.tanh(self.fc_value(x))
        return policy, value

# Create a sample model instance
input_size = 6  # Example input size
action_size = 12  # Example action size
model = KnotNNet(input_size, action_size)

# Create a sample input
sample_input = torch.randn(1, input_size)

# Perform a forward pass to capture the computational graph
policy, value = model(sample_input)

# Generate a visual representation of the model
dot = make_dot((policy, value), params=dict(model.named_parameters()))
dot.render("KnotNNet_Diagram", format="png")  # Saves the diagram as 'KnotNNet_Diagram.png'
