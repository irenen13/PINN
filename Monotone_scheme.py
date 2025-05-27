#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Apr 11 12:01:42 2025

@author: irenenoharinaivo
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Eikonal equation solver using finite differences for gradient approximation in 3D
@author: irenenoharinaivo
"""


# === Network ===




import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
class Net(nn.Module):
    def __init__(self, num_hidden=100):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(3, num_hidden)
        self.fc2 = nn.Linear(num_hidden, num_hidden)
        self.fc3 = nn.Linear(num_hidden, 1)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.fc3(x)



# Sampling uniform points within the unit ball
def sample_unit_ball(n, d, device):
    x = torch.randn(n, d, device=device)
    x = x / torch.norm(x, dim=1, keepdim=True)  # project to unit sphere
    r = torch.rand(n, 1, device=device) ** (1.0 / d)  # uniform radius
    return r * x  # scale direction vector by radius


# === Setup ===
d = 3                     # Dimension
h = 0.1                  # Finite difference step
lam = 1.0                 # Penalty on residual
T = 2000                 # Training iterations
batch_size = 1000         # Sample size
# ReLU function
m = nn.ReLU()

# === Device ===
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

# === Model and Optimizer ===
model = Net(num_hidden=100).to(device)
# model.fc3.bias.data.fill_(-1.0)  # Bias output upward
optimizer = optim.Adam(model.parameters(), lr=0.001)

# === Training Loop ===
print("Iteration, Loss")
for i in range(T + 1):
    optimizer.zero_grad()

    # Sample uniformly in [-1, 1]^3
    # x = (2 * torch.rand(batch_size, d) - 1).to(device)
    x = sample_unit_ball(batch_size, d, device)

    # Finite Difference Approximation
    grad_u_fd = torch.zeros_like(x)
    for j in range(d):
        ei = torch.zeros_like(x)
        ei[:, j] = h

        u_plus = model(x + ei)
        u_minus = model(x - ei)

        #grad_j = (u_plus - u_minus) / (2 * h) (centered difference)
        grad_j = torch.max(m(model(x)-u_minus), m(model(x)-u_plus))/h
        grad_u_fd[:, j] = grad_j.squeeze()

    grad_norm_fd = torch.norm(grad_u_fd, dim=1, keepdim=True)
    loss = lam * torch.mean((grad_norm_fd - 1)**2)

    # Boundary condition
    boundary_x = torch.randn_like(x)
    boundary_x = boundary_x / torch.norm(boundary_x, dim=1, keepdim=True)
    # u_b≈0
    u_b = model(boundary_x)
    loss += torch.mean(u_b**2)

    loss.backward()
    optimizer.step()

    if i % 1000 == 0:
        print(f"{i},{loss.item():.6f}")

# === Visualization of u(x, y, z=0) ===

# # Counting the singualarity points
# with torch.no_grad():
#     norms = torch.norm(x, dim=1)  # Euclidean norm for each point in the batch
#     near_origin = norms < h       # True if the point is closer than h to the origin. False otherwise
#     num_near_origin = near_origin.sum().item()  # Count how many such points
#     print(f"Points within h={h} of origin: {num_near_origin}/{batch_size}")



model.eval()
grid_size = 100
x = torch.linspace(-1, 1, grid_size)
y = torch.linspace(-1, 1, grid_size)
X, Y = torch.meshgrid(x, y, indexing='ij')
Z = torch.zeros_like(X)

X_flat = X.flatten()
Y_flat = Y.flatten()
Z_flat = Z.flatten()

input_grid = torch.stack([X_flat, Y_flat, Z_flat], dim=1).to(device)
with torch.no_grad():
    U = model(input_grid).cpu().numpy().reshape(grid_size, grid_size)

# Plot 3D surface
fig = plt.figure(figsize=(8, 6))
ax = fig.add_subplot(111, projection='3d')
ax.plot_surface(X.cpu().numpy(), Y.cpu().numpy(), U, cmap='viridis')
ax.set_xlabel('x')
ax.set_ylabel('y')
ax.set_zlabel('u(x, y, 0)')
ax.set_title('Monotone Scheme')
plt.tight_layout()
plt.show()

