# === Neural Nets ===
def build_network(num_hidden=100):
    class Net(nn.Module):
        def __init__(self):
            super(Net, self).__init__()
            self.fc1 = nn.Linear(3, num_hidden)
            self.fc2 = nn.Linear(num_hidden, num_hidden)
            self.fc3 = nn.Linear(num_hidden, 1)

        def forward(self, x):
            x = F.relu(self.fc1(x))
            x = F.relu(self.fc2(x))
            return self.fc3(x)

    return Net()
