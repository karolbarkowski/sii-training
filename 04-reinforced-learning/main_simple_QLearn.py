import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim


# Prosty model Q: wejście = [state, action], wyjście = przewidywana wartość Q
class QNetwork(nn.Module):
    def __init__(self):
        super(QNetwork, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(3, 16), nn.ReLU(), nn.Linear(16, 1)  # 2 stany + 1 akcja
        )

    def forward(self, x):
        return self.fc(x)


q_net = QNetwork()
optimizer = optim.Adam(q_net.parameters(), lr=0.01)
loss_fn = nn.MSELoss()


# Symulacja nauki - aktualizujemy model na podstawie doświadczeń
# Zakładamy: state = [x, y], action = 0/1/2/3

# przykładowe dane treningowe (state, action, reward)
experiences = [([0, 0], 1, 1.0), ([0, 1], 2, 0.0), ([1, 1], 0, 1.0)]

for epoch in range(100):
    total_loss = 0
    for state, action, reward in experiences:
        sa_input = torch.tensor([*state, action], dtype=torch.float32)
        target = torch.tensor([reward], dtype=torch.float32)

        pred = q_net(sa_input)
        loss = loss_fn(pred, target)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
    if epoch % 20 == 0:
        print(f"Epoch {epoch}, Loss: {total_loss:.4f}")
