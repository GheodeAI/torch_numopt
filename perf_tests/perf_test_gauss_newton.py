import torch
import torch.nn as nn
import torch.optim as optim
import torch_numopt
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, TensorDataset
from sklearn.datasets import load_diabetes, fetch_california_housing
from sklearn.preprocessing import MinMaxScaler
import time

if __name__ == "__main__":
    torch.manual_seed(0)

    device = "cpu"

    class Net(nn.Module):
        def __init__(self, input_size, device="cpu"):
            super().__init__()
            self.f1 = nn.Linear(input_size, 10, device=device)
            self.f2 = nn.Linear(10, 20, device=device)
            self.f3 = nn.Linear(20, 20, device=device)
            self.f4 = nn.Linear(20, 10, device=device)
            self.f5 = nn.Linear(10, 1, device=device)

            self.activation = nn.ReLU()
            # self.activation = nn.Sigmoid()

        def forward(self, x):
            x = self.activation(self.f1(x))
            x = self.activation(self.f2(x))
            x = self.activation(self.f3(x))
            x = self.activation(self.f4(x))
            x = self.f5(x)

            return x

    # X, y = load_diabetes(return_X_y = True, scaled=False)
    X, y = fetch_california_housing(return_X_y=True)

    X_scaler = MinMaxScaler()
    X = X_scaler.fit_transform(X)

    y_scaler = MinMaxScaler()
    y = y_scaler.fit_transform(y.reshape((-1, 1)))

    torch_data = TensorDataset(torch.Tensor(X).to(device), torch.Tensor(y).to(device))
    data_loader = DataLoader(torch_data, batch_size=1000)

    model = Net(input_size=X.shape[1], device=device)
    loss_fn = nn.MSELoss()
    opt = torch_numopt.GaussNewton(model.parameters(), lr=1, model=model, c1=1e-4, tau=0.1, line_search_method="backtrack")

    times = []

    all_loss = []
    for epoch in range(10):
        start = time.perf_counter()
        print("epoch: ", epoch, end="", flush=True)
        all_loss.append(0)
        for batch_idx, (b_x, b_y) in enumerate(data_loader):
            pre = model(b_x)
            loss = loss_fn(pre, b_y)
            opt.zero_grad()
            loss.backward()

            # parameter update step based on optimizer
            opt.step(b_x, b_y, loss_fn)
            opt.update_params(loss)

            all_loss[epoch] += loss
        end = time.perf_counter()

        all_loss[epoch] /= len(data_loader)

        loss_cpu = all_loss[epoch].detach().cpu().numpy().item()
        print(f", loss: {loss_cpu}, time spent: {end-start}s")
        times.append(end - start)
    print(f"Avg time: {sum(times)/len(times)}s")
