import torch
from torch import nn
from torch.utils.data import SubsetRandomSampler
import torchvision
import torchvision.datasets as datasets
import time
from dgd import DistributedGridDescent
import numpy as np
from sklearn.model_selection import train_test_split

class LinearLayer(nn.Module):
    def __init__(self, input_size, output_size, p):
        super(LinearLayer, self).__init__()
        layer = [nn.Linear(input_size, output_size)]
        layer += [nn.ReLU()]
        layer += [nn.Dropout(p=p)]
        self.layer = nn.Sequential(*layer)

    def forward(self, x):
        return self.layer(x)

class DNN:
    """A simple DNN model."""
    def __init__(self, n_epochs):
        self.n_epochs = n_epochs
        self.net = None
        self.params = None
        self.optimizer = None
        self.lr_scheduler = None

    def build_model(self, **params):
        """Build the model."""
        self.params = params
        layers = [nn.Flatten(),LinearLayer(28**2,params["num_neurons"],params["dropout"])]
        layers += [LinearLayer(params["num_neurons"],params["num_neurons"],params["dropout"]) for _ in range(params["num_linear_layers"])]
        layers += [nn.Linear(params["num_neurons"], 10)]
        self.net = nn.Sequential(*layers)

        if params["optimizer"] == "adam":
            self.optimizer = torch.optim.Adam(
                self.net.parameters(),
                lr=params["learning_rate"],
                weight_decay=params["l2"])
        elif params["optimizer"] == "rmsprop":
            self.optimizer = torch.optim.RMSprop(
                self.net.parameters(),
                lr=params["learning_rate"],
                weight_decay=params["l2"])
        else:
            print("Optimizer not found.")
            exit(0)

        if params["lr_annealing"]:
            self.lr_scheduler = torch.optim.lr_scheduler.MultiplicativeLR(
                self.optimizer,
                lr_lambda=lambda epoch: params["lr_annealing"])

        self.loss_fn = nn.CrossEntropyLoss()

    def score(self, data=None, cv_split=0.5):
        """Evaluate function."""

        X_train, X_test, y_train, y_test = train_test_split(data["train_data"].data,
            data["train_data"].targets, test_size=cv_split, stratify=data["train_data"].targets)

        BS = self.params["batch_size"]

        # train loop
        self.net.train()
        for epoch in range(self.n_epochs):
            for b in range(len(X_train)//BS):
                data = X_train[b*BS:(b+1)*BS]
                target = y_train[b*BS:(b+1)*BS]
                self.optimizer.zero_grad()
                out = self.net(data)
                loss = self.loss_fn(out, target)
                loss.backward()
                self.optimizer.step()
            if self.params["lr_annealing"]:
                self.lr_scheduler.step()

        # test loop
        self.net.eval()
        acc = 0
        with torch.no_grad():
            for b in range(len(X_train)//1024):
                data = X_test[b*1024:(b+1)*1024]
                target = y_test[b*1024:(b+1)*1024]
                out = self.net(data)
                pred = out.data.max(1, keepdim=True)[1]
                acc += pred.eq(target.data.view_as(pred)).sum().numpy()

        return acc / len(y_test)


def main():
    model = DNN(n_epochs=1)
    params = {
        "learning_rate":[3e-3, 1e-3, 3e-4, 1e-4, 3e-5, 1e-5],
        "optimizer":["adam", "rmsprop"],
        "lr_annealing":[False, 0.95, 0.99],
        "batch_size":[32, 64, 128, 256, 1024],
        "num_linear_layers":[1, 2, 4, 8, 16],
        "num_neurons":[512, 256, 128, 64, 32, 16],
        "dropout":[0.0, 0.1, 0.3, 0.5],
        "l2":[0.0, 0.01, 0.1]
    }
    data = {"train_data":None}

    data["train_data"] = torchvision.datasets.MNIST(
        './data',
        train=True,
        download=True,
        transform=torchvision.transforms.Compose([
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize(
            (0.1307,), (0.3081,))]))

    start = time.perf_counter()
    metric = lambda run_set: sum(run_set)/len(run_set)
    dgd = DistributedGridDescent(model, params, metric, cv_split=0.5, n_iter=40, n_random_init=4, n_jobs=1)
    dgd.run(data)
    end = time.perf_counter()
    print("Parallel:")
    print(f"Best parameters = {dgd.best_params_}")
    print(f"E[L(x*)] = {dgd.best_score_ :.3f}")
    print(f"Execution time: {end - start :.3f} s\n")

if __name__ == '__main__':
    main()
