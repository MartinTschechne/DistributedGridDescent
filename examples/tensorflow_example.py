import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras import regularizers
import time
from dgd import DistributedGridDescent
import numpy as np
from sklearn.model_selection import train_test_split
import pandas as pd
from pprint import pprint

class DNN:
    """A model for simple DNN."""
    def __init__(self, n_epochs):
        self.n_epochs = n_epochs
        self.net = None
        self.params = None
        self.optimizer = None
        self.lr_scheduler = None

    def build_model(self, **params):
        """Build the model."""
        self.params = params
        model = keras.Sequential()
        model.add(layers.Input(shape=(500,)))
        # model.add(layers.Conv2D(8, 3,activation="relu",
        #                         kernel_regularizer=regularizers.l2(params["l2"])))
        model.add(layers.MaxPooling2D((2,2)))
        model.add(layers.Flatten())
        for _ in range(params["num_linear_layers"]):
            model.add(layers.Dense(params["num_neurons"],
                                   activation="relu",
                                   kernel_regularizer=regularizers.l2(params["l2"])))
            model.add(layers.Dropout(params["dropout"]))
        model.add(layers.Dense(10,activation="softmax"))

        self.net = model

        if params["lr_annealing"]:
            lr_schedule = keras.optimizers.schedules.ExponentialDecay(
                            initial_learning_rate=params["learning_rate"],
                            decay_steps=1,
                            decay_rate=params["lr_annealing"])
        else:
            lr_schedule = params["learning_rate"]

        if params["optimizer"] == "adam":
            self.optimizer = keras.optimizers.Adam(learning_rate=lr_schedule)
        elif params["optimizer"] == "rmsprop":
            self.optimizer = keras.optimizers.RMSprop(learning_rate=lr_schedule)


    def score(self, data, cv_split):
        '''Evaluate function.'''

        data["data"] = data["data"].astype("float32") / 255.

        data["data"] = data["data"].reshape((-1,28,28,1))

        X_train, X_val, y_train, y_val = train_test_split(
            data["data"],
            data["targets"],
            test_size=cv_split,
            stratify=data["targets"])

        self.net.compile(loss='sparse_categorical_crossentropy',
                        optimizer=self.optimizer,
                        metrics = ["accuracy"])
        self.net.fit(X_train, y_train,
                    batch_size=self.params["batch_size"],
                    epochs=self.n_epochs,
                    verbose=0)

        _, acc = self.net.evaluate(X_val, y_val, verbose=0)
        return acc


def main():
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

    (images, labels), _ = keras.datasets.mnist.load_data()
    data = {"data":images,
            "targets":labels}

    metric = lambda run_set: sum(run_set)/len(run_set)
    model = DNN(n_epochs=1)
    start = time.perf_counter()
    dgd = DistributedGridDescent(model, params, metric,
        n_iter=40, cv_split=0.5, n_random_init=4, n_jobs=2, verbose=1)
    dgd.run(data)
    end = time.perf_counter()
    print("Parallel:")
    print(f"Best parameters = {dgd.best_params_}")
    print(f"E[L(x*)] = {dgd.best_score_ :.3f}")
    print(f"Execution time: {end - start :.3f} s\n")
    df = pd.DataFrame(dgd.results_).set_index("ID").sort_values(["metric"],ascending=False)
    pprint(df)

if __name__ == '__main__':
    main()
