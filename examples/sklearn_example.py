import time
from dgd import DistributedGridDescent
import numpy as np
np.random.seed(42)
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.datasets import make_moons
import pandas as pd

class GBC:
    '''An esimator for simple DNN.'''
    def __init__(self,):
        self.params = None

    def build_model(self, **params):
        '''Build the model.'''
        self.params = params

        self.gb = GradientBoostingClassifier(**params, random_state=42)



    def score(self, data=None, cv_split=0.5):
        '''Evaluate function.'''

        X_train, X_val, y_train, y_val = train_test_split(data["X"],
            data["y"], test_size=cv_split, stratify=data["y"])

        self.gb.fit(X_train,y_train)

        acc = self.gb.score(X_val, y_val)

        return acc # {"acc":acc, "steps":np.random.rand()} # accuracy + fraction of budget needed


def main():
    model = GBC()
    params = {
        "loss":["deviance","exponential"],
        "learning_rate":[0.3,0.1,0.03,0.01,0.003,0.001],
        "n_estimators":[10,30,100,300,1000],
        "criterion":["friedman_mse","mse"],
        "subsample":[1.,0.9,0.8,0.7,0.6,0.5],
        "max_features":[1,2]#3,6,None]
    }

    X, y = make_moons(n_samples=500, shuffle=True, noise=1, random_state=42)
    X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.25,stratify=y)
    data = {"X": X_train,
            "y": y_train}

    start = time.perf_counter()

    # accuracy + the fraction of budget saved, if certain theshold surpassed
    # metric = lambda run_set: sum(rs["acc"] + (1-rs["steps"] if rs["acc"] > 0.7 else 0) for rs in run_set)/len(run_set)
    metric = lambda run_set: sum(run_set)/len(run_set)
    eps = lambda t: 1/(t+1)
    dgd = DistributedGridDescent(model, params, metric, n_iter=32,
        n_random_init=1, eps=eps, cv_split=0.5, n_jobs=4, random_state=42, verbose=2)
    dgd.run(data)
    end = time.perf_counter()
    print(f"Best parameters = {dgd.best_params_}")
    print(f"E[L(x*)] = {dgd.best_score_ :.3f}")
    print(f"Execution time: {end - start :.3f} s\n")
    df = pd.DataFrame(dgd.results_).set_index("ID").sort_values(by=["metric"],ascending=False)
    print(df)

    # Final evaluation
    print("Evaluation:")
    model.build_model(**dgd.best_params_)
    model.gb.fit(X_train,y_train)
    final_score = model.gb.score(X_test, y_test)
    print(f"Final validation score is: {final_score :.4f}")

    # print(pd.DataFrame(dgd.ordered_results_))

if __name__ == '__main__':
    main()
