import random
import time
from dgd import DistributedGridDescent
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import ticker, colors

class Rosenbrock:
    '''A wrapper for the noisy Rosenbrock function..'''
    def __init__(self, N, noise=0.):
        self.noise = noise
        self.N = N
        self.params = None

    def build_model(self, **params):
        '''Build the model.'''
        assert len(params) == self.N
        self.params = params

    def score(self, data=None, cv_split=None):
        '''Evaluate function.'''
        param_list = [v for _,v in self.params.items()]
        time.sleep(0.1)
        return -sum(10*(param_list[i+1] - param_list[i]**2)**2
                + (1. - param_list[i])**2 for i in range(self.N-1)) + self.noise*np.random.randn()

def main():
    N = 2
    rosenbrock = Rosenbrock(N,noise=0.5)
    params = {k:[i*4/500 - 2 for i in range(501)] for k in [f"x{j}" for j in range(N)]}
    metric = lambda run_set: sum(run_set)/len(run_set)

    N_ITER = 128
    N_JOBS = [1,2,4,16,32]

    fig, axes = plt.subplots(1,2,figsize=(6,12))
    ax = axes.ravel()
    # Strong scaling
    print("Strong scaling analysis")
    exec_time = []
    for n_jobs in N_JOBS:
        start = time.perf_counter()
        dgd = DistributedGridDescent(rosenbrock, params, metric, n_iter=N_ITER,
            n_jobs=n_jobs, n_random_init=8, verbose=0)
        dgd.run()
        end = time.perf_counter()
        print(f"n_jobs: {n_jobs}:")
        print(f"Execution time: {end - start :.3f} s\n")
        exec_time.append(end-start)

    ax[0].plot(N_JOBS,exec_time[0]/np.array(exec_time),label="Strong speedup",color="r")

    # Weak scaling
    print("Weak scaling analysis")
    exec_time = []
    for n_jobs in N_JOBS:
        start = time.perf_counter()
        dgd = DistributedGridDescent(rosenbrock, params, metric, n_iter=N_ITER*n_jobs,
            n_jobs=n_jobs, n_random_init=8, verbose=0)
        dgd.run()
        end = time.perf_counter()
        print(f"n_jobs: {n_jobs}:")
        print(f"Execution time: {end - start :.3f} s\n")
        exec_time.append(end-start)

    ax[0].plot(N_JOBS,exec_time[0]/np.array(exec_time),label="Weak efficiency",color="b")
    ax[0].plot(N_JOBS,N_JOBS,'--',label="Ideal speedup",color="r")
    ax[0].plot(N_JOBS,[1]*len(N_JOBS),'--',label="Ideal efficiency",color="b")
    ax[0].set_ylim((0,N_JOBS[-1]+0.5))
    ax[0].set_xlabel("n_jobs")
    ax[0].set_title("Strong and weak scaling analysis")
    ax[0].legend(loc="upper left")

    # noise free 2D rosenbrock for plotting
    f = lambda x,y: (x-1)**2 + 10*(y-x**2)**2
    X, Y = np.array(params["x0"]), np.array(params["x1"])
    X, Y = np.meshgrid(X,Y)
    Z = f(X,Y)

    print("Comparison with Random Grid Search")
    rosenbrock = Rosenbrock(N,noise=0.5)
    params = {k:[i*4/500 - 2 for i in range(501)] for k in [f"x{j}" for j in range(N)]}
    # DGD
    print(f"Start DGD N = {n}")
    dgd_argmins, dgd_mins = [], []
    for i in range(5):
        dgd = DistributedGridDescent(rosenbrock, params, metric, n_iter=N_ITER,
            n_jobs=4, n_random_init=32, verbose=0)
        dgd.run()
        dgd_argmins.append(dgd.best_params_)
        dgd_mins.append(-1*dgd.best_score_)
    dgd_argmins = np.array([[bp[f"x{m}"] for m in range(N)] for bp in dgd_argmins])

    # Random Search
    print(f"Start Random Grid Search N = {n}")
    rnd_argmins, rnd_mins = [], []
    for i in range(5):
        x_star = None
        f_max = -2**32
        for j in range(N_ITER//4):
            x = {k:params[k][np.random.randint(0,501)] for k in [f"x{j}" for j in range(N)]}
            rosenbrock.build_model(**x)
            f_x = sum(rosenbrock.score() for _ in range(4))/4
            if f_x > f_max:
                f_max = f_x
                x_star = x
        rnd_argmins.append(x_star)
        rnd_mins.append(-1*f_max)
    rnd_argmins = np.array([[bp[f"x{m}"] for m in range(N)] for bp in rnd_argmins])
    print("Done.")


    im = ax[1].contour(X,Y,Z,levels=np.logspace(-1,2,25),alpha=0.5,zorder=0)
    ax[1].scatter(1,1,marker="x",color='red', label=r"Global optimum $f(x^*)$ = 0")
    ax[1].set_title("Minima of 5 independent runs\n"+r"Grid size: 500x500, n_iter=128, $\sigma$=0.5")
    ax[1].scatter(dgd_argmins[:,0],dgd_argmins[:,1],
        color="orange",edgecolor="k",zorder=1,
        label=r"DGD $\bar{f}(x)$ = "+f"{np.mean(dgd_mins):.3f}")
    ax[1].scatter(rnd_argmins[:,0],rnd_argmins[:,1],
        color="magenta",zorder=1,edgecolor="k",
        label=r"Random Grid Search $\bar{f}(x)$ = "+f"{np.mean(rnd_mins):.3f}")
    ax[1].set_xlabel("x0")
    ax[1].set_ylabel("x1")
    ax[1].legend(loc="lower left")

    plt.tight_layout()
    # plt.savefig("./scaling_analysis.png",dpi=100)
    plt.show()

if __name__ == '__main__':
    main()
