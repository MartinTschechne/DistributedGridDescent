import random
import time
from dgd import DGDLogger

class Rosenbrock:
    '''An esimator for the noisy Rosenbrock function..'''
    def __init__(self, dim, noise=.1):
        self.noise = noise
        self.dim = dim
        self.params = None

    def build_model(self, **params):
        '''Build the model.'''
        assert len(params) == self.dim
        self.params = params

    def score(self, data=None, cv_split=None):
        '''Evaluate function.'''
        param_list = [v for _,v in self.params.items()]
        time.sleep(1.)
        return -sum(10*(param_list[i+1] - param_list[i]**2)**2
                + (1. - param_list[i])**2 for i in range(self.dim-1)) + self.noise*random.gauss(0.,1.)

def main():
    DIM = 2
    N = 500
    rosenbrock = Rosenbrock(DIM)

    params = {k:[i*4/N - 2 for i in range(N+1)] for k in [f"x{j}" for j in range(DIM)]}

    # DistributedGridDescent
    metric = lambda run_set: sum(run_set)/len(run_set)
    dgd = DGDLogger("./test.json", params, metric, eps=lambda t:1/t, n_random_init=4, random_state=42)
    for _ in range(100):
        config = dgd.next_config()
        rosenbrock.build_model(**config)
        start = time.perf_counter()
        score = rosenbrock.score()
        end = time.perf_counter()
        dgd.log_results(config,score,end - start)

if __name__ == '__main__':
    main()
