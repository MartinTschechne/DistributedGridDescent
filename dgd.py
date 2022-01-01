from copy import copy, deepcopy
from datetime import datetime
import hashlib
import json
import os
import time

from filelock import FileLock
from joblib import Parallel, delayed
import numpy as np

# Template dict for json logging file
RUNSETS = {
    "run_sets": {},
    "run_sets_metric": {},
    "best_set_id": {
        "id": None
        }
    }


class BaseDistributedGridDescent:
    """Base class for Distributed Grid Descent

    Base class for all DGD optimizers. This class implements the vasic DGD algorithm
    and further methods to handle and store configurations.

    Given a grid of considered hyperparameters `param_grid`, a specific set of
    hyperparameters `params` represented as a list of the indices of the specific
    hyperparametes within the grid called `config`. This config can be encoded to
    a unique string called `config_id` which will function as key for the logging
    file. The encoding uses the indices of the specific hyperparameter values
    inside the grid and concatenates them with '-'.

    Example:
    param_grid = {"learning_rate":[1e-2,3e-3,1e-3],"C":[1,0.1,0.01],"n_features":[10,20,42]}
    params = {"learning_rate": 3e-3, "C":0.01, "n_features":10}
    config = [2, 1, 0]
    config_id = "2-1-0"

    Attributes:
        param_grid:
            A parameter grid as dict.
        metric:
            A function returning a number to accumulate multiple scores of the
            same config.
        n_random_init:
            Number of randomly sampled configurations at the beginning. Helps to
            find good statring point.
        eps:
            Eps value for epislon greedy. Choose a random configuration with a
            certain probability.
        n_step_neighborhood:

        random_state:

    """
    def __init__(self,
                 param_grid,
                 metric, *,
                 n_random_init=1,
                 eps=0.,
                 n_step_neighborhood=1,
                 random_state=None):
        if not isinstance(param_grid, dict):
            raise TypeError(f"Parameter grid must be of type dict, "
                            f"but got {type(param_grid)}.")
        for key, val in param_grid.items():
            if not isinstance(val, list):
                raise TypeError(f"Parameter grid value is not a list, "
                                f"key: {key}, value: {type(val)}")
            if len(val) != len(set(val)):
                raise ValueError("Parameter grid values must be unique, "
                                 f"but there are only {len(set(val))} unique values "
                                 f"for parameter {key} with length {len(val)}.")
        self.param_grid = param_grid

        if not isinstance(n_random_init, int):
            raise TypeError("n_random_init must be of type int, "
                            f"but got type {type(n_random_init)}.")
        if n_random_init < 1:
            raise ValueError("n_random_init must be positive, "
                             f"but got n_random_init = {n_random_init}.")
        self.n_random_init = n_random_init

        if not callable(metric):
            raise TypeError("Evaluation metric must be a function.")
        self.metric = metric

        if not callable(eps):
            if not isinstance(eps, (int, float)):
                raise TypeError("eps must be of type float, "
                                f"but got type {type(eps)}.")
            if eps < 0 or eps > 1:
                raise ValueError("epsilon must be in (0,1), "
                                 f"but got eps = {eps}.")
            self.eps = lambda t: eps
        else:
            self.eps = eps

        if not isinstance(n_step_neighborhood, int):
            raise TypeError("n_step_neighborhood must be of type int, "
                            f"but got type {type(n_step_neighborhood)}.")
        if n_step_neighborhood < 1:
            raise ValueError("n_step_neighborhood must be positive, "
                             f"but got n_random_init = {n_step_neighborhood}.")
        self.n_step_neighborhood = n_step_neighborhood

        if random_state is not None and not isinstance(random_state,int):
            raise ValueError("random_state must be of type None or Integer"
                             f"but got {type(random_state)}.")
        self.rng = np.random.RandomState(random_state)


    def _next_config(self, run_sets, best_set_id):
        """Return next config to evaluate.

        If the number of evaluated configurations is less than `n_random_init`,
        return a random hyperparameter configuration. This helps to find a good
        starting point for DGD algorithm.
        Otherwise, calculate the 1-step neighborhood `B` of the current best hyper-
        parameter configuration according to the `metric` function defined by user.
        Calculate a ceiling `M` for the weighting, which is the largest number of
        evaluations of the 1-step neighborhood and add 1.
        Calculate the sampling weights for each configuration with M - number of
        evaluations of the configuration.
        Sample a config from the neighborhood and return it.

        Args:
            run_sets:
                The run sets object where the score map is stored.
            best_set_id:
                The unique ID of the current best configuration.
        """
        iteration = sum(len(run_sets[k]) for k in run_sets)
        if iteration < self.n_random_init or self.rng.rand() < self.eps(iteration):
            config = [self.rng.randint(0,len(self.param_grid[k])) for k in self.param_grid]
        else:
            # 1-step neighborhood
            B = self._neighborhood(best_set_id["id"])
            # n-step neighborhood
            for _ in range(1,self.n_step_neighborhood):
                new_B = []
                for b in B:
                    new_B += self._neighborhood(b)
                B += new_B
                B = list(set(B))
            M = max(len(run_sets.get(b,[])) for b in B) + 1
            W = np.array([(M - len(run_sets.get(b,[]))) for b in B])
            W = W/W.sum()
            config_id = self.rng.choice(B, p=W)
            config = self._id_to_config(config_id)
        return config


    def _log_results(self, config, score_map,
                     run_sets, run_sets_metric, best_set_id):
        """Update results.

        Args:
            config:
                A config
            score_map:
                A dict holding a score, execution time (optional) and a timestamp.
                For example:

                {"score": 0.9, "exec_time": 1.5, "timestamp": 1608921498.354126}

            run_sets:
                The run sets object where the score map is stored.
            run_sets_metric:
                The object which keeps track of the metric (e.g. mean, median)
                of all scores from a configuration.
            best_set_id:
                The unique ID of the current best configuration.
        """
        config_id = self._config_to_id(config)

        if config_id in run_sets:
            run_sets[config_id].append(score_map)
        else:
            run_sets[config_id] = [score_map]

        run_sets_metric[config_id] = self.metric([sm["score"] for sm in run_sets[config_id]])

        best_set_id["id"] = max(run_sets_metric, key=run_sets_metric.get)


    def _config_to_id(self,config):
        """Return the unique ID for a configuration.
        This unique ID is reversible, it is possible to restore the config from
        the ID.

        Example:
        >>> config = [1, 0, 3, 42]
        >>> dgd._config_to_id(config)
        "1-0-3-42"
        """
        return '-'.join([str(c) for c in config])


    def _id_to_config(self,config_id):
        """Return unique ID from a configuration. Reversible operation to
        restore the config from the ID.

        Example:
        >>> config_id = "42-13-17-0"
        >>> dgd._id_to_config(config_id)
        [42, 13, 17, 0]
        """
        return [int(s) for s in config_id.split('-')]


    def _config_to_values(self,config):
        """Return converted configuration from hyperparameter values.

        Example:
        >>> param_grid = {"learning_rate":[1e-2,3e-3,1e-3],"C":[1,0.1,0.01],"n_features":[10,20,42]}
        >>> config = [1, 2, 0]
        {"learning_rate": 3e-3, "C":0.01, "n_features":10}
        """
        return {k: self.param_grid[k][config[i]] for i, k in enumerate(self.param_grid)}


    def _values_to_config(self,values):
        """Return actual hyperparameter values form config.

        Example:
        >>> param_grid = {"learning_rate":[1e-2,3e-3,1e-3],"C":[1,0.1,0.01],"n_features":[10,20,42]}
        >>> values = {"learning_rate":1e-2, "C":0.1, "n_features":42}
        >>> dgd._values_to_config(values)
        [0, 1, 2]
        """
        # if statement to filter out keys which are not in the grid, e.g. "ID"
        return [self.param_grid[k].index(values[k]) for k in values if k in self.param_grid]


    def _config_to_hash(self,config):
        """Returns the hash value of a `config`.

        **This operation is not reversible!
        A config can not be recovered from it's hash.**

        The configuration will be encoded into a hex-number. Having a unique number
        for each hyperparameter set is useful for bookkeeping. self._config_to_id()
        has the disadvantage that it returns very long strings for high-dimensional
        parameter spaces. This function encodes the config into a 32-digit number,
        which is useful for meta files produced by training runs. The hash value can
        be included into the filename and it is possible to find all meta files for
        a specific set of hyperparameters.

        Example:
        >>> config = [1, 2, 3]
        >>> dgd._config_to_hash(config)
        '453e406dcee4d18174d4ff623f52dcd8'
        """
        m = hashlib.md5()
        m.update(str.encode(self._config_to_id(config)))
        return m.hexdigest()


    def _neighborhood(self, config_id):
        """Calculate the 1-step neighborhood of a hyperparameter configuration.

        Example:
        >>> param_grid = {"learning_rate":[1e-2,3e-3,1e-3],"C":[1,0.1,0.01],"n_features":[10,20,42]}
        >>> config_id = "0-1-2"
        >>> dgd._neighborhood(config_id)
        ["0-1-2","1-1-2","0-0-2","0-2-2","0-1-1"]
        """
        N = set([config_id])
        config = self._id_to_config(config_id)
        for i, k in enumerate(self.param_grid):
            new_config_up = copy(config)
            new_config_down = copy(config)
            new_config_up[i] = min(config[i]+1,len(self.param_grid[k])-1)
            new_config_down[i] = max(config[i]-1,0)
            N.add(self._config_to_id(new_config_up))
            N.add(self._config_to_id(new_config_down))
        return list(N)


    def _make_results(self,run_sets,run_sets_metric):
        """Returns a dict of all results which can be postprocessed, e.g. as pandas DataFrame.

        Args:
            run_sets:
            run_sets_metric:

        Returns:
            A dict containing the unique grid point ID, the actual parameters,
            the metric, the standard deviation of the metric, the number of
            evaluations and the mean execution time of a configuration.
        """
        results = []
        for config_id, score_maps in run_sets.items():
            config = self._id_to_config(config_id)
            values = self._config_to_values(config)
            num_evals = len(score_maps)
            metric_stddev = np.std([self.metric([sm["score"]]) for sm in score_maps])
            metric_score = run_sets_metric[config_id]
            mean_exec_time = np.mean([sm["exec_time"] for sm in score_maps])
            ID = self._config_to_hash(config)
            r = {"ID":ID, **values, "metric":metric_score,
                "metric_std":metric_stddev,"num_evals":num_evals,
                "mean_exec_time":mean_exec_time}

            # if the score map is a dict, return mean of subscores as well
            if isinstance(score_maps[0]["score"],dict):
                for k in score_maps[0]["score"]:
                    r[k] = np.mean([sm["score"][k] for sm in score_maps])

            results.append(r)

        return results


    def _order_results(self,run_sets):
        """Returns a dict of scores, config values and exectution times in chronological order.

        Args:
            run_sets:

        Returns:
            A dict containing scores, configurations and execution times in the
            chronological order they were obtained.
        """
        ordered_score_maps = []
        for config_id in run_sets:
            score_maps = [{**srs, "config": self._config_to_values(self._id_to_config(config_id))} for srs in run_sets[config_id]]
            ordered_score_maps += score_maps
        ordered_score_maps = sorted(ordered_score_maps,
                                    key=lambda sm: sm["timestamp"])

        return {"scores":[sm["score"] for sm in ordered_score_maps],
                "configs":[sm["config"] for sm in ordered_score_maps],
                "exec_times":[sm["exec_time"] for sm in ordered_score_maps]}


class DistributedGridDescent(BaseDistributedGridDescent):
    """Distributed Grid Descent class

    Longer Description.

    Attributes:
        model:
            A model to evaluate. Must implement model.build_model() and model.score()
        param_grid:
            A parameter grid as dict.
        metric:
            A function returning a number to accumulate multiple scores of the same config.
        cv_split:
            Fraction of the data used for validation.
        n_iter:
            Number of models to evaluate.
        n_jobs:
            Number of processes to use for tuning.
        n_random_init:
            Number of randomly sampled configurations at the beginning. Helps to find good statring point.
        verbose:
            Verbosity of the output.
        random_state:
            Number to fix randomness.
    """
    def __init__(self, model, *args,
                 cv_split=0.75,
                 n_iter=100,
                 n_jobs=-1,
                 verbose=1,
                 **kwargs):
        super().__init__(*args,**kwargs)

        if not hasattr(model,"build_model"):
            raise AttributeError("model object has no function 'build_model()'.")
        if not callable(model.build_model):
            raise TypeError(f"'build_model' must be a function, but got {type(model.build_model)}.")
        if not hasattr(model,"score"):
            raise AttributeError("model object has no function 'score()'.")
        if not callable(model.score):
            raise TypeError(f"'score()' must be a function, but got {type(model.score)}.")
        self.model = model

        if not isinstance(n_iter, int):
            raise TypeError(f"n_iter must be a of integer, but got {type(n_iter)}.")
        if n_iter <= 0:
            raise ValueError(f"n_iter must be positive, but got {n_iter}.")
        self.n_iter = n_iter

        if not isinstance(cv_split,float):
            raise TypeError(f"cv_split must be of type float, but got {type(cv_split)}.")
        if cv_split <= 0. or cv_split >= 1.:
            raise ValueError(f"cv_split must be in (0,1), but ot {cv_split}")
        self.cv_split = cv_split

        if not isinstance(n_jobs, int):
            raise ValueError("n_jobs must be of type integer.")
        if n_jobs < 0:
            self.n_jobs = os.cpu_count() + 1 + n_jobs
        else:
            self.n_jobs = n_jobs

        if not isinstance(verbose, int):
            raise TypeError(f"verbose must be a integer, but got {type(verbose)}.")
        if verbose < 0:
            raise ValueError(f"verbose must be positive, but got {verbose}.")
        self.verbose = verbose

        self._shared_run_sets = {}
        self._shared_run_sets_metric = {}
        self._shared_best_set_id = {"id":None}
        self._shared_counter = 0


    def run(self,data=None):
        """Execute optimization.

        Function to execute the DGD algorithm in parallel. Creates a Parallel
        object and distributes the work among workers. Finally accumulates all
        results and stores them internally.

        Args:
            data: An single object (dict, list, dataset instance, numpy array,
            etc.) containing the data to train the model on. Can be also None if
            no data is needed.
        """
        parallel = Parallel(n_jobs=self.n_jobs,require='sharedmem')
        # Start optimizing
        with parallel:
            out = parallel(
                delayed(self._step)(deepcopy(self.model),deepcopy(data)) for i in range(self.n_iter))

        # Done. Retrieve best parameters and corresponding score.
        self.results_ = self._make_results(self._shared_run_sets,
                                           self._shared_run_sets_metric)
        self.ordered_results_ = self._order_results(self._shared_run_sets)
        self.best_index_ = self._shared_best_set_id["id"]
        self.best_params_ = self._config_to_values(self._id_to_config(self._shared_best_set_id["id"]))
        self.best_score_ = self._shared_run_sets_metric[self._shared_best_set_id["id"]]


    def _step(self, model, data):
        """Scoring of a single configuration.

        Executes a single step of the DGD algorithm. Requests a new configuration,
        builds a model from the actual configuration parameters, scores the model
        on the data according to score function. Finally logs the score, execution
        time and a timestamp in shared log files.

        Args:
            model:
                An unfitted model.
            data:
                Data to score model on.
        """

        # select configuration
        config = self._next_config(self._shared_run_sets,
                                   self._shared_best_set_id)
        vals = self._config_to_values(config)

        # perform scoreing
        timestamp = datetime.timestamp(datetime.now())
        start = time.perf_counter()
        model.build_model(**vals)
        score = model.score(data,self.cv_split)
        end = time.perf_counter()
        exec_time = end - start

        # log score in run set
        score_map = {"score":score,"exec_time":exec_time,"timestamp":timestamp}
        self._log_results(config, score_map,
                          self._shared_run_sets,
                          self._shared_run_sets_metric,
                          self._shared_best_set_id)

        self._shared_counter += 1

        if self.verbose > 0:
            output = (f"Iteration: {self._shared_counter}/{self.n_iter}, "
                     f"Score: {self.metric([score]) :.3f}, Time: {exec_time :.3f} s")

            if self.verbose > 1:
                output += f"\nParameter: {vals}"

            print(output)


class DGDLogger(BaseDistributedGridDescent):
    """Logger class for the Distributed Grid Descent algorithm.

    Attributes:
        runset_path:
            Path to bookkeeping json-file.
        param_grid:
        metric:
        n_random_init:
        verbose:
        ranodm_state:
    """
    def __init__(self, runset_path, *args, **kwargs):
        super().__init__(*args, **kwargs)

        if not isinstance(runset_path,str):
            raise TypeError("runset_path must be of type string, "
                            f"but got type {type(runset_path)}.")
        self._runset_path = runset_path
        self._lockfile_path = self._runset_path + ".lock"
        self._init_lock = self._runset_path + ".init.lock"

        # check if the log-file already exists, if not create one.
        lock = FileLock(self._init_lock)
        with lock:
            if not os.path.exists(self._runset_path):
                with open(self._runset_path,"w") as file:
                    json.dump(RUNSETS, file)


    def next_config(self,):
        """Public next_config for cluster usage."""
        lock = FileLock(self._lockfile_path)

        with lock:
            with open(self._runset_path,"r") as file:
                run_sets_dict = json.loads(file.read())

            config = self._next_config(run_sets_dict["run_sets"],
                                       run_sets_dict["best_set_id"])

        return self._config_to_values(config)


    def log_results(self, values, score, exec_time = None):
        """Public log_results function for cluster usage."""
        config = self._values_to_config(values)
        timestamp = datetime.timestamp(datetime.now())
        lock = FileLock(self._lockfile_path)

        with lock:
            with open(self._runset_path,"r") as file:
                run_sets_dict = json.loads(file.read())

                score_map = {"score":score,
                             "exec_time":exec_time if exec_time is not None else 0.,
                             "timestamp":timestamp}

                self._log_results(config, score_map,
                                  run_sets_dict["run_sets"],
                                  run_sets_dict["run_sets_metric"],
                                  run_sets_dict["best_set_id"])

            with open(self._runset_path, "w") as file:
                json.dump(run_sets_dict, file)


class DGDListener(BaseDistributedGridDescent):
    """Listener class for the Distributed Grid Descent algorithm.

    Attributes:
        runset_path:
            Path to bookkeeping json-file.
        param_grid:
        metric:
        n_random_init:
        verbose:
        ranodm_state:
    """
    def __init__(self, runset_path, *args, **kwargs):
        super().__init__(*args, **kwargs)

        if not isinstance(runset_path,str):
            raise TypeError("runset_path must be of type string, "
                            f"but got type {type(runset_path)}.")
        self._runset_path = runset_path
        self._lockfile_path = self._runset_path + ".lock"


    @property
    def results_(self,):
        """Returns current results from logging file."""
        lock = FileLock(self._lockfile_path)

        with lock:
            with open(self._runset_path,"r") as file:
                run_sets_dict = json.loads(file.read())

        return self._make_results(run_sets_dict["run_sets"],
                                  run_sets_dict["run_sets_metric"])


    @property
    def ordered_results_(self,):
        """Returns current results in chronological ordered from logging file."""
        lock = FileLock(self._lockfile_path)

        with lock:
            with open(self._runset_path,"r") as file:
                run_sets_dict = json.loads(file.read())

        return self._order_results(run_sets_dict["run_sets"])
