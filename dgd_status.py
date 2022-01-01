from pprint import pprint

import pandas as pd

from dgd import DGDListener

def main():
    DIM = 2
    N = 500
    params = {k:[i*4/N - 2 for i in range(N+1)] for k in [f"x{j}" for j in range(DIM)]}

    # DistributedGridDescent
    metric = lambda run_set: sum(run_set)/len(run_set)
    dgd = DGDListener("./test.json", params, metric)
    df = pd.DataFrame(dgd.results_).set_index("ID").sort_values(["metric"],ascending=False)
    df.head()
    # pprint(dgd.ordered_results_)

if __name__ == '__main__':
    main()
