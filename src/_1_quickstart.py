import mlflow
import os
from mlflow import log_metric, log_param, log_artifact

if __name__ == "__main__":
    # log a parameter (key-value pair)
    log_param("param1", 1)

    # log a metric
    log_metric('foo', 2)
    log_metric('foo', 5)
    log_metric('foo', 4)

    # log an artifact
    if not os.path.exists('outputs'):
        os.makedirs('outputs')
    with open('outputs/test.txt', 'w') as f:
        f.write('hello world!')

    log_artifact('outputs')
