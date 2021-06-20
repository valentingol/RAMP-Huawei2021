import numpy as np
import rampwf as rw

problem = rw.utils.assert_read_problem()

X_train, y_train = problem.get_train_data()
X_test, y_test = problem.get_test_data()
