import os
import numpy as np

path_city_a = "./data/city_A"
path_city_b = "./data/city_B"

a_data = np.load(os.path.join(path_city_a, "source.npy"))
a_labels = np.load(os.path.join(path_city_a, "source_labels.npy"))
b_data = np.load(os.path.join(path_city_b, "target.npy"))
b_labels = np.load(os.path.join(path_city_b, "target_labels.npy"))
test_data = np.load(os.path.join(path_city_b, "test.npy"))
test_labels = np.load(os.path.join(path_city_b, "test_labels.npy"))

# shape
print('shapes:')
print('a_data:', a_data.shape)
print('a_labels:', a_labels.shape)
print('b_data:', b_data.shape)
print('b_labels:', b_labels.shape)
print('test_data:', test_data.shape)
print('test_labels:', test_labels.shape)

# data
print(a_labels)