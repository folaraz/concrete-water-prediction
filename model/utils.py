import numpy as np
import csv

FILE_NAME = '/home/folaraz/PycharmProjects/conc_proj/data/concrete.csv'


class Predict():
    def __init__(self, stats):
        self.mean_cement = stats.get('mean_cement')
        self.std_cement = stats.get('std_cement')
        self.mean_fine = stats.get('mean_fine')
        self.std_fine = stats.get('std_fine')
        self.mean_coarse = stats.get('mean_coarse')
        self.std_coarse = stats.get('std_coarse')
        self.mean_age = stats.get('mean_age')
        self.std_age = stats.get('std_age')
        self.mean_strength = stats.get('mean_strength')
        self.std_strength = stats.get('std_strength')
        self.min_water = stats.get('min_water')
        self.max_water = stats.get('max_water')

    def input(self, cement, fine, coarse, age, strength):
        cement = (cement - self.mean_cement)/self.std_cement
        fine = (fine - self.mean_fine)/self.std_fine
        coarse = (coarse - self.mean_coarse)/self.std_coarse
        age = (age - self.mean_age)/self.std_age
        strength = (strength - self.mean_strength)/self.std_strength

        return np.matrix([[cement, fine, coarse, age, strength]])

    def output(self, water):
        return water * (self.max_water - self.min_water) + self.min_water


class Toapi():
    pass


def read_data(filename):
    """Reads the concrete.csv file
    under the data directory."""
    content = csv.reader(open(filename, 'r'))
    data = [line for line in content]
    data = data[1:]
    stats = dict()
    cement, stats['mean_cement'], stats['std_cement'] = normalize_others(np.array([line[0] for line in data]))
    fine_agg, stats['mean_fine'], stats['std_fine'] = normalize_others(np.array([line[1] for line in data]))
    coarse_agg, stats['mean_coarse'], stats['std_coarse'] = normalize_others(np.array([line[2] for line in data]))
    age, stats['mean_age'], stats['std_age'] = normalize_others(np.array([line[3] for line in data]))
    strength, stats['mean_strength'], stats['std_strength'] = normalize_others(np.array([line[4] for line in data]))
    water, stats['max_water'], stats['min_water'] = normalize_water(np.array([line[5] for line in data]))

    data = list(zip(cement, fine_agg, coarse_agg, age, strength, water))
    n_samples = len(data)
    data = np.asarray(data, dtype=np.float32)

    return data, n_samples, stats


def normalize_others(data):
    data = data.astype(float)
    mean_value = np.mean(data)
    std_value = np.std(data)
    norm = (data - mean_value)/std_value
    return norm, mean_value, std_value


def normalize_water(data):
    data = data.astype(float)
    max_val = np.max(data)
    min_val = np.min(data)
    norm = (data - min_val)/(max_val - min_val)
    return norm, max_val, min_val


def get_data(filename, train_size=0.9):
    # Read in Data and Split Data
    data, n_sample, stats = read_data(filename)
    train, test = data[:int(train_size * n_sample)], data[int(train_size * n_sample):]
    train_feat, train_lab = train[:, :-1], train[:, -1]
    test_feat, test_lab = test[:, :-1], test[:, -1]
    n_test = len(test_lab)

    return (train_feat, train_lab), (test_feat, test_lab), n_test, stats


train, test, n_test, stats = get_data(FILE_NAME)

predict = Predict(stats)

