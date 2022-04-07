import ray
import math
import sympy
import numpy as np
import pandas as pd
from os.path import exists
from scipy.optimize import minimize
from scipy.special import legendre
import matplotlib.pyplot as plt

# Hyperparams
max_iter = 100
terminal_threshold = 0.001
step_size = 0.1
delta = 0.05
train_size = 300
normalize = True

# Constants
USE_SAVED_DATA = False
ROUND = 5
AXIS_COLUMN = 1
X0 = 'intensity'
X1 = 'symmetry'
TEST = True
TRAIN = False
D3_COLUMNS = ['bias', X0, X1]
L8_COLUMNS = [f'%s%s' % x for x in zip(['l']*45, np.arange(45))]
digit_columns = [str(x) for x in np.arange(256)]
TRAIN_FILE = 'data/ZipDigits.train'
TEST_FILE = 'data/ZipDigits.test'
LEGENDRE_FILE = 'data/L8_contour.csv'
FEATURE_FILE = 'data/HW9.csv'

# Terms should be a list of sympy expressions
# var_list should be a list of variable names
# values should be the values of said varaibles
@ray.remote
def compute_legendre(terms, values, var_list= ('x', 'y')):
    assert len(values) == len(var_list), 'compute_legendre, values and variables should be of same length'
    pairs = dict(zip(sympy.symbols(' '.join(var_list)), values))
    return [expr.subs(pairs) for expr in terms]

@ray.remote
def compute_dot_legendre(terms, values, res, var_list= ('x', 'y')):
    assert len(values) == len(var_list), 'compute_legendre, values and variables should be of same length'
    pairs = dict(zip(sympy.symbols(' '.join(var_list)), values))
    return np.dot(res, [expr.subs(pairs) for expr in terms])

def compute_H(X, lam):
    t = np.matmul(X.transpose(), X) + np.identity(45) * lam
    H = np.matmul(X, np.linalg.inv(t))
    H = np.matmul(H, X.transpose())
    return H

def compute_y_hat(H, y):
    y_hat = np.matmul(H, y)
    return y_hat

def pseudo_inv(X, y, reg=2):
    t = np.matmul(X.transpose(), X) + np.identity(45) * reg
    pseudo_inv = np.matmul(np.linalg.inv(t), X.transpose())
    w = np.matmul(pseudo_inv, y)
    return w

def compute_Etest(data, lam):
    X = data.get_x(False, use_legendre=True)
    Y = data.get_y(False)
    N = len(X)
    H = compute_H(X, lam)
    y_hat = compute_y_hat(H, Y)

    return sum([(y_hat[n] - Y[n]) ** 2 for n in range(N)])  / N

def compute_Ecv(data, l):
    X = data.get_x(False, use_legendre=True)
    Y = data.get_y(False)
    N = len(X)
    H = compute_H(X, l)

    y_hat = compute_y_hat(H, Y)

    return sum([((y_hat[n] - Y[n]) / (1 - H[n][n])) ** 2 for n in range(N)]) / N

def legendre_expansion(dimension, vars_list=('x', 'y')):
    assert len(vars_list) > 1, 'legendre_expansion requires at least two variables'
    assert dimension > 2, 'legendre_expansion dimension must be larger than 23'
    
    # Prepare dimension values
    variables = sympy.symbols(' '.join(vars_list))
    expression = sympy.expand((1 + sum(variables))**dimension)

    # Remove coefficients
    monom_list = sympy.Poly(expression, variables).monoms()
    legendre_terms = []

    for monom in monom_list:
        poly_list = [np.round(legendre(k), ROUND) for k in monom]
        terms = []
        for i in range(len(poly_list)):
            poly = poly_list[i]
            powers = variables[i] ** np.flip(np.arange(len(poly)))
            terms.append(sum([powers[n]*poly[n] for n in range(len(poly))]))

        legendre_terms.append(np.product(terms))

    # This will recombine everything without Legendre Polynomials
    # terms = [np.product([variables[n]**monom_list[i][n] for n in range(len(vars_list))]) for i in range(len(monom_list))]

    return legendre_terms
    # return np.concatenate([[legendre_terms[-1]], legendre_terms[:-1]])

def intensity(data):
    intensity = sum(data[x] for x in digit_columns) / 256
    if normalize:
        intensity = intensity - min(intensity)
        intensity = intensity / max(intensity)
        intensity = intensity * 2 - 1
    return intensity

def symmetry(data):
    result = []
    for index in data.index:
        digit = data.iloc[index][digit_columns].values.reshape((16,16))
        left = np.array([x[7::-1] for x in digit])
        right = np.array([x[8:] for x in digit])
        result.append(np.abs(left - right).mean())

    result = np.array(result)
    if normalize:
        result = result - min(result)
        result = result / max(result)
        result = result * 2 -1
    return pd.Series(result, index=data.index)

def calc_generalization_bound(N, M, delta):
    return np.sqrt(1/(2*N)*np.log(2*M/delta))

def calc_ein(w, x, y):
    assert len(x) == len(y)

    terms = []
    for n in range(len(x)):
        terms.append((np.dot(w, x[n]) - y[n]) ** 2)

    return sum(terms) / len(x)

class DataSet:
    ONES = 1
    OTHER = -1

    def __init__(self):
        self.train, self.test = self.load_data()

        # Dimension 3 variables
        self.model = np.zeros(45)
        self.h = []
        self.ein = None
        self.etest = None
        self.ein_bound = None
        self.etest_bound = None
        self.regularizer = None

    def load_data(self):
        if USE_SAVED_DATA:
            # Use the dataset saved from a previous run
            data = pd.read_csv(FEATURE_FILE)
        else:
            # Compute all data from scratch
            print('computing features...')
            raw_data_1 = pd.read_csv(TEST_FILE, header=None, sep=' ')
            raw_data_2 = pd.read_csv(TRAIN_FILE, header=None, sep=' ')
            raw_data = pd.concat([raw_data_1, raw_data_2], sort=False).reset_index()

            raw_data = raw_data.drop([257, 'index'], axis=AXIS_COLUMN)
            raw_data.columns = np.concatenate([['digit'], np.arange(256)])

            data = pd.DataFrame()
            data['intensity'] = intensity(raw_data)
            data['symmetry'] = symmetry(raw_data)
            data['y'] = raw_data['digit'].apply(lambda d: DataSet.ONES if d == 1 else DataSet.OTHER)
            data['bias'] = np.ones(len(raw_data))
            data.index = raw_data.index

            # Legendre
            # Use ray to do in parallel
            expression = legendre_expansion(8)
            values = np.array(ray.get([compute_legendre.remote(expression, row) for row in data[['intensity', 'symmetry']].values]), np.float64)

            print('done')

            legendre_df = pd.DataFrame(values, columns=L8_COLUMNS)
            data = pd.concat([data, legendre_df], axis=1)
            data.to_csv(FEATURE_FILE)

        train_indicies = np.random.choice(data.index, size=train_size)

        train = data.iloc[train_indicies]
        test = data.iloc[[x for x in data.index if x not in train_indicies]]
        return train, test 

    def get_one_rows(self, is_test):
        data = self.test if is_test else self.train
        return data[data['y'] == DataSet.ONES]

    def get_other_rows(self, is_test):
        data = self.test if is_test else self.train
        return data[data['y'] == DataSet.OTHER]

    def get_x(self, is_test, use_legendre=False):
        data = self.test if is_test else self.train
        columns = L8_COLUMNS if use_legendre else D3_COLUMNS
        return data[columns].values

    def get_y(self, is_test):
        data = self.test if is_test else self.train
        return data['y'].values

    def fit(self, regularizer=0):
        self.regularizer = regularizer
        model = pseudo_inv(self.get_x(False, use_legendre=True), self.get_y(False), reg=regularizer)
        self.model = model

        self.ein = calc_ein(self.model, self.get_x(TRAIN, use_legendre=True), self.get_y(TRAIN))
        self.etest = calc_ein(self.model, self.get_x(TEST, use_legendre=True), self.get_y(TEST))
        # self.ein_bound = calc_generalization_bound(len(self.train), len(h), delta)
        self.ein_bound = -1
        self.etest_bound = calc_generalization_bound(len(self.test), 1, delta)

        print('Linear regression Fit results:')
        print(f'Ein:    {self.ein:.4f} +/- {self.ein_bound:.4f}')
        print(f'Etest:  {self.etest:.4f} +/- {self.etest_bound:.4f}')

    def predict(self):
        predictions = [np.sign(np.dot(self.model, self.get_x(TEST)[n])) for n in range(len(self.test))]
        diff = predictions - self.get_y(TEST)
        accuracy = len(diff[diff == 0]) / len(diff)

        return predictions, accuracy

    def plot(self):
        if self.model is None:
            print('Warning: DataSet has not been fit!')

        if self.model is not None:
            X = np.linspace(-1, 1, 100)
            Y = np.linspace(-1, 1, 100)

            if USE_SAVED_DATA:
                _Z = np.loadtxt(LEGENDRE_FILE, delimiter=',')
            else:
                print('computing countour...')
                expression = legendre_expansion(8)
                _Z = np.array(ray.get([compute_legendre.remote(expression, (x, y)) for y in Y for x in X]))
                np.savetxt(LEGENDRE_FILE, _Z, delimiter=',')
                print('done')
            
            Z = np.array([np.dot(self.model, z) for z in _Z])

        fig, ax = plt.subplots()
        ax.set_title(f'train set, λ = %f' % self.regularizer)
        ax.set_xlabel('intensity')
        ax.set_ylabel('symmetry')
        ax.scatter(self.get_one_rows(TRAIN)[X0], self.get_one_rows(TRAIN)[X1], edgecolors='b', facecolors='none', label='ones')
        ax.scatter(self.get_other_rows(TRAIN)[X0], self.get_other_rows(TRAIN)[X1], marker='x', c='r', label='fives')
        if self.model is not None:
            ax.contour(X, Y, Z.reshape((len(X), len(Y))), levels=[0])

        ax.legend()
        plt.show(block=True)

        fig, ax = plt.subplots()
        ax.set_title(f'test set, λ = %f' % self.regularizer)
        ax.set_xlabel('intensity')
        ax.set_ylabel('symmetry')
        ax.scatter(self.get_one_rows(TEST)[X0], self.get_one_rows(TEST)[X1], edgecolors='b', facecolors='none', label='ones')
        ax.scatter(self.get_other_rows(TEST)[X0], self.get_other_rows(TEST)[X1], marker='x', c='r', label='fives')
        if self.model is not None:
            ax.contour(X, Y, Z.reshape((len(X), len(Y))), levels=[0])

        ax.legend()
        plt.show(block=True)

    # For testing the GD converges
    def plot_ein(self):
        fig, ax = plt.subplots()
        ax.plot([np.linalg.norm(x) for x in self.h])
        plt.show(block=True)

def plot_lambda_ecv(data):
    xs = np.linspace(0, 2, 100)[1:]
    Ecv = np.array([compute_Ecv(data, x) for x in xs])
    Etest = np.array([compute_Etest(data, x) for x in xs])

    fig, ax = plt.subplots()
    ax.set_title('λ vs Ecv')
    ax.set_xlabel('λ')
    ax.set_ylabel('Ecv')
    ax.plot(xs, Ecv, label='Ecv')
    ax.plot(xs, Etest, label='Etest')
    ax.plot(xs, (Ecv + Etest) / 2)
    ax.legend()

    plt.show(block=True)

def find_best_lambda(data):
    xs = np.linspace(0, 2, 100)[1:]
    Ecv = [compute_Ecv(data, x) for x in xs]

    return xs[np.argmin(Ecv)]

if  not exists(TRAIN_FILE) or not exists(TEST_FILE):
    print('Data files not present, quitting')
    quit()

print(f'Load precalculated features from file is set to %s' % USE_SAVED_DATA)
if not USE_SAVED_DATA:
    print('features will be calculated and saved, you can set this flag to True in later runs')

if USE_SAVED_DATA and (not exists(LEGENDRE_FILE) or not exists(FEATURE_FILE)):
    print('Precalculated data files are not present, quitting')
    quit()

# Load and prepare data
data = DataSet()
data.fit(regularizer=1.2)
data.plot()

# print(find_best_lambda(data))