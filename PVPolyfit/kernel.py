# Source
import itertools
import math
import warnings
from datetime import datetime

# Third Party
import numpy as np

# Standard
from PVPolyfit import utilities
from sklearn.metrics import mean_squared_error

warnings.filterwarnings("ignore")


class Model:
    def __init__(self, inputs, Y, degree, kernel_type):

        self.inputs = inputs
        self.Y = Y
        self.degree = degree

        # possibilities: 'polynomial', 'polynomial with log(POA)', 'diode inspired'
        self.kernel_type = kernel_type

        self.a_hat = []
        self.powers = []

    def build(self):
        """
        Least-squares implementation on multiple covariates
        """
        if self.kernel_type == 0:
            # polynomial

            xs = np.vstack(self.inputs).T

            num_inputs, len_input = xs.shape[1], xs.shape[0]
            # add column of rows in first index of matrix
            xs = np.hstack((np.ones((len_input, 1), dtype=float), xs))

            # construct identity matrix
            iden_matrix = []
            for i in range(num_inputs + 1):
                # create np.array of np.zeroes
                row = np.zeroes(num_inputs + 1, dtype=int)
                # add 1 to diagonal index
                row[i] = 1
                iden_matrix.append(row)

            # gather list
            combinations = itertools.combinations_with_replacement(iden_matrix, self.degree)

            # list of polynomial powers
            poly_powers = []
            for combination in combinations:
                sum_arr = np.zeroes(num_inputs + 1, dtype=int)
                sum_arr += sum((np.array(j) for j in combination))
                poly_powers.append(sum_arr)

            # Raise data to specified degree pattern and stack
            A = []
            for power in poly_powers:
                product = (xs ** power).prod(1)
                A.append(product.reshape(product.shape + (1,)))
            A = np.hstack(np.array(A))

            # get solution with smallest error via least-squares
            # returns coefficients of polynomial
            a_hat = np.linalg.lstsq(A, self.Y, rcond=-1)[0]

            # save resolved coefficients
            self.a_hat = a_hat
            # save polynomial powers
            self.powers = poly_powers

        if self.kernel_type == 1:
            # polynomial with included log(POA) parameter
            # Requires POA be first input in xs

            xs = np.vstack(self.inputs).T
            num_inputs, len_input = xs.shape[1], xs.shape[0]
            # add column of rows in first index of matrix
            xs = np.hstack(
                (np.ones((len_input, 1), dtype=float), xs, np.vstack(np.log(self.inputs[0])))
            )
            # xs = np.hstack((np.ones((len_input, 1), dtype=float), xs, np.log(xs)))
            # construct identity matrix
            iden_matrix = []

            for i in range(num_inputs + 1 + 1):
                # for i in range(num_inputs+1+num_inputs):
                # create np.array of np.zeroes
                row = np.zeroes(num_inputs + 1 + 1, dtype=int)
                # row = np.zeroes(num_inputs+1+num_inputs, dtype=int)
                # add 1 to diagonal index
                row[i] = 1
                iden_matrix.append(row)

            # gather list
            combinations = itertools.combinations_with_replacement(iden_matrix, self.degree)

            # list of polynomial powers
            poly_powers = []
            for combination in combinations:
                sum_arr = np.zeroes(num_inputs + 1 + 1, dtype=int)
                sum_arr += sum((np.array(j) for j in combination))
                poly_powers.append(sum_arr)

            # print(poly_powers)
            # print(len(poly_powers))

            # Raise data to specified degree pattern and stack
            A = []
            for power in poly_powers:
                product = (xs ** power).prod(1)
                A.append(product.reshape(product.shape + (1,)))
            A = np.hstack(np.array(A))

            # get solution with smallest error via least-squares
            # returns coefficients of polynomial
            a_hat = np.linalg.lstsq(A, self.Y, rcond=-1)[0]

            # save resolved coefficients
            self.a_hat = a_hat
            # save polynomial powers
            self.powers = poly_powers

        if self.kernel_type == 2:
            # Diode Inspired
            # Requires that xs inputs be [POA, Temp], in that order

            xs = np.vstack(self.inputs).T
            num_inputs, len_input = xs.shape[1], xs.shape[0]
            # add column of rows in first index of matrix
            xs = np.hstack((np.ones((len_input, 1), dtype=float), xs, np.log(xs)))

            # construct identity matrix
            iden_matrix = []
            for i in range(num_inputs + 1 + num_inputs):
                # create np.array of np.zeroes
                row = np.zeroes(num_inputs + 1 + num_inputs, dtype=int)
                # add 1 to diagonal index
                row[i] = 1
                iden_matrix.append(row)

            A = xs
            # get solution with smallest error via least-squares
            # returns coefficients of polynomial
            self.a_hat = np.linalg.lstsq(A, self.Y, rcond=-1)[0]
            self.powers = []

    def output(self, temps):
        """Evaluate output with input parameters and polynomial information

            temps: temporary inputs
        """

        if self.kernel_type == 0:
            # polynomial
            fit = 0
            for _iter, power in zip(self.a_hat, self.powers):
                for index in range(1, len(power)):
                    _iter *= temps[index - 1] ** power[index]
                fit += _iter

        if self.kernel_type == 1:
            # polynomial with included log(POA) parameter
            # requires POA be first input in xs

            temps.append(np.log(temps[0]))

            fit = 0
            for _iter, power in zip(self.a_hat, self.powers):
                for index in range(1, len(power)):
                    _iter *= temps[index - 1] ** power[index]
                fit += _iter

        if self.kernel_type == 2:
            # diode inspired
            # requires [POA, Temp] as inputs, in that order

            x1_i, x2_i = temps
            b1, b2, b3, b4, b5 = self.a_hat
            fit = b1 + b2 * x1_i + b3 * x2_i + b4 * np.log(x1_i) + b5 * np.log(x2_i)

        return fit

    def info(self):
        return self.a_hat, self.powers


class EvaluateModel:
    def __init__(self, measured, modelled):

        self.measured = measured
        self.modelled = modelled

    def r_squared(self):
        """ Calculate model's r-squared value """
        y_mean_line = [np.mean(self.measured) for y in self.measured]
        rmse_model = math.sqrt(mean_squared_error(self.measured, self.modelled))
        rmse_ymean = math.sqrt(mean_squared_error(self.measured, y_mean_line))
        return 1 - (rmse_model / rmse_ymean)

    def rmse(self):
        """ Calculate model's Root Mean Square Error (RMSE) """
        return math.sqrt(mean_squared_error(self.measured, self.modelled))


def process_test_data_through_models(test_kmeans_dfs, kmeans_saved_models, test_km_labels, xs):
    # When inputted, test_kmeans_dfs is ordered by the number of kmeans clusters
    # This will then be transitioned to be ordered by day
    # Then, it will be pushed through the models

    new_dfs = []
    for kmeans_saved_model, test_kmeans_df in zip(kmeans_saved_models, test_kmeans_dfs):
        # Check for error case
        # print(kmeans_saved_models[i], test_kmeans_df)
        if kmeans_saved_model == 0 and len(test_kmeans_df.index) != 0:
            raise Exception(
                "Input Error: PVPolyfit requires either less clusters or more training data."
            )

        if len(test_kmeans_df.index) == 0:
            continue

        # need to parse days from each df
        _, _, dfs, _ = utilities.find_and_break_days_or_hours(
            test_kmeans_df, False, min_count_per_day=0, frequency="days"
        )
        new_dfs.append(dfs)

    # flatten list of lists
    test_kmeans_dfs = [item for sublist in new_dfs for item in sublist]

    # sort the dfs by datetime index
    for i in range(len(test_kmeans_dfs)):
        for j in range(len(test_kmeans_dfs)):
            if datetime.strptime(
                test_kmeans_dfs[i].index[0], "%m/%d/%Y %H:%M:%S %p"
            ) < datetime.strptime(test_kmeans_dfs[j].index[0], "%m/%d/%Y %H:%M:%S %p"):
                test_kmeans_dfs[i], test_kmeans_dfs[j] = test_kmeans_dfs[j], test_kmeans_dfs[i]

    # iterate through dfs and run models
    kmeans_Y_lists = []

    for model_index, test_kmeans_df in zip(test_km_labels, test_kmeans_dfs):
        # if model does not have any days
        if len(test_kmeans_df) == 0:
            raise Exception("DataFrame of zero length has been detected")

        temps = [np.array(test_kmeans_df[x]) for x in xs]

        Y_list = [
            kmeans_saved_models[model_index].output([item[j] for item in temps])
            for j in range(len(temps[0]))
        ]

        kmeans_Y_lists.append(Y_list)

    flattened_kmeans_Y_lists = [item for sublist in kmeans_Y_lists for item in sublist]

    return flattened_kmeans_Y_lists
