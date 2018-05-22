import numpy as np
from inspect import signature

'''This module contains the fitness metric to be used to calculate fitness. Till now, only MSE, Root MSE, and
Mean Absolute error have been defined as the metrics for calculating fitness. Further more number of functions to calculate fitness can be defined here.                               '''


# This object would be able to hold the information about the fitness metric of the tree.
class _Fitness:

    def __init__(self, function, sign):
        self.function = function  # callable
        # This function is of the form: func(y, y_pred, weights). Here 'y' is the vector containing the correct
        # labels available in the data. 'y_pred' is the vector that contains the output value calculated by the
        # GP tree. 'weights' denote the weight vector of the data.

        self.sign = sign  # +1 or -1
        # If 1, then the higher value of fitness denotes a good value and if -1 then a lower value is a good value for the fitness.


def make_fitness_metric(function, sign):
    # This factory function is the only public method of this module that creates the fitness object of the tree.
    # The fitness object contains the fitness metric used to calculate the fitness that determines the liklihood of
    # the tree to undergo various Genetic operations.

    # Check if sign is positive or negative.
    if not isinstance(sign, int):
        raise ValueError("Parameter: sign must be an int. Recieved %s" % type(sign))
    else:
        if sign < 0:
            sign = -1
        else:
            sign = 1

    # Check the number of arguments of the function.
    sig = signature(function)

    if len(sig.parameters) != 3:
        raise ValueError("There should be 3 parameters for the function call. Recieved %d." % len(sig.parameters))

    return _Fitness(function, sign)


def get_fitness_metric(fitness):
    return _common_fitness[fitness].function


def get_fitness_sign(fitness):
    return _common_fitness[fitness].sign


# -------------------------------------------------------------------------
#       Definition of some error metrics. More can be added here.
# -------------------------------------------------------------------------

def _root_mean_square_error(y, y_pred, w):
    # Calculating the root mean square error.
    return np.sqrt(np.average(((y_pred - y) ** 2), weights=w))


def _mean_absolute_error(y, y_pred, w):
    # Calculating the mean absolute error.
    return np.average(np.abs(y_pred - y), weights=w)


def _mean_square_error(y, y_pred, w):
    # Calculating the mean square error.
    return np.average(((y_pred - y) ** 2), weights=w)


root_mean_square_error = make_fitness_metric(_root_mean_square_error, -1)
mean_square_error = make_fitness_metric(_mean_square_error, -1)
mean_absolute_error = make_fitness_metric(_mean_absolute_error, -1)

_common_fitness = {"Root_Mean_Square_Error": root_mean_square_error,
                   "Mean_Square_Error": mean_square_error,
                   "Mean_Absolute_Error": mean_absolute_error}


