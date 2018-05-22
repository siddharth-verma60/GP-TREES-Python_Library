import numpy as np

'''This module contains all the tasks related to creation of user defined functions.
Only the function create_function is available publically which can be used to make own function.
A dictionary of common functions (_common_functions) is created to have a pool of commonly used
functions to use them directly without creating one.'''

__all__ = ['create_function', 'get_arity']


# Class for declaring own functions
class _Function:

    def __init__(self, function, name, arity, ret_type=None):
        self.function = function
        # A callable function of the form function(a, *args)

        self.name = name
        # Alias name of the function as it would appear everywhere in
        # code and visualisation.

        self.arity = arity
        # No. of inputs of the function

        self.ret_type = ret_type


# This function is called when a new function is to be made.
def create_function(function, name, arity):
    if not isinstance(arity, int):
        raise ValueError('arity of the function %s should be int, got %s' % (name, type(arity)))

    return _Function(function, name, arity);


def _protectedDiv(left, right):
    try:
        # Converting the np primitives to ordinary primitives so as to avoid any numpy related error.
        left = np.float64(left).item()
        right = np.float64(right).item()

        return left / right
    except ZeroDivisionError:
        return 1


def _protectedSqrt(arg):
    return np.sqrt(np.abs(arg))


def get_arity(func):
    return _common_functions[func].arity


def get_function(func):
    return _common_functions[func].function


# Making some common function to be used in the tree. More functions can be created here.
add1 = create_function(np.add, "add", 2)
sub1 = create_function(np.subtract, "sub", 2)
mul1 = create_function(np.multiply, "mul", 2)
div1 = create_function(_protectedDiv, "divide", 2)
less1 = create_function(np.less, "less_than", 2)
great1 = create_function(np.greater, "greater_than", 2)
max1 = create_function(np.maximum, "max", 2)
min1 = create_function(np.minimum, "min", 2)
sin1 = create_function(np.sin, "sin", 1)
cos1 = create_function(np.cos, "cos", 1)
tan1 = create_function(np.tan, "tan", 1)
neg1 = create_function(np.negative, "negate", 1)
abs1 = create_function(np.abs, "absolute", 1)
sqrt1 = create_function(_protectedSqrt, "sq_root", 1)

_common_functions = {'add': add1,
                     'sub': sub1,
                     'mul': mul1,
                     'div': div1,
                     'less_than': less1,
                     'greater_than': great1,
                     'sqrt': sqrt1,
                     'max': max1,
                     'min': min1,
                     'sin': sin1,
                     'cos': cos1,
                     'tan': tan1,
                     'negate': neg1,
                     'absolute': abs1}



