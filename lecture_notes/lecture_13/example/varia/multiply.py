"""Compute various functions related to multiplication.
"""

def multiply(x,y):
    """Multiply two numbers.

    Parameters
    ----------
    x: number
        the first number to be multiplied
    y: number
        the second number to be multiplied

    Returns
    -------
    number
        the product of the numbers


    .. danger::
        Multiplying by zero is irreversible.
    """
    return x * y

def square(x):
    """Compute a square of a number.

    Parameters
    ----------
    x: number
        the number to be squared

    Returns
    -------
    number
        the square of `x` (i.e., \(x^2\))
    """
    return x * x

def cube(x):
    """Compute a cube of a number.

    Parameters
    ----------
    x: number
        the number to be squared

    Returns
    -------
    number
        the cube of `x` (i.e., \(x^3\))
    """
    return square(x) * x
