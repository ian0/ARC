"""NUI Galway CT5132/CT5148 Programming and Tools for AI (James McDermott)

Solution for Assignment 3: File ed36ccf7.json

Student name(s): Ian Matthews
Student ID(s):   12100610

"""
import numpy as np
import sys
from common_utils import load_file, print_grid


def solve(grid):
    """
    Given the input grid from any training or evaluation pair in the input json file,
    solve should return the correct output grid in the same format.
    Allowed formats are : 1. a JSON string containing a list of lists; or 2. a Python list of lists;
    or 3. a Numpy 2D array of type int
    :param grid: the input grid
    :return: the modified grid
    >>> ig =  [[0, 0, 0], [5, 0, 0], [0, 5, 5]]
    >>> solve(ig)
    array([[0, 0, 5],
           [0, 0, 5],
           [0, 5, 0]])
    """
    grid = np.asarray(grid)
    return np.rot90(grid)


def main():
    """
    Main method, reads in file specified file from the command line,
    calls the solve function to generate output
    """
    inputs = load_file(sys.argv[1])
    for grid in inputs:
        output = solve(grid)
        print_grid(output)


if __name__ == "__main__":
    import doctest

    doctest.testmod()
    main()
