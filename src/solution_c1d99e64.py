"""NUI Galway CT5132/CT5148 Programming and Tools for AI (James McDermott)

Solution for Assignment 3: File c1d99e64.json

Student name(s): Ian Matthews
Student ID(s):   12100610

"""
import numpy as np
import sys
from common_utils import load_file, print_grid


def solve(input_grid):
    """
    Given the input grid from any training or evaluation pair in the input json file,
    solve should return the correct output grid in the same format.
    Allowed formats are : 1. a JSON string containing a list of lists; or 2. a Python list of lists;
    or 3. a Numpy 2D array of type int
    :param input_grid: the input gird to be transformed
    :return: the modified grid
    >>> ia = [[4, 4, 4, 0, 2, 4],[4, 4, 4, 0, 0, 4],[0, 0, 0, 0, 0, 0],[4, 0, 4, 0, 0, 2]]
    >>> solve(ia)
    array([[4, 4, 4, 2, 2, 4],
           [4, 4, 4, 2, 0, 4],
           [2, 2, 2, 2, 2, 2],
           [4, 0, 4, 2, 0, 2]])
    """
    grid = np.asarray(input_grid)
    # find the rows and which are all dark, will colour these red
    rows_to_colour = [row for row in range(grid.shape[0]) if np.all(grid[row, :] == 0)]

    # colour in the cols
    for j in range(grid.shape[1]):
        if np.all(grid[:, j] == 0):
            grid[:, j] = 2
    # colour in rows
    for row in range(len(rows_to_colour)):
        grid[rows_to_colour[row], :] = 2

    return grid


def main():
    """
    Main method, reads in file specified file from the command line,
    calls the solve function to generate output
    """
    try:
        inputs = load_file(sys.argv[1])
        for grid in inputs:
            output = solve(grid)
            print_grid(output)
    except IndexError:
        print("please enter input file")


if __name__ == "__main__":
    import doctest
    doctest.testmod()
    main()
