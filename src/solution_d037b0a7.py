"""NUI Galway CT5132/CT5148 Programming and Tools for AI (James McDermott)

Solution for Assignment 3: File d037b0a7.json

Student name(s): Ian Matthews
Student ID(s):   12100610

"""
import numpy as np
import sys
from common_utils import load_file, print_grid


def fill_grid_columns(row_coords, grid):
    """
    :param row_coords: Get the height of the row we want to fill
    :param grid: the input grid to source the colour
    :return: the modified input grid
    >>> rc = np.asarray([0, 2])
    >>> ip = np.asarray([[0, 0, 6], [0, 4, 0], [3, 0, 0]])
    >>> fill_grid_columns(rc, ip)
    array([[0, 0, 6],
           [0, 4, 6],
           [3, 0, 6]])
    """
    r = row_coords[0]
    c = row_coords[1]
    grid[r:3, c] = grid[r][c]
    return grid


def solve(input_grid):
    """
    Given the input grid from any training or evaluation pair in the input json file,
    solve should return the correct output grid in the same format.
    Allowed formats are : 1. a JSON string containing a list of lists; or 2. a Python list of lists;
    or 3. a Numpy 2D array of type int
    :param input_grid: the input gird to be transformed
    :return: the modified grid
    # >>> ig =  [[4, 0, 8], [0, 0, 0], [0, 7, 0]]
    # >>> solve(ig)
    array([[4, 0, 8],
           [4, 0, 8],
           [4, 7, 8]])
    """
    grid = np.asarray(input_grid)
    # get the positions of the coloured squares
    result = np.where(grid != 0)
    row_heights = np.asarray(list(zip(result[0], result[1])))

    # iterate over the columns, filling each column by the row range
    [fill_grid_columns(coords, grid) for coords in row_heights]

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
