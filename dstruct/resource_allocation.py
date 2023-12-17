import random as rnd
import pandas as pd
import numpy as np
from collections import Counter
from dstruct.evo import Evo
import time
from dstruct.profiler import Profiler, profile
import pickle

def read_csv_to_df(filename, header=True):
    """ Read csv and convert to dataframe
        Args: name of file (string), header boolean
        Returns: df from csv file
        Read csv and output as df by either including header row or not
    """
    df = pd.DataFrame(pd.read_csv(filename, header=None if not header else 0))
    return df

def create_initial_sol(sol_shape=(43, 17), assign_ta_prob=0.3):
    """ Generate random initial solution (np array)
        Args: shape of desired solution (num_rows, num_cols) tuple,
              probability that TA gets assigned to a section (float)
        Returns: initial solution as np array
        Initialize zero array and populate with 1's based on random
        probability check (assign_ta_prob)
    """
    sol = np.zeros(sol_shape)

    for row in range(sol.shape[0]):
        for num in range(sol.shape[1]):
            rnd_num = rnd.random()
            if rnd_num < assign_ta_prob:
                sol[row, num] = 1
    return sol

############################
# Objective Functions:
############################
@profile
def overallocation(sol, allocations):
    """ Objective function to compute overallocation penalty over all TAs
        Args: current solution (np array), list of max section # allocation
              for each TA
        Returns: total sum of overallocation penalty over all TAs
        Zip solution row with corresponding max allocation #, compute counter
        dict for each row and compare each max allocation # with the count for
        "allocated" (1.0) to get penalty, and sum all of these penalties
    """
    return sum(max(Counter(row)[1.0] - max_assigned, 0) for row, max_assigned in zip(sol, allocations))

@profile
def unpreferred(sol, ta_df):
    """ Objective function to compute number of "unpreferred" assigned sections
        Args: current solution (np array), subset of TA data corresponding to
              the rows/cols in solution (np array)
        Returns: total number of sections that were assigned to a TA that said
                 they're willing to TA that section, but that they don't prefer it
        Set condition that TA gets assigned to section AND that they said they're
        willing to teach said section
    """
    condition = (sol == 1) & (ta_df == "W")
    return np.sum(condition)

@profile
def min_under_support(sol, under_allocation):
    """ Objective Funtion to compute the underallocation of TAs
        Args: current solution (np array), list of min TA's needed
              for each section
        Returns: total sum of underallocation penalty over all TAs
        First create a list that includes the sum of all TAs in each column of the
        current solution (np array). Map a lambda equation that subtracts min TAs needed by
        the actual assigned ta to each class from the solution.
        """
    class_assigned_ta = list(np.sum(sol, axis=0))
    test = list(map(lambda x, y: y-x, class_assigned_ta, under_allocation))
    return sum(num for num in test if num > 0)

@profile
def unwilling(sol, ta_df):
    """ Objective Funtion to compute the underallocation of TAs
        Args: current solution (np array), list of min TA's needed
        for each section
        Returns: total sum of underallocation penalty over all TAs
        First create a list that includes the sum of all TAs in each column of the
        current solution (np array). Map a lambda equation that subtracts min TAs needed by
        the actual assigned ta to each class from the solution.
    """
    condition = (sol == 1) & (ta_df == "U")
    return np.sum(condition)

@profile
def minimize_time_conflicts(sol, daytimes_arr):
    """ Objective function to compute the time conflicts
        Args: current solution (np array), array of daytimes for each section
        Returns: total number of time conflicts
    """
    def has_time_conflict(row):
        assigned = [i for i, val in enumerate(row) if val == 1]
        pairs = [(x, y) for x in assigned for y in assigned if x < y]
        return any(daytimes_arr[x] == daytimes_arr[y] for (x, y) in pairs)

    conflicts = [has_time_conflict(row) for row in sol]

    total_conflicts = sum(conflicts)
    return total_conflicts

############################
# Agent Functions:
############################

@profile
def swapper(solutions):
    """ Agent: An agent to modify an existing solution """
    L = solutions[0]
    i = rnd.randrange(0, len(L))
    j = rnd.randrange(0, len(L))
    L[i], L[j] = L[j], L[i]
    return L

@profile
def inverse_rows(solutions):
    """ Agent: An agent to inverse the values of random rows in solutions """
    L = solutions[0].copy()

    # Randomly select rows to inverse
    rows_to_inverse = np.random.choice(L.shape[0], 5, replace=False)

    # Inverse the values of the selected rows
    L[rows_to_inverse, :] = 1 - L[rows_to_inverse, :]

    return L

@profile
def random_flipping(solutions):
    """ Agent: An agent to flip random values in solutions """
    L = solutions[0].copy()

    # Randomly flip a small percentage of elements in the solution matrix
    num_flips = int(np.ceil(0.1 * L.size))
    indices_to_flip = np.unravel_index(np.random.choice(L.size, num_flips, replace=False), L.shape)
    L[indices_to_flip] = 1 - L[indices_to_flip]

    return L

@profile
def random_assignment(sols):
    """Randomly assign TAs to sections."""
    sol = sols[0]
    num_tas, num_sections = sol.shape
    for i in range(num_tas):
        for j in range(num_sections):
            sol[i, j] = rnd.choice([0, 1])
    return sol

@profile
def minimize_unwilling(sols, ta_df):
    """Agent to minimize TA assignments where TAs are unwilling to support."""
    sol = sols[0]

    ta_rows, ta_cols = np.where(sol == 1)

    for row, col in zip(ta_rows, ta_cols):
        if ta_df.iloc[row, col + 3] == 'U':
            sol[row, col] = 0

    return sol


def main():

    # Read data
    ta_df = read_csv_to_df('../tas.csv')
    sec_df = read_csv_to_df('../sections.csv')

    daytimes_arr = np.array(sec_df.daytime)

    # Create list with max assigned sections for all TAs
    allocations = list(ta_df.max_assigned)

    # Create list with the min number of TAs for all TAs
    under_allocation = list(sec_df.min_ta)

    # Create subset of TA data corresponding to sections data
    ta_arr = np.array(ta_df.iloc[:, 3:])

    # Creating an instance of the framework
    E = Evo()

    # Register all objectives (fitness criteria)
    E.add_fitness_criteria('overallocation', overallocation, allocations)
    E.add_fitness_criteria('time_conflict', minimize_time_conflicts, daytimes_arr)
    E.add_fitness_criteria('undersupport', min_under_support, under_allocation)
    E.add_fitness_criteria('unwilling', unwilling, ta_arr)
    E.add_fitness_criteria('unpreferred', unpreferred, ta_arr)

    # Register all the agents
    E.add_agent('swapper', swapper, k=1)
    E.add_agent('inverse_rows', inverse_rows, k=1)
    E.add_agent('random_flipping', random_flipping, k=1)
    E.add_agent("minimize_unwilling", minimize_unwilling, ta_df, k=1)
    E.add_agent("minimize_unwilling", minimize_unwilling, ta_df, k=1)

    # Generate initial solution based on random probability check
    initial_sol = create_initial_sol()
    E.add_solution(initial_sol)

    # Display population summary
    # print(E)

    # Run the solver
    # E.evolve(time_limit=600)
    # E.create_sol_csv()

    # Display final population
    # print(E)
    # Profiler.report()

    # with open('sols_dict.pkl', "wb") as f:
    #     pickle.dump(E.pop, f)

    with open('sols_dict.pkl', "rb") as f:
        pop_dict = pickle.load(f)

    # Get 2d array for our best solution
    print(pop_dict[(('overallocation', 27), ('unpreferred', 27), ('undersupport', 1.0),
                    ('time_conflict', 3), ('unwilling', 0))])


if __name__ == "__main__":
    main()
