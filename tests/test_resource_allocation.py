"""
    File to test all 5 objectives from resource_allocation.py against
    Objective       Test1 Test2 Test3
    Overallocation   37    41    23
    Conflicts        8     5     2
    Undersupport     1     0     7
    Unwilling        53    58    43
    Unpreferred      15    19    10
    All passed!

    Author: Josue Antonio
    Date: 11/9/23
"""
import pytest
import numpy as np
from dstruct.resource_allocation import overallocation, read_csv_to_df, \
    unpreferred, min_under_support, unwilling, minimize_time_conflicts

@pytest.fixture
def objectives():
    # Read data
    test1_df = np.array(read_csv_to_df('../test1.csv', header=False))
    test2_df = np.array(read_csv_to_df('../test2.csv', header=False))
    test3_df = np.array(read_csv_to_df('../test3.csv', header=False))
    ta_df = read_csv_to_df('../tas.csv')
    sec_df = read_csv_to_df('../sections.csv')

    daytimes_arr = np.array(sec_df.daytime)

    # Create list with max assigned sections for all TAs
    allocations = list(ta_df.max_assigned)

    # Create list with the min number of TAs for all TAs
    under_allocation = list(sec_df.min_ta)

    return test1_df, test2_df, test3_df, ta_df, sec_df, daytimes_arr, allocations, under_allocation

def test_overallocation(objectives):
    test1_df, test2_df, test3_df, ta_df, sec_df, daytimes_arr, allocations, under_allocation = objectives
    overallo_list = [37, 41, 23]

    assert overallocation(test1_df, allocations) == overallo_list[0], "Overallocation didn't pass test1!"
    assert overallocation(test2_df, allocations) == overallo_list[1], "Overallocation didn't pass test2!"
    assert overallocation(test3_df, allocations) == overallo_list[2], "Overallocation didn't pass test3!"

def test_underpreferred(objectives):
    test1_df, test2_df, test3_df, ta_df, sec_df, daytimes_arr, allocations, under_allocation = objectives
    unpre_list = [15, 19, 10]
    ta_df = np.array(ta_df.iloc[:, 3:])

    assert unpreferred(test1_df, ta_df) == unpre_list[0], "Unpreferred didn't pass for test1!"
    assert unpreferred(test2_df, ta_df) == unpre_list[1], "Unpreferred didn't pass for test2!"
    assert unpreferred(test3_df, ta_df) == unpre_list[2], "Unpreferred didn't pass for test3!"

def test_min_under_support(objectives):
    test1_df, test2_df, test3_df, ta_df, sec_df, daytimes_arr, allocations, under_allocation = objectives
    undersupport_list = [1, 0, 7]

    assert min_under_support(test1_df, under_allocation) == undersupport_list[0], "Undersupport didn't pass for test1!"
    assert min_under_support(test2_df, under_allocation) == undersupport_list[1], "Undersupport didn't pass for test2!"
    assert min_under_support(test3_df, under_allocation) == undersupport_list[2], "Undersupport didn't pass for test3!"

def test_unwilling(objectives):
    test1_df, test2_df, test3_df, ta_df, sec_df, daytimes_arr, allocations, under_allocation = objectives
    ta_df = np.array(ta_df.iloc[:,3:])
    unwilling_list = [53, 58, 43]

    assert unwilling(test1_df, ta_df) == unwilling_list[0], "Unwilling didn't pass for test1!"
    assert unwilling(test2_df, ta_df) == unwilling_list[1], "Unwilling didn't pass for test2!"
    assert unwilling(test3_df, ta_df) == unwilling_list[2], "Unwilling didn't pass for test3!"

def test_minimize_time_conflicts(objectives):
    test1_df, test2_df, test3_df, ta_df, sec_df, daytimes_arr, allocations, under_allocation = objectives
    conflicts_list = [8, 5, 2]

    assert minimize_time_conflicts(test1_df, daytimes_arr) == conflicts_list[0], "Conflicts didn't pass for test1!"
    assert minimize_time_conflicts(test2_df, daytimes_arr) == conflicts_list[1], "Conflicts didn't pass for test2!"
    assert minimize_time_conflicts(test3_df, daytimes_arr) == conflicts_list[2], "Conflicts didn't pass for test3!"
