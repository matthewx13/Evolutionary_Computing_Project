import random as rnd

from dstruct.evo import Evo

def sumstepdown(L):  # Objective function
    """ Our measure of sortedness for a list of numbers """
    # zip(L, L[1:]) get consecutive pairs from list
    return sum([x - y for x, y in zip(L, L[1:]) if y < x])

def sumratio(L):
    """ Ratio of sum of first-half to 2nd-half values """
    sz = len(L)
    return round(sum(L[:sz//2]) / sum(L[sz//2 + 1:]), 5)

def swapper(solutions):  # Agent to modify existing solution
    L = solutions[0]
    i = rnd.randrange(0, len(L))
    j = rnd.randrange(0, len(L))
    L[i], L[j] = L[j], L[i]
    return L
def main():

    # Creating an instance of the framework
    E = Evo()

    # Register all objectives (fitness criteria)
    E.add_fitness_criteria('ssd', sumstepdown)
    E.add_fitness_criteria('sumratio', sumratio)

    # Register all the agents
    E.add_agent('swapper', swapper, 1)

    # Initialize the population
    N = 30
    L = [rnd.randrange(1, 99) for _ in range(N)]
    E.add_solution(L)

    # Display population summary
    print(E)

    # Run the solver
    E.evolve(n=10000, dom=100)

    # Display final population
    print(E)

main()
