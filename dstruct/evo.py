import copy
import random as rnd
from functools import reduce
import pickle
import time
import pandas as pd
class Evo:

    def __init__(self):
        self.pop = {}  # eval -> solution   eval = ((name1, val1), (name2, val2), ...)
        self.fitness = {}  # name -> function
        self.agents = {}  # name -> (operator, num_sols_input)

    def add_fitness_criteria(self, name, f, *args):
        """ add a fitness criteria"""
        self.fitness[name] = f, *args

    def add_agent(self, name, op, *args, k=1):
        """ add an agent to the evolver with its operation and input arguments """
        self.agents[name] = (op, *args, k)

    def add_solution(self, sol):
        """ adding a solution to the population"""
        eval = tuple((name, f(sol, *args)) for name, (f, *args) in self.fitness.items())
        self.pop[eval] = sol

    def get_random_solutions(self, k=1):
        """ get random solutions from the current population """
        popvals = tuple(self.pop.values())
        return [copy.deepcopy(rnd.choice(popvals)) for _ in range(k)]

    def run_agent(self, name):
        """ run a specified agent to generate a new solution"""
        op, *args, k = self.agents[name]
        picks = self.get_random_solutions(k)
        new_solution = op(picks, *args)
        self.add_solution(new_solution)

    @staticmethod
    def _dominates(p, q):
        """ Return whether p dominates q """
        pscores = [score for _, score in p]
        qscores = [score for _, score in q]
        score_diffs = list(map(lambda x, y: y - x, pscores, qscores))
        min_diff = min(score_diffs)
        max_diff = max(score_diffs)
        return min_diff >= 0.0 and max_diff > 0.0

    @staticmethod
    def _reduce_nds(S, p):
        """ reduce the non-dominated solutions set S by removing dominated solutions"""
        return S - {q for q in S if Evo._dominates(p, q)}

    def remove_dominated(self):
        """ remove dominated solutions from the population """ 
        nds = reduce(Evo._reduce_nds, self.pop.keys(), self.pop.keys())
        self.pop = {k: self.pop[k] for k in nds}

        
    def evolve(self, n=1, time_limit=30):
        """
        Runs the evolver for n iterations to produce new solutions
        Args: n: Number of iterations 
              time_limit: Time to run evolve
        Return: None
        """

        # get agent names and start timer
        st = time.time()
        elapsed = 0

        agent_names = list(self.agents.keys())

        # run evolver until time limit is reached
        while elapsed < time_limit:
            for i in range(n):

                # create a rand numr to randomly get pop
                rand = rnd.random()

                # this will pick an agent
                pick = rnd.choice(agent_names)

                # run agent 
                self.run_agent(pick)

                # randomly get pop
                if rand * rand < 0.01:
                    self.remove_dominated()

                # print the status of the evolver
                print("Population size: ", len(self.pop))

                # compute the time difference, if less than time limit, continue running evolve
                et = time.time()
                if elapsed < time_limit:
                    elapsed = et - st

                else:
                    break

    
    def create_sol_csv(self):
        """
        This function creates a CSV file of the final solutions with the column
        names of "groupname", "overallocation", "conflicts", "undersupport", 
        "unwilling", "unpreferred"
        
        return: None
        """

        # Create list to store scores
        score_list = []

        for final_nds_pop, sol in self.pop.items():

            # Define group name for each NDS row
            inner_row = ["random"]

            for i, score in final_nds_pop:
                inner_row.append(score)

            # Add scores
            score_list.append(inner_row)

        # Convert to CSV file
        pd.DataFrame(score_list, columns=["groupname", "overallocation", "conflicts", "undersupport", "unwilling",
                                          "unpreferred"]).to_csv("final_solutions.csv")

    

    def __str__(self):
        """ Output the solutions in the population """
        rslt = ""
        for eval, sol in self.pop.items():
            rslt += str(dict(eval)) + ":\t" + str(sol) + "\n"
        return rslt
