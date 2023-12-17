import time
from collections import defaultdict


def profile(f):
    """ Convenience function to make the decorator tag simpler
    @Profiler.profile --> @profile"""
    return Profiler.profile(f)
class Profiler:
    """ A code profiling class.
    Keep track of function class and running time."""

    calls = defaultdict(int) # func name --> call count (default = o)
    time = defaultdict(float) # func name --> total time (default = 0.0)

    @staticmethod
    def profile(f):
        """ The profiler decorator."""
        def wrapper(*args, **kwargs):
            function_name = str(f).split()[1]

            # start timer
            start = time.time_ns()

            # calling function
            val = f(*args, **kwargs)

            # compute elapsed time
            elapsed = (time.time_ns() - start) / 10**9

            # update calls and time dictionaries
            Profiler.calls[function_name] += 1
            Profiler.time[function_name] += elapsed

            return val

        return wrapper

    @staticmethod
    def report():
        """ Summarize # calls, total runtime, and time/call for each function"""
        print("Function              Calls    TotSec    Sec/Call")
        for name, calls in Profiler.calls.items():
            elapsed = Profiler.time[name]
            print(f'{name:20s} {calls:6d} {elapsed:10.6f} {elapsed / calls:10.6f}')
