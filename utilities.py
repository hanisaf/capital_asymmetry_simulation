import numpy as np
import itertools
import pandas as pd
import numpy as np
import inspect
import os
import concurrent.futures


def run_configuration(SimulationClass, conf, conf_number, seeds, verbose):
    results = []
    conf = dict(conf)
    run_number = 0
    for seed in seeds:
        try:
            sim = SimulationClass(seed=seed, **conf, configuration=conf_number, run=run_number)
        except AssertionError as e:
            if verbose:
                print(e)
                print(conf)
            continue
        sim.go(verbose=False)
        results.append(sim)
        run_number += 1
    if verbose: print('.', end='')
    conf_number += 1
    if conf_number % 10 == 0:
        if verbose: print(conf_number)    
    return results

def run_configurations(SimulationClass, parameters_ranges, runs_per_configuration = 100, auto_seed=True, seeds = None, parallel=False, verbose = True):
    configurations = [[(k, v) for v in parameters_ranges[k]] for k in parameters_ranges]
    configurations = list(itertools.product(*configurations))

    if auto_seed:        
        seeds = range(runs_per_configuration)
    else:
        assert len(seeds) == runs_per_configuration
    if verbose: print(f"number of configurations = {len(configurations)}")

    results = []
    if parallel:      
        cpu_count = os.cpu_count() - 1
        print(f'You have {cpu_count} CPUs that the simulation will use')       
        with concurrent.futures.ProcessPoolExecutor(cpu_count) as executor:
            futures = [executor.submit(run_configuration, SimulationClass,  conf, conf_number, seeds, verbose) for conf_number, conf in enumerate(configurations)]
        for f in concurrent.futures.as_completed(futures):
            results.extend(f.result())
        
    else:
        for conf_number, conf in enumerate(configurations):
            result = run_configuration(SimulationClass, conf, conf_number, seeds, verbose)
            results.extend(result)

    if verbose: 
        print('Done running the simulations!')
        print('Assembling the results ...')

    return results

def extract_history(simulation):
    attributes = inspect.getmembers(simulation, lambda a:not(inspect.isroutine(a)))
    public_attributes = [(a, v) for a, v in attributes if not a.startswith('_')]
    d = {}
    for a, v in public_attributes:
        a = a.replace('_', ' ').title() # format
        if a.endswith(' S'):
            a = a[:-2] # remove trailing s for singular noun
        if type(v) == type([]):
            d[a] = v # history attributes
        else:
            d[a] = (simulation.simulation_time + 1)* [v] # single value attributes, repeat per each step
    data = pd.DataFrame(d)
    return data

def round_float_values_in_data_frame(a_data_frame):
    for col, dtype in a_data_frame.dtypes.to_dict().items():
        if str(dtype) == 'float64':
            a_data_frame[col] = np.round(a_data_frame[col], decimals=5)    
    return a_data_frame

def create_result_table(list_of_simulations, history=True, round_values=True, agg_function='last'):
    data = pd.concat((extract_history(s) for s in list_of_simulations))
    data = data.sort_values(['Configuration', 'Run', 'Time Step'])
    if not history:
        data = data.groupby(['Configuration', 'Run']).aggregate(agg_function).sort_index().reset_index()
    if round_values: data = round_float_values_in_data_frame(data)
    return data


class SimulationRunner:
    def run_configuration(self, SimulationClass, conf, conf_number, seeds, verbose):
        results = []
        conf = dict(conf)
        run_number = 0
        for seed in seeds:
            sim = SimulationClass(seed=seed, **conf, configuration=conf_number, run=run_number)
            sim.go(verbose=False)
            results.append(sim)
            run_number += 1
        if verbose: print('.', end='')
        conf_number += 1
        if conf_number % 10 == 0:
            if verbose: print(conf_number)    
        return results

    def run_configurations(self, SimulationClass, parameters_ranges, runs_per_configuration = 100, auto_seed=True, seeds = None, parallel=False, verbose = True):
        configurations = [[(k, v) for v in parameters_ranges[k]] for k in parameters_ranges]
        configurations = list(itertools.product(*configurations))

        if auto_seed:        
            seeds = range(runs_per_configuration)
        else:
            assert len(seeds) == runs_per_configuration
        if verbose: print(f"number of configurations = {len(configurations)}")

        results = []
        if parallel:      
            cpu_count = os.cpu_count() - 1
            print(f'You have {cpu_count} CPUs that the simulation will use')       
            with concurrent.futures.ProcessPoolExecutor(cpu_count) as executor:
                futures = [executor.submit(self.run_configuration, SimulationClass,  conf, conf_number, seeds, verbose) for conf_number, conf in enumerate(configurations)]
            for f in concurrent.futures.as_completed(futures):
                results.extend(f.result())
            
        else:
            for conf_number, conf in enumerate(configurations):
                result = self.run_configuration(SimulationClass, conf, conf_number, seeds, verbose)
                results.extend(result)

        if verbose: 
            print('Done running the simulations!')
            print('Assembling the results ...')

        return results

    def extract_history(self,simulation):
        attributes = inspect.getmembers(simulation, lambda a:not(inspect.isroutine(a)))
        public_attributes = [(a, v) for a, v in attributes if not a.startswith('_')]
        d = {}
        for a, v in public_attributes:
            a = a.replace('_', ' ').title() # format
            if a.endswith(' S'):
                a = a[:-2] # remove trailing s for singular noun
            if type(v) == type([]):
                d[a] = v # history attributes
            else:
                d[a] = (simulation.simulation_time + 1)* [v] # single value attributes, repeat per each step
        data = pd.DataFrame(d)
        return data

    def round_float_values_in_data_frame(self, a_data_frame):
        for col, dtype in a_data_frame.dtypes.to_dict().items():
            if str(dtype) == 'float64':
                a_data_frame[col] = np.round(a_data_frame[col], decimals=5)    
        return a_data_frame

    def create_result_table(self, list_of_simulations, history=True, round_values=True, agg_function='last'):
        data = pd.concat((self.extract_history(s) for s in list_of_simulations))
        data = data.sort_values(['Configuration', 'Run', 'Time Step'])
        if not history:
            data = data.groupby(['Configuration', 'Run']).aggregate(agg_function).sort_index().reset_index()
        if round_values: data = self.round_float_values_in_data_frame(data)
        return data

class Utilities:
    # helper functions
    def remove_diagonal(matrix) -> np.array:
        """zeros diagonals on the matrix since we don't count self-interaction"""
        size = matrix.shape[0]
        mask = 1 - np.eye(size, dtype=np.int32)
        return mask * matrix

    def select_with_probability(matrix, probability):
        "select with probability elements of value 1 from a matrix of 0 & 1"
        assert(matrix.ndim in [1, 2])
        assert(set(np.unique(matrix)).issubset({0, 1}))
        res = matrix.copy()
        if matrix.ndim == 1:
            x = np.where(res == 1)[0]
            replace_v = np.random.choice(
                [0, 1], len(x), p=[1-probability, probability])
            res[x] = replace_v
        elif matrix.ndim == 2:
            x, y = np.where(res == 1)
            replace_v = np.random.choice(
                [0, 1], len(x), p=[1-probability, probability])
            res[x, y] = replace_v
        return res

    def a_vector_of_n_ones_in_m_zeros(n, m, random=True):
        v = np.concatenate([np.ones(n), np.zeros(m - n)])
        if random:
            np.random.shuffle(v)
        return v

    def select_k_ones(vector, k, random=True):
        "(randomly) select k elements of value 1 from a vector of 0 & 1"
        assert(vector.ndim in [1])
        assert(set(np.unique(vector)).issubset({0, 1}))
        assert(k <= len(vector))
        # find the index of the one values
        x = np.where(vector == 1)[0]
        if random:
            np.random.shuffle(x)
        # select the first k values
        x = x[:k]  # index of the one values
        new_v = np.zeros(len(vector))
        new_v[x] = 1
        return new_v
    # end of helper functions

    #https://stackoverflow.com/questions/48999542/more-efficient-weighted-gini-coefficient-in-python
    def gini(x, w=None):
        # modification, we are not interested in x of size 1
        if len(x) < 2:
            return np.nan    
        # The rest of the code requires numpy arrays.
        x = np.asarray(x)
        if w is not None:
            w = np.asarray(w)
            sorted_indices = np.argsort(x)
            sorted_x = x[sorted_indices]
            sorted_w = w[sorted_indices]
            # Force float dtype to avoid overflows
            cumw = np.cumsum(sorted_w, dtype=float)
            cumxw = np.cumsum(sorted_x * sorted_w, dtype=float)
            return (np.sum(cumxw[1:] * cumw[:-1] - cumxw[:-1] * cumw[1:]) / 
                    (cumxw[-1] * cumw[-1]))
        else:
            sorted_x = np.sort(x)
            n = len(x)
            cumx = np.cumsum(sorted_x, dtype=float)
            # The above formula, with all weights equal to 1 simplifies to:
            return (n + 1 - 2 * np.sum(cumx) / cumx[-1]) / n

    # convert an integer to a binary string
    def int_to_bin(n, length):
        bit = bin(n)
        bit = bit[2:]
        if len(bit) < length:
            bit = (length - len(bit))*'0' + bit
        return bit

# test code
if __name__ == '__main__':
    from simulations import SimulationNK # needed for testing
    params = {"N":[4,3], "K":[3,2]}
    SimulationClass = SimulationNK
    results = run_configurations(SimulationClass, params, parallel=False, runs_per_configuration=5)
    data = create_result_table(results)
    sim = results[0]
    print(data.head())
    print()