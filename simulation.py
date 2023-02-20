import numpy as np  # Python 3.9.7, numpy version 1.23.1

class Simulation:
    def __init__(self, seed: int = None,
                 simulation_time: int = 60, simulation_size: int = 100,
                 initial_knowledge_location: int = 0, initial_economic_stock: int=100,
                 exploration_norms: int = 0,
                 **kwargs) -> None:

        if seed is not None:
            self.seed: int = seed
            np.random.seed(seed)

        # system parameters
        self.simulation_time = simulation_time
        self.simulation_size: int = simulation_size  # number of companies
        self.initial_knowledge_location = initial_knowledge_location
        self.initial_economic_stock = initial_economic_stock
        self._exploration_norms = exploration_norms

        self._S: np.array = self.init_S()  # S: status of each company alive/dead
        self._D: np.array = self._S * initial_knowledge_location # D: distnance on the knowledge landscape
        self._K: np.array = self.init_K()  # K: knowledge capital
        self._E: np.array = self.init_E()  # E: Economic capital

        self.time_step_s: list = []
        self.economic_gini_s: list = []  
        self.knowledge_gini_s: list = []
        self.companies_s : list = []
        # any extra parameters passed are stored
        for k, v in kwargs.items():
            exec(f'self.{k} = {v}')

    # helper functions
    def remove_diagonal(self, matrix) -> np.array:
        """zeros diagonals on the matrix since we don't count self-interaction"""
        mask = 1 - np.eye(self.simulation_size, dtype=np.int32)
        return mask * matrix

    def select_with_probability(self, matrix, probability):
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

    def a_vector_of_n_ones_in_m_zeros(self, n, m, random=True):
        v = np.concatenate([np.ones(n), np.zeros(m - n)])
        if random:
            np.random.shuffle(v)
        return v

    def select_k_ones(self, vector, k, random=True):
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
    def gini(self, x, w=None):
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

    def init_S(self) -> np.array:
        return np.ones(self.simulation_size, int)

    def init_K(self) -> np.array:
        return self._S * self.landscape(self._D)
        # t = np.random.poisson(
        #     lam=self.initial_knowledge_asymmetry, size=self.simulation_size)
        # return t

    def init_E(self) -> np.array:
        return self._S * self.initial_economic_stock
        # u = np.random.poisson(
        #     lam=self.initial_economic_asymmetry, size=self.simulation_size)
        # return u

    # simple fitness landscape, the further the distance the more the fitness
    # but there is a chance of landing worse than the starting point
    def landscape(self, distance):
        #1+Log[1+x]*Cos[x]
        return 1 + np.log(1 + distance) * np.cos(distance)
        #return 1 + distance * np.cos(distance)

    def report(self, timestep):
        self.time_step_s.append(timestep)
        economic_gini = self.gini(self._E)
        knowledge_gini = self.gini(self._K)
        self.knowledge_gini_s.append(knowledge_gini)
        self.economic_gini_s.append(economic_gini)
        self.companies_s.append(self._S.sum())

    def step(self, timestep):
        # first all companies decide to explore based on exploration norms
        distances = np.random.rayleigh(self._exploration_norms, self.simulation_size)
        # exploration requires economic resources
        self._E = self._E - distances
        # companies exausting their economic resources die
        self._S = 0 + self._E > 0
        # alive companies increase their knowledge based on distance
        new_knowledge = self.landscape(distances)
        # new_knowledge replaces _K if it is higher
        self._D = ((new_knowledge > self._K) * distances + (new_knowledge <= self._K) * self._D) * self._S
        self._K = np.maximum(self._K, new_knowledge) * self._S

        # alive companies can increase their economic resources
        # https://en.wikipedia.org/wiki/Cobb%E2%80%93Douglas_production_function
        self._E = (self._E ** 1 )* (self._K ** 0.25) * self._S
        self.report(timestep)  # report results

    def go(self, verbose=True):
        self.report(0)  # report initial values
        for t in range(self.simulation_time):
            if verbose:
                print(f"companies={self.companies_s[-1]}, economic gini={self.economic_gini_s[-1]}, knowledge gini={self.knowledge_gini_s[-1]}")
            self.step(t+1)  # values after step t + 1

# one run to test code, to run multiple times check code in utilities.py
if __name__ == '__main__':
    s = Simulation()
    s.go(verbose=True)


