from nk import NKLandscape
import numpy as np
from utilities import Utilities
import random


class SimulationNK():
    def __init__(self, seed: int = None,
                 simulation_time: int = 50, simulation_size: int = 100,
                 initial_economic_stock: int = 100,
                 N: int = 5, K: int = 2,
                 exploration_norms: float = 0.1,
                 exploration_effort: int = 10,
                 A: float = 1.0, **kwargs):

        assert N > 0, 'N must be greater than 0'
        assert K > 0, 'K must be greater than 0'
        assert K < N, 'K must be less than N'

        if seed is not None:
            self.seed: int = seed
            np.random.seed(seed)
            random.seed(seed)

        self.N = N
        self.K = K
        self._landscape = NKLandscape(N, K)

        self.A = A
        self.simulation_time = simulation_time
        self.simulation_size: int = simulation_size  # number of companies
        self.initial_economic_stock = initial_economic_stock
        self.exploration_norms = exploration_norms
        self.exploration_effort = exploration_effort
        self._S: np.array = self.init_S()  # S: status of each company alive/dead
        self._D: np.array = self.init_D()  # D: coordinate on the knowledge landscape
        self._K: np.array = self.init_K()  # K: knowledge capital
        self._E: np.array = self.init_E()  # E: Economic capital

        self.time_step_s: list = []
        self.economic_gini_s: list = []
        self.knowledge_gini_s: list = []
        self.knowledge_median_s: list = []
        self.knowledge_q1_s: list = []
        self.knowledge_q3_s: list = []
        self.companies_s: list = []
        self.economic_median_s: list = []
        self.economic_q1_s: list = []
        self.economic_q3_s: list = []
        # any extra parameters passed are stored
        for k, v in kwargs.items():
            exec(f'self.{k} = {v}')

    # initialize all companies to alive
    def init_S(self) -> np.array:
        return np.ones(self.simulation_size, int)

    def init_E(self) -> np.array:
        return self._S * np.int64(self.initial_economic_stock)

    def init_K(self) -> np.array:
        return self._S * self.fitness(self._D)

    def init_D(self) -> np.array:
        # convert initial knowledge location to a binary string
        #coord = Utilities.int_to_bin(self.initial_knowledge_location, self.N)[:self.N]
        coord = self._landscape.minGene
        return np.array([np.array(list(coord)).astype(int) for _ in range(self.simulation_size)])

    def fitness(self, coordinates):
        return np.array([self._landscape.compFit(''.join(coord.astype(str))) for coord in coordinates])

    def produce(self):
        return (self._E * self.A) * (1 + self._K)

    def report(self, timestep):
        self.time_step_s.append(timestep)
        economic_gini = Utilities.gini(self._E)
        knowledge_gini = Utilities.gini(self._K)
        self.knowledge_gini_s.append(knowledge_gini)
        self.economic_gini_s.append(economic_gini)
        self.companies_s.append(self._S.sum())
        self.knowledge_median_s.append(np.median(self._K))
        self.knowledge_q1_s.append(np.quantile(self._K, 0.25))
        self.knowledge_q3_s.append(np.quantile(self._K, 0.75))
        self.economic_median_s.append(np.median(self._E))
        self.economic_q1_s.append(np.quantile(self._E, 0.25))
        self.economic_q3_s.append(np.quantile(self._E, 0.75))

    def explore(self):
        # in probability proportional to exploration norms
        # choose a random bit to flip
        probablities = np.random.uniform(size=self.simulation_size)
        bits = np.random.randint(0, self.N, size=self.simulation_size)
        mask = probablities < self.exploration_norms
        coordinates = self._D.copy()
        coordinates[mask] = np.logical_xor(
            coordinates[mask], np.eye(self.N, dtype=int)[bits[mask]])
        efforts = np.ones(self.simulation_size) * \
            self.exploration_effort * mask
        fitnesses = self.fitness(coordinates)
        return coordinates, efforts, fitnesses

    def step(self, timestep):
        # first all companies decide to explore based on exploration norms
        coordinates, efforts, fitnesses = self.explore()

        # only alive companies can explore
        mask1 = self._S == 1
        # companies need to have economic resources to explore
        mask2 = self._E > efforts
        self._E[mask1 & mask2] = self._E[mask1 &
                                         mask2] - efforts[mask1 & mask2]

        # companies exausting their economic resources die
        # self._S = 0 + self._E > 0
        # new_knowledge replaces _K if it is higher
        # select coordinates where fitness is higher than _K
        # and replace _D with coordinates
        # and replace _K with fitnesses
        mask3 = fitnesses > self._K
        self._D[mask1 & mask2 & mask3] = coordinates[mask1 & mask2 & mask3]
        self._K[mask1 & mask2 & mask3] = fitnesses[mask1 & mask2 & mask3]
        # alive companies can increase their economic resources
        self._E[mask1] = self.produce()[mask1]
        self.report(timestep)  # report results

    def go(self, verbose=True):
        self.report(0)  # report initial values
        for t in range(self.simulation_time):
            # if all companies are dead, stop
            if self._S.sum() == 0:
                break
            if verbose:
                print(
                    f"t={t}, companies={self.companies_s[-1]}, economic gini={self.economic_gini_s[-1]}, knowledge gini={self.knowledge_gini_s[-1]}")
            self.step(t+1)  # values after step t + 1


# one run to test code, to run multiple times check code in utilities.py
if __name__ == '__main__':
    s = SimulationNK(N=5, K=5)
    s.go(verbose=True)
