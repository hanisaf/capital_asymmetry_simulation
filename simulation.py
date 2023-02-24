from abc import abstractmethod, ABC
import numpy as np  # Python 3.9.7, numpy version 1.23.1
from utilities import Utilities

class Simulation(ABC):
    def __init__(self, seed: int = None,
                 simulation_time: int = 50, simulation_size: int = 100,
                 initial_knowledge_location: int = 0, initial_economic_stock: int=100,
                 exploration_norms: int = 5, alpha : float = 1.0, beta : float = 0.1,
                 **kwargs) -> None:

        if seed is not None:
            self.seed: int = seed
            np.random.seed(seed)

        # system parameters
        self.simulation_time = simulation_time
        self.simulation_size: int = simulation_size  # number of companies
        self.initial_knowledge_location = initial_knowledge_location
        self.initial_economic_stock = initial_economic_stock
        self.exploration_norms = exploration_norms
        self.alpha = alpha
        self.beta = beta

        self._S: np.array = self.init_S()  # S: status of each company alive/dead
        self._D: np.array = self.init_D() # D: distnance on the knowledge landscape
        self._K: np.array = self.init_K()  # K: knowledge capital
        self._E: np.array = self.init_E()  # E: Economic capital

        self.time_step_s: list = []
        self.economic_gini_s: list = []  
        self.knowledge_gini_s: list = []
        self.companies_s : list = []
        # any extra parameters passed are stored
        for k, v in kwargs.items():
            exec(f'self.{k} = {v}')

    def init_S(self) -> np.array:
        return np.ones(self.simulation_size, int)

    @abstractmethod
    def init_D(self) -> np.array:
        pass

    def init_K(self) -> np.array:
        return self._S * self.fitness(self._D)

    def init_E(self) -> np.array:
        return self._S * self.initial_economic_stock


    @abstractmethod
    def explore(self):
        pass

    @abstractmethod
    def fitness(self):
        pass

    def report(self, timestep):
        self.time_step_s.append(timestep)
        economic_gini = Utilities.gini(self._E)
        knowledge_gini = Utilities.gini(self._K)
        self.knowledge_gini_s.append(knowledge_gini)
        self.economic_gini_s.append(economic_gini)
        self.companies_s.append(self._S.sum())

    def step(self, timestep):
        # first all companies decide to explore based on exploration norms
        coordinates, efforts, fitnesses = self.explore()
        # exploration requires economic resources
        self._E = self._E - efforts
        # companies exausting their economic resources die
        self._S = 0 + self._E > 0
        # new_knowledge replaces _K if it is higher
        # select coordinates where fitness is higher than _K
        # and replace _D with coordinates
        # and replace _K with fitnesses
        mask = fitnesses > self._K
        self._D[mask] = coordinates[mask]
        self._K[mask] = fitnesses[mask]
        # alive companies can increase their economic resources
        # https://en.wikipedia.org/wiki/Cobb%E2%80%93Douglas_production_function
        self._E = (self._E ** self.alpha )* (self._K ** self.beta) * self._S
        self.report(timestep)  # report results

    def go(self, verbose=True):
        self.report(0)  # report initial values
        for t in range(self.simulation_time):
            if verbose:
                print(f"companies={self.companies_s[-1]}, economic gini={self.economic_gini_s[-1]}, knowledge gini={self.knowledge_gini_s[-1]}")
            self.step(t+1)  # values after step t + 1




