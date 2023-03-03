from abc import abstractmethod, ABC
import numpy as np  # Python 3.9.7, numpy version 1.23.1
from utilities import Utilities
import random

class Simulation(ABC):
    def __init__(self, seed: int = None,
                 simulation_time: int = 50, simulation_size: int = 100,
                 initial_economic_stock: int=100,
                 exploration_norms: int = 5, **kwargs) -> None:

        if seed is not None:
            self.seed: int = seed
            np.random.seed(seed)
            random.seed(seed)

        # system parameters
        self.simulation_time = simulation_time
        self.simulation_size: int = simulation_size  # number of companies
        
        self.initial_economic_stock = initial_economic_stock
        self.exploration_norms = exploration_norms


        self._S: np.array = self.init_S()  # S: status of each company alive/dead
        self._D: np.array = self.init_D() # D: coordinate on the knowledge landscape
        self._K: np.array = self.init_K()  # K: knowledge capital
        self._E: np.array = self.init_E()  # E: Economic capital

        self.time_step_s: list = []
        self.economic_gini_s: list = []  
        self.knowledge_gini_s: list = []
        self.knowledge_median_s: list = []
        self.knowledge_q1_s: list = []
        self.knowledge_q3_s: list = []
        self.companies_s : list = []
        self.economic_median_s: list = []
        self.economic_q1_s: list = []
        self.economic_q3_s: list = []
        # any extra parameters passed are stored
        for k, v in kwargs.items():
            exec(f'self.{k} = {v}')

    # initialize all companies to alive
    def init_S(self) -> np.array:
        return np.ones(self.simulation_size, int)

    @abstractmethod
    def init_D(self) -> np.array:
        pass

    def init_K(self) -> np.array:
        return self._S * self.fitness(self._D)

    def init_E(self) -> np.array:
        return self._S * np.int64(self.initial_economic_stock)

    @abstractmethod
    def explore(self):
        pass

    @abstractmethod
    def fitness(self):
        pass

    @abstractmethod
    def produce(self):
        pass

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

    @abstractmethod
    def step(self, timestep):
        pass

    def go(self, verbose=True):
        self.report(0)  # report initial values
        for t in range(self.simulation_time):
            # if all companies are dead, stop
            if self._S.sum() == 0:
                break
            if verbose:
                print(f"t={t}, companies={self.companies_s[-1]}, economic gini={self.economic_gini_s[-1]}, knowledge gini={self.knowledge_gini_s[-1]}")
            self.step(t+1)  # values after step t + 1




