from nk import NKLandscape, genSeqBits
from simulation import Simulation
import numpy as np
from utilities import Utilities

class SimulationNK(Simulation):
    def __init__(self, N : int = 5, K : int = 2, 
                exploration_norms : float = 0.1,
                exploration_effort: int = 10, 
                A : float = 1.0, **kwargs):
        self.N = N
        self.K = K
        self.A = A

        self._landscape = NKLandscape(N,K)
        super().__init__(**kwargs)

        self.exploration_norms = exploration_norms
        self.exploration_effort = exploration_effort
         
    def init_D(self) -> np.array:
        # convert initial knowledge location to a binary string
        coord = Utilities.int_to_bin(self.initial_knowledge_location, self.N)[:self.N]
        return np.array([np.array(list(coord)).astype(int) for _ in range(self.simulation_size)])

    def fitness(self, coordinates):
        return np.array([self._landscape.compFit(''.join(coord.astype(str))) for coord in coordinates])

    def produce(self):
        return  (self._E * self.A ) * (1 + self._K)

    def explore(self):
        # in probability proportional to exploration norms
        # choose a random bit to flip
        probablities = np.random.uniform(size=self.simulation_size)
        bits = np.random.randint(0, self.N, size=self.simulation_size)
        mask = probablities < self.exploration_norms
        coordinates = self._D.copy()
        coordinates[mask] = np.logical_xor(coordinates[mask], np.eye(self.N, dtype=int)[bits[mask]])
        efforts = np.ones(self.simulation_size) * self.exploration_effort * mask
        fitnesses = self.fitness(coordinates)
        return coordinates, efforts, fitnesses

class SimulationCos(Simulation):
    def __init__(self, alpha : float = 1.0, beta : float = 0.1, **kwargs):
        super().__init__(**kwargs)
        self.alpha = alpha
        self.beta = beta        

    # simple fitness landscape, the further the distance the more the fitness
    # but there is a chance of landing worse than the starting point
    def fitness(self, coordinates):
        return 1 + np.log(1 + coordinates) * np.cos(coordinates)

    # simple exploration
    def explore(self):
        distances = np.random.rayleigh(self.exploration_norms, self.simulation_size)
        efforts = distances
        coordinates = self._D + distances
        fitnesses = self.fitness(coordinates)
        return coordinates, efforts, fitnesses

    def produce(self):
        # https://en.wikipedia.org/wiki/Cobb%E2%80%93Douglas_production_function
        return (self._E ** self.alpha )* (self._K ** self.beta)

    def init_D(self) -> np.array:
        return self._S * self.initial_knowledge_location

# one run to test code, to run multiple times check code in utilities.py
if __name__ == '__main__':
    s = SimulationNK()
    s.go(verbose=True)