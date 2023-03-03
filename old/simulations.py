from nk import NKLandscape
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
        #coord = Utilities.int_to_bin(self.initial_knowledge_location, self.N)[:self.N]
        coord = self._landscape.minGene
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
    

    def step(self, timestep):
        # first all companies decide to explore based on exploration norms
        coordinates, efforts, fitnesses = self.explore()
        
        # only alive companies can explore
        mask1 = self._S == 1
        # companies need to have economic resources to explore
        mask2 = self._E > efforts
        self._E[mask1 & mask2] = self._E[mask1 & mask2] - efforts[mask1 & mask2]
        
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

class SimulationCos(Simulation):
    def __init__(self, initial_knowledge_location: int = 0, 
        alpha : float = 1.0, beta : float = 0.1, **kwargs):
        super().__init__(**kwargs)
        self.alpha = alpha
        self.beta = beta        
        self.initial_knowledge_location = initial_knowledge_location

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

    def step(self, timestep):
        # first all companies decide to explore based on exploration norms
        coordinates, efforts, fitnesses = self.explore()
        
        # exploration requires economic resources
        mask1 = self._S == 1
        #TODO still getting negative values here
        self._E[mask1] = np.maximum( self._E[mask1] - efforts[mask1], 0)
        
        # companies exausting their economic resources die
        self._S = 0 + self._E > 0
        # new_knowledge replaces _K if it is higher
        # select coordinates where fitness is higher than _K
        # and replace _D with coordinates
        # and replace _K with fitnesses
        mask2 = fitnesses > self._K
        self._D[mask1 & mask2] = coordinates[mask1 & mask2]
        self._K[mask1 & mask2] = fitnesses[mask1 & mask2]
        # alive companies can increase their economic resources
        self._E[mask1] = self.produce()[mask1] 
        self.report(timestep)  # report results

# one run to test code, to run multiple times check code in utilities.py
if __name__ == '__main__':
    s = SimulationNK()
    s.go(verbose=True)