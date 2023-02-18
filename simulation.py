import numpy as np  # Python 3.9.7, numpy version 1.23.1
import streamlit as st
import pandas as pd


class Simulation:
    def __init__(self, seed: int = None,
                 simulation_time: int = 60, simulation_size: int = 100,
                 initial_knowledge_asymmetry: int = 10, initial_economic_asymmetry: int=10,
                 **kwargs) -> None:

        if seed is not None:
            self.seed: int = seed
            np.random.seed(seed)

        # system parameters
        self.simulation_time = simulation_time
        self.simulation_size: int = simulation_size  # number of companies
        self.initial_knowledge_asymmetry = initial_knowledge_asymmetry
        self.initial_economic_asymmetry = initial_economic_asymmetry
        self._K: np.array = self.init_K()  # K: knowledge capital
        self._E: np.array = self.init_E()  # E: Economic capital

        self.time_step_s: list = []
        self.economic_gini_s: list = []  
        self.knowledge_gini_s: list = []
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

    def init_K(self) -> np.array:
        t = np.random.poisson(
            lam=self.initial_knowledge_asymmetry, size=self.simulation_size)
        return t

    def init_E(self) -> np.array:
        u = np.random.poisson(
            lam=self.initial_economic_asymmetry, size=self.simulation_size)
        return u

    def report(self, timestep):
        self.time_step_s.append(timestep)
        economic_gini = self.gini(self._E)
        knowledge_gini = self.gini(self._K)
        self.knowledge_gini_s.append(knowledge_gini)
        self.economic_gini_s.append(economic_gini)


    def step(self, timestep):
        self.report(timestep)  # report results

    def go(self, verbose=True):
        self.report(0)  # save initial values
        for t in range(self.simulation_time):
            if verbose:
                print(f"economic gini={self.economic_gini_s[-1]}, knowledge gini={self.knowledge_gini_s[-1]}")
            self.step(t+1)  # values after step t + 1


if __name__ == '__main__':
    # create a streamlit app to run the simulation
    # and display the results
    # add a selector bar to select the simulation parameters
    # and a button to run the simulation
    st.sidebar.title("Simulation parameters")
    simulation_time = st.sidebar.slider("Simulation time", 1, 100, 10)
    simulation_size = st.sidebar.slider("Simulation size", 1, 100, 10)
    # add a slider to select initial_knowledge_asymmetry
    initial_knowledge_asymmetry = st.sidebar.slider("Initial knowledge asymmetry", 0, 10, 1)
    # add a slider to select initial_economic_asymmetry
    initial_economic_asymmetry = st.sidebar.slider("Initial economic asymmetry", 0, 10, 1)
    # add a button to run the simulation
    if st.sidebar.button("Run simulation"):
        s = Simulation(simulation_time=simulation_time,
                    simulation_size=simulation_size)
        s.go(verbose=False)
        # create a streamlit app to visualize results

        # use a grid layout
        c = st.container()

        col1, col2 = c.columns(2)



        col1.subheader("Economic gini")
        col1.line_chart(s.economic_gini_s)#.title("Economic gini")
        # add a line chart to visualize knowledge gini
        # make the title of the chart "Knowledge gini"
        col2.subheader("Knowledge gini")
        col2.line_chart(s.knowledge_gini_s)#.title("Knowledge gini")
        # create a histogram to visualize the distribution of _E
        col1.subheader("Economic distribution")
        col1.bar_chart(s._E)#.title("Economic distribution")
        # create a histogram to visualize the distribution of _K
        col2.subheader("Knowledge distribution")
        col2.bar_chart(s._K)#.title("Knowledge distribution")



