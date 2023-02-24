import numpy as np

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