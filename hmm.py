import numpy as np

class HMM:

    def __init__(self, num_states):

        self.D = 0 # num of unique observations
        self.L = num_states # num of hidden states

        self.token_dict = {} # map of integers to tokens

        self.A = None # transition (row: from; col: to), 0-indexed
        self.PI = None # initial state distribution, 0-indexed
        self.O = None # observation (row: state; col: observation), 0-indexed

    def train(self, data, epsilon=0.001, scaling=True):
        X = self.registerObs(data)

        L = self.L
        D = self.D

        # Initialize Matrices
        self.A = self.normalize(np.random.rand(L, L))
        self.PI = self.normalize(np.random.rand(L))
        self.O = self.normalize(np.random.rand(L, D))

        norm_arr = []
        iterations = 0

        while (True):
            iterations += 1
            # E Step
            gamma_arr, xi_arr = self.computeMarginals(X, scaling)

            # M step (Computes marginals + Updates)
            change_norm = self.update(X, gamma_arr, xi_arr)
            print change_norm
            norm_arr.append(change_norm)

            # Stopping Condition
            if len(norm_arr) > 1 and norm_arr[-1] / norm_arr[0] < epsilon:
                print "iterations:", iterations
                break

        print self.PI
        print self.A
        print self.O

        return (self.token_dict, self.PI, self.A, self.O)

    """ Registers observations as integers and returns data transformed into
    integers. """
    def registerObs(self, data):
        # Reset Variables
        self.D = 0
        token_to_num = {}

        X = [] # data transformed into integers corresponding to tokens
        for seq in data:
            X_i = [] # this sequence transformed into integers
            for token in seq:
                if token not in token_to_num:
                    token_to_num[token] = self.D
                    self.D += 1
                X_i.append(token_to_num[token])
            X.append(X_i)

        self.token_dict = {v: k for k, v in token_to_num.items()}

        return X

    """ Makes all rows add up to 1 """
    @staticmethod
    def normalize(matrix):
        if len(matrix.shape) == 1:
            return matrix / matrix.sum()
        sums = matrix.sum(axis=1)
        return matrix / sums.reshape(sums.shape[0], 1)

    def forwardBackward(self, seq, scaling=True):
        """ This function computes alpha and beta values for a sequence
            using the Forward-Backward algorithm.
        """
        M = len(seq) # length of given sequence
        L = self.L # num of states

        alphas = np.zeros((M, L)) # row: position; col: state
        betas = np.zeros((M, L)) # row: position; col: state

        # FORWARD ALGORITHM
        for i in range(M): # For each observation
            for s in range(L): # For each state
                # Base case
                if i == 0:
                    alphas[i, s] = self.O[s, seq[0]] * self.PI[s]
                else:
                    sum = 0
                    # For each previous state
                    for prev in range(L):
                        sum += alphas[i-1, prev] * self.A[prev, s]
                    alphas[i, s] = sum * self.O[s, seq[i]]
            # Scaling
            if scaling:
                scale = np.sum(alphas[i])
                alphas[i] = alphas[i] / scale

        # BACKWARD ALGORITHM
        for i in reversed(range(M)): # For each observation
            for s in range(L): # For each state
                # Base case
                if i == M-1:
                    betas[i, s] = 1
                else:
                    # For each next state
                    for next in range(L):
                        betas[i, s] += betas[i+1, next] * \
                                       self.A[s, next] * self.O[next, seq[i+1]]
            # Scaling
            if scaling:
                scale = np.sum(betas[i])
                betas[i] = betas[i] / scale

        return (alphas, betas)

    def computeMarginals(self, X, scaling=True):
        # Calculate alphas and betas for all sequences
        L = self.L
        alphas_arr = []
        betas_arr = []
        for seq in X:
            alphas, betas = self.forwardBackward(seq, scaling)
            alphas_arr.append(alphas)
            betas_arr.append(betas)
        # P(y_i = z)
        gamma_arr = [] # Indexed by: # Sequence, Position, State

        # P(y_i = prev, y_i+1 = next)
        xi_arr = [] # Indexed by: # Sequence, Position of Prev, Prev, Next

        # Compute Gammas
        for j in range(len(X)): # iterate over all sequences
            seq_len = len(X[j])
            alphas = alphas_arr[j]
            betas = betas_arr[j]

            # gammas for this sequence
            gamma = np.zeros((seq_len, L))

            for i in range(seq_len):
                for state in range(L):
                    # just numerator
                    gamma[i, state] = alphas[i, state] * betas[i, state]
                # divide by denominator
                gamma[i] = gamma[i] / gamma[i].sum()

            gamma_arr.append(gamma)

        # Compute Xi's
        for j in range(len(X)): # iterate over all sequences
            seq = X[j]
            seq_len = len(seq)
            alphas = alphas_arr[j]
            betas = betas_arr[j]

            # xi's for this sequence
            xi = np.zeros((seq_len-1, L, L))

            for i in range(seq_len-1):
                for prev in range(L):
                    for next in range(L):
                        # just numerator
                        xi[i, prev, next] = alphas[i, prev] * \
                                            self.O[next, seq[i+1]] * \
                                            self.A[prev, next] * \
                                            betas[i+1, next]
                # divide by denominator
                xi[i] = xi[i] / xi[i].sum()

            xi_arr.append(xi)

        return (gamma_arr, xi_arr)

    def update(self, X, gamma_arr, xi_arr):
        L = self.L # num states
        D = self.D # num unique tokens

        # new matrices
        PI = np.zeros(self.PI.shape)
        A = np.zeros(self.A.shape)
        O = np.zeros(self.O.shape)

        # update PI (initial distribution matrix)
        for state in range(L):
            prob_sum = 0
            for j in range(len(X)): # iterate over all sequences
                prob_sum += gamma_arr[j][0, state]

            PI[state] = prob_sum / len(X) # average across sequences

        # Make sure numbers add up to 1
        np.testing.assert_almost_equal(PI.sum(), 1)

        # Update A (transition matrix)
        for prev in range(L):
            for next in range(L):
                numerator = 0
                denominator = 0

                for j in range(len(X)): # iterate over all sequences
                    # for each index in seq excluding last index
                    for i in range(len(X[j])-1):

                        numerator += xi_arr[j][i, prev, next]
                        denominator += gamma_arr[j][i, prev]

                # UPDATE A_{prev, next}
                A[prev, next] = numerator / denominator

            # Make sure rows add up to 1
            np.testing.assert_almost_equal(A[prev].sum(), 1)

        # update O (emission matrix)
        for state in range(L):
            for token in range(D):
                numerator = 0
                denominator = 0

                for j in range(len(X)): # iterate over all sequences
                    for i in range(len(X[j])): # for each index in seq
                        prob = gamma_arr[j][i, state]

                        if X[j][i] == token: # indicator function
                            numerator += prob
                        denominator += prob

                O[state, token] = numerator / denominator

            # Make sure rows add up to 1
            np.testing.assert_almost_equal(O[state].sum(), 1)

        # frobenius norm of the differences between update and previous matrices
        change_norm = np.linalg.norm(self.A - A) + np.linalg.norm(self.O - O) \
                      + np.linalg.norm(self.PI - PI)

        # update matrices
        self.O = O
        self.PI = PI
        self.A = A

        return change_norm


if __name__ == '__main':
    # Testing
    testing = True
    # Rochester example
    if testing:
        h = HMM(2)
        print "\n"
        data = [['R', 'W', 'B', 'B']]
        h.train(data)
    """
    # For testing, use these initial matrices (for Rochester example)
    self.A = np.array([[.6, .4], [.3, .7]])
    self.PI = np.array([.8, .2])
    self.O = np.array([[.3, .4, .3], [.4, .3, .3]])
    print "A:\n", self.A
    print "O:\n", self.O """
