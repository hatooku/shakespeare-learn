import numpy as np

class HMM:

    def __init__(self, num_states):

        self.D = 0 # num of unique observations
        self.L = num_states # num of hidden states

        self.token_dict = {} # map of integers to tokens

        self.A = None # transition (row: from; col: to), 0-indexed
        self.PI = None # initial state distribution, 0-indexed
        self.O = None # observation (row: state; col: observation), 0-indexed

    def train(self, data, scaling=True):
        X = self.registerObs(data)

        # Initialize Matrices
        """
        self.A = self.normalize(np.random.rand(L, L))
        self.PI = self.normalize(np.random.rand(L))
        self.O = self.normalize(np.random.rand(L, D))
        """
        # TESTING
        self.A = np.array([[.6, .4], [.3, .7]])
        self.PI = np.array([.8, .2])
        self.O = np.array([[.3, .4, .3], [.4, .3, .3]])

        alphas_arr = []
        betas_arr = []
        for seq in X:
            alphas, betas = self.forwardBackward(seq, scaling)
            print "alphas\n", alphas.transpose()
            print "betas\n", betas.transpose()
            alphas_arr.append(alphas)
            betas_arr.append(betas)
        #self.update(X, alphas_arr, betas_arr)
        #print self.A

        return

    """ Registers observations as integers and returns data transformed into
    integers. """
    def registerObs(self, data):
        # Reset Variables
        self.D = 0
        self.token_dict = {}

        X = [] # data transformed into integers corresponding to tokens
        for seq in data:
            X_i = [] # this sequence transformed into integers
            for token in seq:
                if token not in self.token_dict:
                    self.token_dict[token] = self.D
                    self.D += 1
                X_i.append(self.token_dict[token])
            X.append(X_i)
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

    def update(self, X, alphas_arr, betas_arr):
        L = self.L # num states

        # Update A (transition matrix)
        for prev in range(L):
            for next in range(L):
                numerator = 0
                denominator = 0

                for j in range(len(X)): # iterate over all sequences
                    seq = X[j]
                    alphas = alphas_arr[j]
                    betas = betas_arr[j]
                    for i in range(len(seq)): # for each index in seq
                        # Compute P(y_i = prev, y_i+1 = next) and add to
                        # numerator
                        # Names: numerator_top, numerator_bottom
                        num_top = alphas[i, prev] * self.O[next, seq[i+1]] * \
                               self.A[prev, next] * betas[i+1, next]
                        num_bot = 0
                        # SHOULD BE PRECOMPUTED (FIX LATER)
                        for prev_state in range(L):
                            for next_state in range(L):
                                bot1 += alphas[i, prev_state] * \
                                        self.O[next_state, seq[i+1]] * \
                                        self.A[prev_state, next_state] * \
                                        betas[i+1, next_state]
                        numerator += num_top / num_bot

                        # Compute P(y_i = b) and add to denominator
                        # Names: denominator_top, denominator_bottom
                        denom_top = alphas[i, prev] * betas[i, prev]
                        denom_bot = 0
                        # SHOULD BE PRECOMPUTED (FIX LATER) OR DOT PRODUCT
                        for state in len(L):
                            denom_bot += alphas[i, state] * betas[i, state]
                        denominator += denom_top / denom_bot

                # UPDATE A_{prev, next}
                self.A[prev, next] = numerator / denominator

# Testing
testing = True

if testing:
    h = HMM(2)
    print "\n"
    data = [['R', 'W', 'B', 'B']]
    h.train(data, scaling=False)
