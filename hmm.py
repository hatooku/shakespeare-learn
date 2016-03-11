import numpy as np
from multiprocessing import Pool
import time

class HMM:

    def __init__(self, num_states):

        self.D = 0 # num of unique observations
        self.L = num_states # num of hidden states

        self.token_dict = {} # map of integers to tokens

        self.A = None # transition (row: from; col: to), 0-indexed
        self.PI = None # initial state distribution, 0-indexed
        self.O = None # observation (row: state; col: observation), 0-indexed

    def train(self, data, epsilon=0.001):
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
            starttime = time.time()
            iterations += 1
            # E Step
            gamma_arr, xi_arr = self.computeMarginals(X)

            # M step (Computes marginals + Updates)
            change_norm = self.update(X, gamma_arr, xi_arr)
            print change_norm
            norm_arr.append(change_norm)
            print("--- %s seconds ---" % (time.time() - starttime))

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

    def forwardBackward(self, seq):
        """ This function computes alpha and beta values for a sequence
            using the Forward-Backward algorithm.
        """
        M = len(seq) # length of given sequence
        L = self.L # num of states

        #fbPool = Pool()

        # FORWARD ALGORITHM
        alphas = self.forward(seq)
        #alphas = fbPool.apply(unwrap_self_forward, args=([(self, seq)]))
        # BACKWARD ALGORITHM
        betas = self.backward(seq)
        #betas = fbPool.apply(unwrap_self_backward, args=([(self, seq)]))

        #fbPool.close()
        #fbPool.join()

        return (alphas, betas)

    """ Runs the Forward Algorithm.
        @param i : index
        @param s : state
    """
    def forward(self, seq):
        M = len(seq)
        L = self.L
        alphas = np.zeros((M, L)) # row: position; col: state
        # FORWARD ALGORITHM
        for i in range(M): # For each observation
            for s in range(L): # For each state
                # Base case
                if i == 0:
                    alphas[i, s] = self.O[s, seq[i]] * self.PI[s]
                else:
                    sum = 0
                    # For each previous state
                    for prev in range(self.L):
                        sum += alphas[i-1, prev] * self.A[prev, s]
                    alphas[i, s] = sum * self.O[s, seq[i]]
        alphas = self.normalize(alphas)
        return alphas

    """ Runs the Backward Algorithm.
        @param i : token
        @param s : state
    """
    def backward(self, seq):
        M = len(seq)
        L = self.L
        betas = np.zeros((M, L)) # row: position; col: state
        # BACKWARD ALGORITHM
        for i in reversed(range(M)): # For each observation
            for s in range(L): # For each state
                # Base case
                if i == M-1:
                    betas[i, s] = 1
                else:
                    # For each next state
                    for next in range(self.L):
                        betas[i, s] += betas[i+1, next] * \
                                       self.A[s, next] * self.O[next, seq[i+1]]
        betas = self.normalize(betas)
        return betas

    def computeMarginals(self, X):
        # Calculate alphas and betas for all sequences
        L = self.L
        alphas_arr = []
        betas_arr = []
        for seq in X:
            alphas, betas = self.forwardBackward(seq)
            alphas_arr.append(alphas)
            betas_arr.append(betas)



        mgPool = Pool()

        gamma_arr = mgPool.apply_async(unwrap_self_gamma,
                                 args=([(self, X, alphas_arr, betas_arr)])).get()
        xi_arr = mgPool.apply_async(unwrap_self_xi,
                                 args=([(self, X, alphas_arr, betas_arr)])).get()

        mgPool.close()
        mgPool.join()

        #gamma_arr = self.computeGammas(X, alphas_arr, betas_arr)
        #xi_arr = self.computeXis(X, alphas_arr, betas_arr)

        return (gamma_arr, xi_arr)

    def computeGammas(self, X, alphas_arr, betas_arr):
        gamma_arr = [] # Indexed by: # Sequence, Position, State
        L = self.L
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

        return gamma_arr

    def computeXis(self, X, alphas_arr, betas_arr):
        L = self.L
        # P(y_i = prev, y_i+1 = next)
        xi_arr = [] # Indexed by: # Sequence, Position of Prev, Prev, Next

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

        return xi_arr

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
        #np.testing.assert_almost_equal(PI.sum(), 1)

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
            #np.testing.assert_almost_equal(A[prev].sum(), 1)

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
            #np.testing.assert_almost_equal(O[state].sum(), 1)

        # frobenius norm of the differences between update and previous matrices
        change_norm = np.linalg.norm(self.A - A) + np.linalg.norm(self.O - O) \
                      + np.linalg.norm(self.PI - PI)

        # update matrices
        self.O = O
        self.PI = PI
        self.A = A

        return change_norm


def unwrap_self_forward(arg, **kwarg):
    return HMM.forward(*arg, **kwarg)

def unwrap_self_backward(arg, **kwarg):
    return HMM.backward(*arg, **kwarg)

def unwrap_self_gamma(arg, **kwarg):
    return HMM.computeGammas(*arg, **kwarg)

def unwrap_self_xi(arg, **kwarg):
    return HMM.computeXis(*arg, **kwarg)

if __name__ == '__main__':
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
