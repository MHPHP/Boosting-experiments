import numpy as np
from src.Abstract_classes import *
from numba import jit
from math import log
import random
from concurrent.futures import ThreadPoolExecutor
from concurrent.futures import ProcessPoolExecutor
import multiprocessing
import math

class normal_adaboost(Strong_learner):
    def __init__(self, weak_learner):
        self.weak_learner = weak_learner
        self.learners = []
        self.model_weights = []
        self.data_weights = np.array([])

    def fit(self, data, labels, new_learners=1, stop_when_done = False):
        """
        Fits weak learners to the data.
        data: the data to classify | numpy array (n, d), where n is the number of samples and d is the number of features
        labels: the correct labels | numpy array (n,) [1,-1]
        new_learners: the number of weak learners to fit to the data | int
        """
        if len(self.data_weights) != len(data):
            self.data_weights = np.array([1/len(data)] * len(data))
        if stop_when_done == False:
            for i in range(new_learners):
                weak_learner = self.weak_learner()
                weak_learner.fit(data, labels, sample_weight=self.data_weights)
                self.learners.append(weak_learner)
                predictions = np.array(weak_learner.predict(data))
                error = np.sum(self.data_weights[predictions != labels])
                #check if the error is 0, if it is, we can't calculate alpha, so we just set it to 1
                if error == 0:
                    self.model_weights.append(1)
                    break
                alpha = 0.5 * np.log((1-error)/error)
                self.model_weights.append(alpha)
                z = np.sum(self.data_weights * np.exp(-alpha * labels * predictions))
                self.data_weights = self.data_weights * np.exp(-alpha * labels * predictions) / z
        else:
            score_all = 0
            while score_all != 1:
                weak_learner = self.weak_learner()
                weak_learner.fit(data, labels, sample_weight=self.data_weights)
                self.learners.append(weak_learner)
                predictions = np.array(weak_learner.predict(data))
                error = np.sum(self.data_weights[predictions != labels])
                #check if the error is 0, if it is, we can't calculate alpha, so we just set it to 1
                if error == 0:
                    self.model_weights.append(1)
                    print("Weak learner with 0 error found, stopping training of new learners")
                    break
                alpha = 0.5 * np.log((1-error)/error)
                self.model_weights.append(alpha)
                z = np.sum(self.data_weights * np.exp(-alpha * labels * predictions))
                self.data_weights = self.data_weights * np.exp(-alpha * labels * predictions) / z

                score_all = self.score(data, labels)

    def voting_classify(self, data):
        """
        Calculate the sum of the output for all classifiers, but without the sign
        data: the data to classify | numpy array (n, d), where n is the number of samples and d is the number of features
        """
        votes = np.zeros(len(data))
        for i, learner in enumerate(self.learners):
            votes += self.model_weights[i] * learner.predict(data)
        return votes
    
    def predict(self, data):
        """
        Classify the data
        data: the data to classify | numpy array (n, d), where n is the number of samples and d is the number of features
        """
        return np.sign(self.voting_classify(data))

    def score(self, data, labels):
        """
        Calculate the accuracy of the model
        data: the data to classify | numpy array (n, d), where n is the number of samples and d is the number of features
        labels: the correct labels | numpy array (n,) [1,-1]
        """
        return np.mean(self.predict(data) == labels)

    def get_number_of_learners(self):
        return len(self.learners)

class adaboost_v(Strong_learner):
    def __init__(self, weak_learner, v = 0.01):
        self.weak_learner = weak_learner
        self.learners = []
        self.model_weights = []
        self.data_weights = np.array([])
        self.v = v
        self.gammas = []

    def fit(self, data, labels, new_learners=1, stop_when_done = False):
        """
        Fits weak learners to the data.
        data: the data to classify | numpy array (n, d), where n is the number of samples and d is the number of features
        labels: the correct labels | numpy array (n,) [1,-1]
        new_learners: the number of weak learners to fit to the data | int
        """
        if len(self.data_weights) != len(data):
            self.data_weights = np.array([1/len(data)] * len(data))
        if stop_when_done == False: 
            for i in range(new_learners):
                weak_learner = self.weak_learner()
                weak_learner.fit(data, labels, sample_weight=self.data_weights)
                self.learners.append(weak_learner)
                predictions = np.array(weak_learner.predict(data))
                gamma = np.sum(self.data_weights * predictions * labels)
                self.gammas.append(gamma)
                if np.abs(gamma)  >= 1:
                    self.model_weights.append(np.sign(gamma))
                    print("gamma is 1")
                    break
                gamma_min = min(self.gammas)
                rho = gamma_min - self.v
                alpha = 0.5 * np.log((1+gamma)/(1-gamma)) - 0.5 * np.log((1+rho)/(1-rho))
                self.model_weights.append(alpha)
                z = np.sum(self.data_weights * np.exp(-alpha * labels * predictions))
                self.data_weights = self.data_weights * np.exp(-alpha * labels * predictions) / z
        else:
            score_all = 0
            while score_all != 1:
                weak_learner = self.weak_learner()
                weak_learner.fit(data, labels, sample_weight=self.data_weights)
                self.learners.append(weak_learner)
                predictions = np.array(weak_learner.predict(data))
                gamma = np.sum(self.data_weights * predictions * labels)
                self.gammas.append(gamma)
                if np.abs(gamma)  >= 1:
                    self.model_weights.append(np.sign(gamma))
                    print("gamma is 1")
                    break
                gamma_min = min(self.gammas)
                rho = gamma_min - self.v
                alpha = 0.5 * np.log((1+gamma)/(1-gamma)) - 0.5 * np.log((1+rho)/(1-rho))
                self.model_weights.append(alpha)
                z = np.sum(self.data_weights * np.exp(-alpha * labels * predictions))
                self.data_weights = self.data_weights * np.exp(-alpha * labels * predictions) / z

                score_all = self.score(data, labels)

    def voting_classify(self, data):
        """
        Calculate the sum of the output for all classifiers, but without the sign
        data: the data to classify | numpy array (n, d), where n is the number of samples and d is the number of features
        """
        votes = np.zeros(len(data))
        for i, learner in enumerate(self.learners):
            votes += self.model_weights[i] * learner.predict(data)
        return votes/np.sum(self.model_weights)
    
    def predict(self, data):
        """
        Classify the data
        data: the data to classify | numpy array (n, d), where n is the number of samples and d is the number of features
        """
        return np.sign(self.voting_classify(data))

    def score(self, data, labels):
        """
        Calculate the accuracy of the model
        data: the data to classify | numpy array (n, d), where n is the number of samples and d is the number of features
        labels: the correct labels | numpy array (n,) [1,-1]
        """
        return np.mean(self.predict(data) == labels)

    def get_number_of_learners(self):
        return len(self.learners)



class adversarial_weak_learner():
    def __init__(self, d, universe_size, t, S_size,  gamma, H=None, random_subset = False, best_bad = False):
        self.h_size = 2**(d-1)
        self.universe_size = universe_size
        self.t = t
        self.S_size = S_size
        self.gamma = gamma
        if H is None:
            h_0 = np.ones((1, universe_size))
            indices = np.random.choice(universe_size, t, replace=False)
            h_0[0, indices] = -1
            self.H = np.random.choice([-1, 1], size=(self.h_size, universe_size))
            self.H = np.vstack((self.H, h_0))
        else:
            self.H = H
        self.learned_hypothesis = None
        self.random_subset = random_subset
        self.best_bad = best_bad


    def fit(self, data, labels, sample_weight):
        """
        data: (ndarray) indicies of the the sampled data.
        labels: (ndarray) the labels of the data (not used, but needed to match the other interfaces)
        sample_weight: (ndarray)  the weights of the data.
        """
        #Find all hypotheses that have a gamma advantage
        new_weights = self.translate_weights(data, sample_weight)
        gamma_advantages = self.find_gamma_advantage(new_weights)
        #check if there are any hypothesis with a gamma advantage
        if len(gamma_advantages) == 0:
            raise ValueError("No hypothesis has a gamma advantage")
        #only look at the hypothesis that have a gamma advantage
        H_adv = self.H[gamma_advantages]
        if self.random_subset:
            self.seed = hash(tuple(data)) % (2**32)
            self.rng = np.random.default_rng(self.seed)
            S = self.find_random_subset(data)
        else:
            S = self.find_S_subset(data)
        #Find the hypothesis with the most negative advantage
        most_negative_hypothesis, negativ_adv = self.find_most_negative_hypothesis(H_adv, S)
        self.learned_hypothesis = most_negative_hypothesis
        #self.H = None
        #print("the negative advantage on S is: ", negativ_adv)
        #print(f"the advantage on the data is: {self.score(most_negative_hypothesis, new_weights) - 0.5:.3f} and the sum of the hypothesis is: {sum(most_negative_hypothesis):.3f}   And the negative advantage on S is: {negativ_adv - 0.5:.3f}")

    @staticmethod
    @jit(nopython=True)
    def translate_weights_static(data, weights, universe_size):
        new_weights = np.zeros(universe_size)
        for i in range(len(data)):
            new_weights[data[i]] += weights[i]
        return new_weights

    def translate_weights(self, data, weights):
        return self.translate_weights_static(data, weights, self.universe_size)

    @staticmethod
    @jit(nopython=True)
    def find_gamma_advantage_static(H, D, gamma):
        advantages_indices = []
        for i in range(len(H)):
            score = np.sum(((H[i] + 1) / 2) * D) - 0.5
            if score > gamma:
                advantages_indices.append(i)
        return np.array(advantages_indices)

    def find_gamma_advantage(self, D):
        # Use the static method for the Numba-optimized operation
        return self.find_gamma_advantage_static(self.H, D, self.gamma)

    @staticmethod
    @jit(nopython=True)
    def find_most_negative_hypothesis_static(H, S, universe_size, gamma, best_bad):
        if best_bad:
            max_score = np.inf
            best_hyp = -1  # Initialize with an invalid index
            D_S = np.zeros(universe_size)
            D_S[S] += 1 / len(S)
            
            for i in range(H.shape[0]):
                score = np.sum(((H[i] + 1) / 2) * D_S) - 0.5
                if score > max_score and score <  -gamma:
                    max_score = score
                    best_hyp = i
                    
            return best_hyp, max_score
        else:
            min_score = np.inf
            best_hyp = -1  # Initialize with an invalid index
            D_S = np.zeros(universe_size)
            D_S[S] += 1 / len(S)
            
            for i in range(H.shape[0]):
                score = np.sum(((H[i] + 1) / 2) * D_S) - 0.5
                if score < min_score:
                    min_score = score
                    best_hyp = i
                    
            return best_hyp, min_score

    def find_most_negative_hypothesis(self, H, S):
        # Assuming H and S are already prepared for this operation
        idx, score = self.find_most_negative_hypothesis_static(H, S, self.universe_size, self.gamma, self.best_bad)
        return H[idx], score  # Adjust based on how you use the result


    def predict(self, data):
        if self.learned_hypothesis is None:
            raise ValueError("The model has not been fitted yet")
        return self.learned_hypothesis[data]

    @staticmethod
    @jit(nopython=True)
    def score_static(h, D):
        return np.sum(((h + 1) / 2) * D)

    def find_S_subset(self, data):
        not_sampled = np.setdiff1d(np.arange(self.universe_size), data)

        return not_sampled[:self.S_size]

    def find_random_subset(self, data):
        not_sampled = np.setdiff1d(np.arange(self.universe_size), data)
        S_subset = self.rng.choice(not_sampled, self.S_size, replace=False)
        return S_subset

    def return_H(self):
        return self.H

def sub_sample_inner(A,B=[]):
    """"Implementing Hanneke Sub-sample algorithm using list
    Parameters:
    ------------
    A: list
    B: list (May be empty)
    ------------
    """
    print(A)
    if len(A) <= 3:
        return A + B
    else: 
        #splitting the array into 4 parts
        q, r = divmod(len(A), 4)
        # Initialize split indices
        indices = [0]
        for i in range(1, 5):
            indices.append(indices[-1] + q + (1 if i <= r else 0))
        A0 = A[indices[0]:indices[1]]
        A1 = A[indices[1]:indices[2]]
        A2 = A[indices[2]:indices[3]]
        A3 = A[indices[3]:indices[4]]

        #recursively call the function
        sub1 = sub_sample(A0, A2 + A3 + B)
        sub2 = sub_sample(A0, A1 + A3 + B)
        sub3 = sub_sample(A0, A1 + A2 + B)
        return [sub1] + [sub2] + [sub3]

def sub_sample(A, B =[]):
    """
    An outer function of the Hanneke sub_sampple algotihm that also does the unpacking of the list
    """
    return unpack_nested_lists(sub_sample_inner(A, B))

def unpack_nested_lists(nested_list):
    result = []
    for element in nested_list:
        if isinstance(element, list) and element and isinstance(element[0], list):
            result.extend(unpack_nested_lists(element))  # Recursively unpack if the element is a list
        else:
            result.append(element)
    return result

class optimal_weak_to_strong_learner(Strong_learner):
    def __init__(self, adaboost, sub_sample_algorithm):
        self.sub_sample_algorithm = sub_sample_algorithm
        self.adaboost = adaboost
        self.adaboosts = []
        
    def fit(self, data, labels, n_weak_learners, stop_when_done = False):
        #make data to list
        data = data.tolist()
        labels = labels.tolist()
        #combine the data and labels, to tuples
        data = list(zip(data, labels))
        unpacked_samples = self.sub_sample_algorithm.sub_sample(data)
        for sample in unpacked_samples:
            #unzip the sample
            data_sample, labels = zip(*sample)
            #convert to numpy array
            data_sample = np.array(data_sample)
            labels = np.array(labels)
            adaboost = self.adaboost()
            adaboost.fit(data_sample, labels, n_weak_learners, stop_when_done = stop_when_done)
            self.adaboosts.append(adaboost)

    def predict(self, data):
        result = []
        for adaboost in self.adaboosts:
            result.append(adaboost.predict(data))
        return np.sign(np.sum(result, axis=0))

    def score(self, data, labels):
        return np.mean(self.predict(data) == labels)

class hanneke_sub_sample(abstractSubSampler):
    def __init__(self):
        pass

    def sub_sample(self, A, B =[]):
        """
        An outer function of the Hanneke sub_sampple algotihm that also does the unpacking of the list In Class
        """
        return unpack_nested_lists(self.sub_sample_inner(A, B))

    def sub_sample_inner(self, A,B=[]):
        """"Implementing Hanneke Sub-sample algorithm using list
        Parameters:
        ------------
        A: list
        B: list (May be empty)
        ------------
        """
        
        if len(A) <= 3:
            return A + B
        else: 
            #splitting the array into 4 parts
            q, r = divmod(len(A), 4)
            # Initialize split indices
            indices = [0]
            A_size = len(A)
            for i in range(1, 5):
                if i == 1:
                    indices.append(A_size - 3* math.floor(A_size/4 ))
                else:
                    indices.append(indices[1] + math.floor(A_size/4) * (i-1))

            A0 = A[indices[0]:indices[1]]
            A1 = A[indices[1]:indices[2]]
            A2 = A[indices[2]:indices[3]]
            A3 = A[indices[3]:indices[4]]

            

            #recursively call the function
            sub1 = self.sub_sample(A0, A2 + A3 + B)
            sub2 = self.sub_sample(A0, A1 + A3 + B)
            sub3 = self.sub_sample(A0, A1 + A2 + B)
            return [sub1] + [sub2] + [sub3]
    
class bootstrap_sub_sample(abstractSubSampler):
    def __init__(self,  log_factor = None, n = None, gamma = None):
        self.log_factor = log_factor
        self.n = n
        self.gamma = gamma
        #Set the number of samples
        self.number_of_samples = None
        if n is not None:
            self.number_of_samples = n

    def sub_sample(self, data):
        length_of_data = len(data)
        if self.number_of_samples is None:
            if self.log_factor is None:
                #Use natural log as default
                self.number_of_samples = int(np.log(length_of_data/self.gamma))
            else:
                self.number_of_samples = int(np.log(length_of_data/self.gamma, self.log_factor))
        #Create the samples
        samples = []
        for i in range(self.number_of_samples):
            samples.append(random.choices(data, k = length_of_data))
        return samples

class split_sub_sampler(abstractSubSampler):
    def __init__(self, splits=10, times=1):
        self.splits = splits
        self.times = times

    def sub_sample(self, data):
        all_samples = []
        
        for _ in range(self.times):
            # Copy and shuffle the data to ensure different splits every time
            shuffled_data = data[:]  # Make a copy of the data
            random.shuffle(shuffled_data)

            length_of_data = len(shuffled_data)
            split_size = length_of_data // self.splits
            remainder = length_of_data % self.splits

            # Generate the splits
            splits = []
            start = 0
            for i in range(self.splits):
                end = start + split_size + (1 if i < remainder else 0)
                splits.append(shuffled_data[start:end])
                start = end

            # Create the subsets
            samples = []
            for i in range(self.splits):
                subset = []
                for j in range(self.splits):
                    if i != j:
                        subset.extend(splits[j])
                samples.append(subset)

            # Append this round's samples to all samples
            all_samples.extend(samples)

        return all_samples