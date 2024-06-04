from src.Abstract_classes import AbstractDataLoader
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

class DataloaderCSV(AbstractDataLoader):
    def __init__(self, random_state=None):
        super().__init__(random_state)
        self.features = None
        self.labels = None
        self.features_test = None
        self.labels_test = None
        self.main_path = "../data/processed/"

    def load_data(self, name):
        """
        Load data from the specified path.
        """
        features_path = self.main_path + name +"/" + name + "_features.csv"
        labels_path = self.main_path + name +"/" + name + "_labels.csv"
        #Load data and save it as ndarrays
        #self.features = np.loadtxt(features_path, delimiter=',')
        #self.labels = np.loadtxt(labels_path, delimiter=',')
        self.features = pd.read_csv(features_path).to_numpy()
        labels = pd.read_csv(labels_path).to_numpy()
        #make labels 1d
        self.labels = labels.ravel()
        
    
    def get_data(self, amount=None):
        """
        Fetch a specific amount of data for training/testing. If amount is None, fetch all available data.
        This method should return the data as two ndarrays: (data, labels).
        """
        if amount is None:
            return self.features, self.labels
        else:
            return self.features[:amount], self.labels[:amount]

    def get_test_data(self, amount=None):
        """
        Fetch a specific amount of data for testing. If amount is None, fetch all available data.
        This method should return the data as two ndarrays: (data, labels).
        """
        if self.features_test is None or self.labels_test is None:
            raise ValueError("No test data available")
        if amount is None:
            return self.features_test, self.labels_test
        else:
            return self.features_test[:amount], self.labels_test[:amount]
            
    def shuffle_data(self, random_state = None):
        """
        Shuffle the data
        """
        if self.features is not None and self.labels is not None:
            indices = np.arange(self.get_data_size())
            rng = np.random.default_rng(random_state)
            shuffled_indices = rng.permutation(indices)
            self.labels = self.labels[shuffled_indices]
            self.features = self.features[shuffled_indices]
    
    def split_data(self, test_size = 0.2, random_state = None):
        """
        Split the data into training and testing sets., dont return the data but save the split
        """
        if self.features is None or self.labels is None:
            raise ValueError("No data to split")
        self.features, self.features_test, self.labels, self.labels_test = train_test_split(self.features, self.labels, test_size=test_size, random_state=random_state)
    
    def get_data_size(self):
        """
        Return the size of the data.
        """
        return len(self.labels)
            
            

class DataloaderINPUT(AbstractDataLoader):
    def __init__(self, random_state=None):
        super().__init__(random_state)
        self.features = None
        self.labels = None
        self.features_test = None
        self.labels_test = None
        self.main_path = "../data/processed/"

    def load_data(self, x, y):
        """
        Load data from the specified path.
        """
        #Load data and save it as ndarrays
        self.features = x
        #make labels 1d
        self.labels = y
        
    
    def get_data(self, amount=None):
        """
        Fetch a specific amount of data for training/testing. If amount is None, fetch all available data.
        This method should return the data as two ndarrays: (data, labels).
        """
        if amount is None:
            return self.features, self.labels
        else:
            return self.features[:amount], self.labels[:amount]

    def get_test_data(self, amount=None):
        """
        Fetch a specific amount of data for testing. If amount is None, fetch all available data.
        This method should return the data as two ndarrays: (data, labels).
        """
        if self.features_test is None or self.labels_test is None:
            raise ValueError("No test data available")
        if amount is None:
            return self.features_test, self.labels_test
        else:
            return self.features_test[:amount], self.labels_test[:amount]
            
    def shuffle_data(self, random_state = None):
        """
        Shuffle the data
        """
        if self.features is not None and self.labels is not None:
            indices = np.arange(self.get_data_size())
            rng = np.random.default_rng(random_state)
            shuffled_indices = rng.permutation(indices)
            self.labels = self.labels[shuffled_indices]
            self.features = self.features[shuffled_indices]
    
    def split_data(self, test_size = 0.2, random_state = None):
        """
        Split the data into training and testing sets., dont return the data but save the split
        """
        if self.features is None or self.labels is None:
            raise ValueError("No data to split")
        self.features, self.features_test, self.labels, self.labels_test = train_test_split(self.features, self.labels, test_size=test_size, random_state=random_state)
    
    def get_data_size(self):
        """
        Return the size of the data.
        """
        return len(self.labels)

