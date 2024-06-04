## a file to store the abstract classes for the project

# An abstract class for a strong leaner, like adaboost
class Strong_learner():
    def __init__(self):
        pass

    def fit(self, data, labels):
        """
        Fits the learner to the data.
        data: the data to classify | numpy array (n, d), where n is the number of samples and d is the number of features
        labels: the correct labels | numpy array (n,) [1,-1]
        """
        raise NotImplementedError("Subclasses must implement this method")

    def predict(self, data):
        """
        Classify the data
        data: the data to classify | numpy array (n, d), where n is the number of samples and d is the number of features
        """
        raise NotImplementedError("Subclasses must implement this method")
    
    def score(self, data, labels):
        """
        Calculate the accuracy of the model
        data: the data to classify | numpy array (n, d), where n is the number of samples and d is the number of features
        labels: the correct labels | numpy array (n,) [1,-1]
        """
        raise NotImplementedError("Subclasses must implement this method")

class AbstractDataLoader():
    def __init__(self, random_state=None):
        self.path = None
        self.random_state = random_state
        self.features = None
        self.labels = None
        self.features_test = None
        self.labels_test = None

    def load_data(self):
        """
        Load data from the specified path.
        """
        raise NotImplementedError("Subclasses must implement this method")


    def split_data(self, test_size=0.2):
        """
        Split the data into training and testing sets., dont return the data but save the split
        """
        raise NotImplementedError("Subclasses must implement this method")


    def get_data(self, amount=None):
        """
        Fetch a specific amount of data for training/testing. If amount is None, fetch all available data.
        This method should return a tuple: (data, labels).
        """
        raise NotImplementedError("Subclasses must implement this method")

    def set_random_state(self, random_state):
        self.random_state = random_state

    def get_data_size(self):
        return len(self.features)
    
    def shuffle_data(self):
        """
        Shuffel the data and labels
        """
        raise NotImplementedError("Subclasses must implement this method")

    def get_test_data(self):
        return self.features_test, self.labels_test
    
    def get_test_data_size(self):
        return len(self.features_test)

class abstractSubSampler():
    def __init__(self, random_state = None):
        self.random_state = random_state
    
    def sub_sample(self, data):
        """
        Subsample the data
        """
        raise NotImplementedError("Subclasses must implement this method")