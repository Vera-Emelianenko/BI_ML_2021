import numpy as np

class KNNClassifier:
    """
    K-neariest-neighbor classifier using L1 loss
    """
    
    def __init__(self, k=1):
        self.k = k
    

    def fit(self, X, y):
        self.train_X = X
        self.train_y = y

    def predict(self, X, n_loops=0):
        """
        Uses the KNN model to predict clases for the data samples provided
        
        Arguments:
        X, np array (num_samples, num_features) - samples to run
           through the model
        num_loops, int - which implementation to use

        Returns:
        predictions, np array of ints (num_samples) - predicted class
           for each sample
        """
        
        if n_loops == 0:
            distances = self.compute_distances_no_loops(X)
        elif n_loops == 1:
            distances = self.compute_distances_one_loop(X)
        else:
            distances = self.compute_distances_two_loops(X)
        
        if len(np.unique(self.train_y)) == 2:
            return self.predict_labels_binary(distances)
        else:
            return self.predict_labels_multiclass(distances)


    def compute_distances_two_loops(self, X):
    
        """
        Computes L1 distance from every sample of X to every training sample
        Uses simplest implementation with 2 Python loops

        Arguments:
        X, np array (num_test_samples, num_features) - samples to run
        
        Returns:
        distances, np array (num_test_samples, num_train_samples) - array
           with distances between each test and each train sample

(num_test, num_train), где координата [i][j] соотвествует расстоянию между i-м вектором в test (test[i]) и j-м вектором в train (train[j]).
        """
        test_X = X
        two_loop_dist = np.zeros(shape = (len(test_X),len(self.train_X)))
        for i in range (len(test_X)): 
            for j in range (len(self.train_X)):
                two_loop_dist [i,j] = np.abs(test_X[i] - self.train_X[j]).sum()
        return two_loop_dist
        

    def compute_distances_one_loop(self, X):
        """
        Computes L1 distance from every sample of X to every training sample
        Vectorizes some of the calculations, so only 1 loop is used

        Arguments:
        X, np array (num_test_samples, num_features) - samples to run
        
        Returns:
        distances, np array (num_test_samples, num_train_samples) - array
           with distances between each test and each train sample
        """
        test_X = X
        one_loop_dist = np.zeros(shape = (len(self.train_X), len(test_X)))
        for i in range (len(self.train_X)):
            one_loop_dist[i]  = np.sum(np.abs(test_X-self.train_X[i]), axis=1)
        one_loop_dist = np.transpose(one_loop_dist)
        return one_loop_dist


    def compute_distances_no_loops(self, X):
        """
        Computes L1 distance from every sample of X to every training sample
        Fully vectorizes the calculations using numpy

        Arguments:
        X, np array (num_test_samples, num_features) - samples to run
        
        Returns:
        distances, np array (num_test_samples, num_train_samples) - array
           with distances between each test and each train sample
        """

        """
        YOUR CODE IS HERE
        """
        test_X = X
        no_loop_dist = np.abs(test_X[:,None] - self.train_X).sum(-1)
        return no_loop_dist


    def predict_labels_binary(self, distances):
        print (distances) 
        """
        Returns model predictions for binary classification case

        Arguments:
        distances, np array (num_test_samples, num_train_samples) - array
        with distances between each test and each train sample
        Returns:
        pred, np array of bool (num_test_samples) - binary predictions 
        for every test sample
        """

        n_train = distances.shape[1]
        n_test = distances.shape[0]
        prediction = np.zeros(n_test)

        for i in range (n_test):
            ind = np.argpartition(distances[i], self.k)[:self.k]
            classes_of_neighbours = self.train_y[ind].astype(int)
            prediction[i] = np.argmax(np.bincount(classes_of_neighbours))
        return prediction

    


    def predict_labels_multiclass(self, distances):
        """
        Returns model predictions for multi-class classification case
        
        Arguments:
        distances, np array (num_test_samples, num_train_samples) - array
           with distances between each test and each train sample
        Returns:
        pred, np array of int (num_test_samples) - predicted class index 
           for every test sample
        """

        n_train = distances.shape[0]
        n_test = distances.shape[0]
        prediction = np.zeros(n_test, np.int)

        """
        YOUR CODE IS HERE
        """
        pass

