from sklearn.cluster import KMeans
import os
import numpy as np

DIR_PATH = os.path.abspath(os.path.join(os.path.dirname(os.path.realpath(__file__)), os.pardir))

class KMeansCluster(object):
    def __init__(self, descriptor_list=None):
        self.descriptor_list = descriptor_list

    # A k-means clustering algorithm who takes 2 parameter which is number
    # of cluster(k) and the other is descriptors list(unordered 1d array)
    # Returns an array that holds central points.
    def cluster(self, k=20):
        kmeans = KMeans(n_clusters=k, n_init=10)
        kmeans.fit(self.descriptor_list)
        visual_words = kmeans.cluster_centers_
        return visual_words

    def save(self, visual_words, path_to_save=DIR_PATH + "/results/visual_words.npy"):
        np.save(path_to_save, visual_words)
        # print ("Finish!!!")

    def load(self, path_to_save=DIR_PATH + "/results/visual_words.npy"):
        try:
            return np.load(path_to_save)
        except:
            print ("The file does not exist!!!")

if __name__ == "__main__":
    kmeans = KMeansCluster()

    print (kmeans.load())
