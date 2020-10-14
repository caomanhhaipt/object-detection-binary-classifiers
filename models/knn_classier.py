from scipy.spatial import distance

class KnnClassifier(object):
    def __init__(self, images, tests):
        self.images = images
        self.tests = tests

    # 1-NN algorithm. We use this for predict the class of test images.
    # Takes 2 parameters. images is the feature vectors of train images and tests is the feature vectors of test images
    # Returns an array that holds number of test images, number of correctly predicted images and records of class based images respectively
    def classifier(self):
        num_test = 0
        correct_predict = 0
        class_based = {}

        for test_key, test_val in self.tests.items():
            class_based[test_key] = [0, 0]  # [correct, all]
            for tst in test_val:
                predict_start = 0
                # print(test_key)
                minimum = 0
                key = "a"  # predicted
                for train_key, train_val in self.images.items():
                    for train in train_val:
                        if (predict_start == 0):
                            minimum = distance.euclidean(tst, train)
                            # minimum = L1_dist(tst,train)
                            key = train_key
                            predict_start += 1
                        else:
                            dist = distance.euclidean(tst, train)
                            # dist = L1_dist(tst,train)
                            if (dist < minimum):
                                minimum = dist
                                key = train_key

                if (test_key == key):
                    correct_predict += 1
                    class_based[test_key][0] += 1
                num_test += 1
                class_based[test_key][1] += 1
                # print(minimum)
        return [num_test, correct_predict, class_based]

    # Calculates the average accuracy and class based accuracies.
    def accuracy(self, results):
        avg_accuracy = (results[1] / results[0]) * 100
        print("Average accuracy: %" + str(avg_accuracy))
        print("\nClass based accuracies: \n")
        for key, value in results[2].items():
            acc = (value[0] / value[1]) * 100
            print(key + " : %" + str(acc))
