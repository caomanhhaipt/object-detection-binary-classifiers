from sklearn import svm
from sklearn.metrics import accuracy_score

class SVMClassifier(object):
    def __init__(self, X_train, y_train):
        self.X_train = X_train
        self.y_train = y_train

    def classifier(self):
        self.clf = svm.LinearSVC(verbose=False)
        self.clf.fit(self.X_train, self.y_train)

    def predict(self, X_test):
        return self.clf.predict(X_test)

    def accuracy(self, y_test, y_predict, labels):
        class_based = {}
        num_test = {}
        for index, item in enumerate(y_test):
            if item not in class_based:
                class_based[item] = 0
                num_test[item] = 0

            if item == y_predict[index]:
                class_based[item] += 1

            num_test[item] += 1

        for item in class_based:
            print(labels[item] + ": " + str(class_based[item] / num_test[item]))

        accuracy1 = accuracy_score(y_predict, y_test)
        print("Avg: " + str(accuracy1))

