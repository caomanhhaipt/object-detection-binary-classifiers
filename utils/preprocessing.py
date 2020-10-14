import cv2
import numpy as np
from scipy.spatial import distance

def show_images(data):
    for key, images in data.items():
        for img in images:
            cv2.imshow("test", img)
            cv2.waitKey()

# Creates descriptors using sift
# Takes one parameter that is images dictionary
# Return an array whose first index holds the decriptor_list without an order
# And the second index holds the sift_vectors dictionary which holds the descriptors but this is seperated class by class
def sift_features(images):
    sift_vectors = {}
    descriptor_list = []
    sift = cv2.xfeatures2d.SIFT_create()
    for key, value in images.items():
        features = []
        index = 0
        for img in value:
            kp, des = sift.detectAndCompute(img, None)
            if des is None:
                des = np.array([]).reshape(0, 128)
            descriptor_list.extend(des)
            features.append(des)
        sift_vectors[key] = features
    return [descriptor_list, sift_vectors]

# Find the index of the closest central point to the each sift descriptor.
# Takes 2 parameters the first one is a sift descriptor and the second one is the array of central points in k means
# Returns the index of the closest central point.
def find_index(image, center):
    count = 0
    ind = 0
    for i in range(len(center)):
        if(i == 0):
           count = distance.euclidean(image, center[i])
           #count = L1_dist(image, center[i])
        else:
            dist = distance.euclidean(image, center[i])
            #dist = L1_dist(image, center[i])
            if(dist < count):
                ind = i
                count = dist
    return ind

# Takes 2 parameters. The first one is a dictionary that holds the descriptors that are separated class by class
# And the second parameter is an array that holds the central points (visual words) of the k means clustering
# Returns a dictionary that holds the histograms for each images that are separated class by class.
def image_class(all_bovw, centers):
    dict_feature = {}
    for key, value in all_bovw.items():
        category = []
        for img in value:
            histogram = np.zeros(len(centers))
            for each_feature in img:
                ind = find_index(each_feature, centers)
                histogram[ind] += 1
            category.append(histogram)
        dict_feature[key] = category
    return dict_feature

def calculate_idf(all_bovw, centers):
    t = {}
    count_image = 0
    for key, value in all_bovw.items():
        for img in value:
            for feature in img:
                ind = find_index(feature, centers)
                if ind not in t:
                    t[ind] = 1
                else:
                    t[ind] += 1

            count_image += 1

    idf = {}
    for item in t:
        idf[item] = np.log(1.0*count_image/(t[item]+1))

    return idf

def calculate_tf_idf(all_bovw, centers, idf):
    dict_feature = {}
    for key, value in all_bovw.items():
        category = []
        for img in value:
            f = {}
            count_feature = 0
            for feature in img:
                ind = find_index(feature, centers)
                if ind not in f:
                    f[ind] = 1
                else:
                    f[ind] += 1

                count_feature += 1
            tf = {}
            for item in f:
                tf[item] = f[item]/count_feature

            histogram = np.zeros(len(centers))
            for item in idf:
                if item not in f:
                    pass
                else:
                    histogram[item] += idf[item]*f[item]

            category.append(histogram)
        dict_feature[key] = category

    return dict_feature

def convert_for_svm(tf_idf_train, tf_idf_test):
    X_train = []
    y_train = []
    label = 0
    labels = []
    for item in tf_idf_train:
        labels.append(item)
        for img in tf_idf_train[item]:
            X_train.append(img)
            y_train.append(label)

        label += 1

    X_test = []
    y_test = []
    label = 0
    for item in tf_idf_test:
        for img in tf_idf_test[item]:
            X_test.append(img)
            y_test.append(label)

        label += 1

    return X_train, y_train, X_test, y_test, labels