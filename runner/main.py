import time
from utils.data_loader import load_cfar10, load_cfar10_batch
from utils.preprocessing import show_images, sift_features, \
    calculate_tf_idf, calculate_idf, image_class, convert_for_svm
from models.knn_classier import KnnClassifier
from models.kmeans_cluster import KMeansCluster
from models.svm_classifier import SVMClassifier

def main():
    start = time.time()

    labels_to_take = ['ship', 'dog']  # choose classes from 10 original classes to import
    images = load_cfar10_batch(labels_to_take, 300, "train")
    test = load_cfar10_batch(labels_to_take, 300, "test")

    sifts = sift_features(images)

    # Takes the descriptor list which is unordered one
    descriptor_list = sifts[0]

    # Takes the sift features that is seperated class by class for train data
    all_bovw_feature = sifts[1]

    # Takes the sift features that is seperated class by class for test data
    test_bovw_feature = sift_features(test)[1]

    # Takes the central points which is visual words
    kmeans = KMeansCluster(descriptor_list)
    visual_words = kmeans.cluster(k=20)
    kmeans.save(visual_words)

    idf = calculate_idf(all_bovw_feature, visual_words)
    tf_idf_train = calculate_tf_idf(all_bovw_feature, visual_words, idf)
    tf_idf_test = calculate_tf_idf(test_bovw_feature, visual_words, idf)

    X_train, y_train, X_test, y_test, labels = convert_for_svm(tf_idf_train, tf_idf_test)

    svm = SVMClassifier(X_train, y_train)
    svm.classifier()

    y_predict = svm.predict(X_test)

    print ("-"*20 + "SVM" + "-"*20)
    svm.accuracy(y_test, y_predict, labels)

    # Creates histograms for train data
    bovw_train = image_class(all_bovw_feature, visual_words)

    # Creates histograms for test data
    bovw_test = image_class(test_bovw_feature, visual_words)

    knn = KnnClassifier(bovw_train, bovw_test)

    results_bowl = knn.classifier()
    print("-" * 20 + "BOW + KNN" + "-" * 20)
    knn.accuracy(results_bowl)

    # Call the knn function
    knn.images = tf_idf_train
    knn.tests = tf_idf_test
    results_tf_idf = knn.classifier()

    print("-" * 20 + "tf/idf + KNN" + "-" * 20)
    knn.accuracy(results_tf_idf)
    print("\nRun time: ", time.time() - start)

if __name__ == '__main__':
    main()