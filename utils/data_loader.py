import pickle
import numpy as np
import cv2
import os

DIR_PATH = os.path.abspath(os.path.join(os.path.dirname(os.path.realpath(__file__)), os.pardir))

def load_label_names():
    return ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']


# takes all images and convert them to grayscale.
# return a dictionary that holds all images category by category.
def load_cfar10(cifar10_dataset_folder_path=DIR_PATH + "/data/cifar-10-batches-py", type='train'):
    label_names = load_label_names()
    images = {}
    if type == "train":
        batch_numbers = [1,2,3,4,5]
    else:
        batch_numbers = [6] #batch_6 is testing batch

    for batch_id in batch_numbers:
        with open(cifar10_dataset_folder_path + '/data_batch_' + str(batch_id), mode='rb') as file:
            batch = pickle.load(file, encoding='latin1')
        for (label,img_mat) in zip(batch['labels'],batch['data']):
            r_channel = img_mat[:1024].reshape(32, 32)
            g_channel = img_mat[1024: 2 * 1024].reshape(32, 32)
            b_channel = img_mat[2 * 1024:].reshape(32, 32)
            image_repr = np.stack([r_channel, g_channel, b_channel], axis=2)
            img = cv2.cvtColor(image_repr, cv2.COLOR_RGB2GRAY)
            #add to dict
            label_name = label_names[label]
            if label_name not in images:
                images[label_name] = [img]
            else:
                images[label_name].append(img)
    return images

# takes a specific amount of images from particular classes and convert them to grayscale.
# return a dictionary that holds all images category by category.
def load_cfar10_batch(labels_to_take, amount, type, cifar10_dataset_folder_path=DIR_PATH + "/data/cifar-10-batches-py"):
    label_names = load_label_names()
    images = {}
    if type == "train":
        batch_numbers = [1,2,3,4,5]
    else:
        batch_numbers = [6] #batch_6 is testing batch

    for batch_id in batch_numbers:
        with open(cifar10_dataset_folder_path + '/data_batch_' + str(batch_id), mode='rb') as file:
            batch = pickle.load(file, encoding='latin1')
        for (label,img_mat) in zip(batch['labels'],batch['data']):
            if label_names[label] in labels_to_take:
                r_channel = img_mat[:1024].reshape(32, 32)
                g_channel = img_mat[1024: 2 * 1024].reshape(32, 32)
                b_channel = img_mat[2 * 1024:].reshape(32, 32)
                image_repr = np.stack([r_channel, g_channel, b_channel], axis=2)
                img = cv2.cvtColor(image_repr, cv2.COLOR_RGB2GRAY)
                #add to dict
                label_name = label_names[label]
                if label_name not in images:
                    images[label_name] = [img]
                else:
                    images[label_name].append(img)
            else:
                pass

    #slice random images from original data for each label (amount images)
    # start = randint(0, 1000-amount)
    start = 0
    for key, value in images.items():
        images[key] = images[key][start:start+amount]
    return images

if __name__ == "__main__":
    print (DIR_PATH)