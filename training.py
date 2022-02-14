import numpy as np
import tensorflow as tf
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC

from joblib import dump

from keras.applications.resnet import ResNet50, preprocess_input

# Import the functions to load and transform images
from keras.preprocessing.image import load_img, img_to_array

# Create a Redidual-network already trained in the IMAGENET
ResNet50_model = ResNet50(weights='imagenet', include_top=True)

chosen_breeds = ['Labrador_retriever', 'Boston_bull', 'German_shepherd', 'golden_retriever', 'French_bulldog',
                 'standard_poodle', 'beagle', 'Rottweiler', 'German_short-haired_pointer', 'Scotch_terrier', 'boxer',
                 'Siberian_husky', 'Blenheim_spaniel', 'Doberman', 'miniature_schnauzer', 'Shih-Tzu', 'Pomeranian',
                 'Maltese_dog', 'pug', 'Sussex_spaniel']

with open("train.csv", mode="r") as train_data_file:
    train_imgs = []
    for line in train_data_file:
        img, breed = line.strip().split(",")
        if breed in chosen_breeds:
            train_imgs.append([img, breed])
    train_imgs = train_imgs[1:]

with open("test.csv", mode="r") as test_data_file:
    test_imgs = []
    for line in test_data_file:
        img, breed = line.strip().split(",")
        if breed in chosen_breeds:
            test_imgs.append([img, breed])
    test_imgs = test_imgs[1:]

class_probs_train = []
labels_train = []

for image_data in train_imgs:
    img, label = image_data
    label = label.replace("_", " ")
    img = load_img(img, target_size=(224, 224))
    img_array = img_to_array(img)
    img_preprocessed = preprocess_input(img_array)
    imgs_list = [np.array(img_preprocessed)]
    img_ready = np.asarray(imgs_list)

    tf.expand_dims(img_ready, axis=0)
    probs = ResNet50_model.predict(img_ready)
    class_probs_train.append(probs[0])
    labels_train.append(label)

knn_model = KNeighborsClassifier(n_jobs=-1, n_neighbors=7)
tree_model = DecisionTreeClassifier()
forest_model = RandomForestClassifier()
svm_model = SVC()

models = [knn_model, tree_model, forest_model, svm_model]

for model in models:
    model.fit(class_probs_train, labels_train)

class_probs_test = []
labels_test = []

for image_data in test_imgs:
    img, label = image_data
    label = label.replace("_", " ")
    img = load_img(img, target_size=(224, 224))

    # Transform the image into an array
    img_array = img_to_array(img)
    img_preprocessed = preprocess_input(img_array)
    imgs_list = [np.array(img_preprocessed)]
    img_ready = np.asarray(imgs_list)

    tf.expand_dims(img_ready, axis=0)
    probs = ResNet50_model.predict(img_ready)

    class_probs_test.append(probs[0])
    labels_test.append(label)

for model in models:
    acc = model.score(class_probs_test, labels_test)
    print(type(model), " model accuracy: {:.2f}%".format(acc * 100))
    dump(model, str(type(model))[8:-2]+'.joblib')
