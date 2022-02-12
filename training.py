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

train_data_file = open("train.csv", mode="r")
train_imgs = []
for line in train_data_file:
    train_imgs.append(line.strip().split(","))
train_imgs = train_imgs[1:]
train_data_file.close()

test_data_file = open("test.csv", mode="r")
test_imgs = []
for line in test_data_file:
    test_imgs.append(line.strip().split(","))
test_imgs = test_imgs[1:]
test_data_file.close()

class_probs_train = []
labels_train = []

# num_of_train_cases = 2000
train_count = 0
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
    train_count += 1
    if train_count % 500 == 0:
        print("Przetworzono", train_count, "obrazkow do zbioru treningowego")

image_probs_file = open("X_train.txt", mode="w")
for x in [str(x.tolist()) for x in class_probs_train]:
    image_probs_file.write(x+"\n")

labels_file = open("y_train.txt", mode="w")
for x in labels_train:
    labels_file.write(x+"\n")


knn_model = KNeighborsClassifier(n_jobs=-1, n_neighbors=3)
tree_model = DecisionTreeClassifier()
forest_model = RandomForestClassifier()
svm_model = SVC()

models = [knn_model, tree_model, forest_model, svm_model]

for model in models:
    model.fit(class_probs_train, labels_train)

class_probs_test = []
labels_test = []
num_of_test_cases = 100

test_count = 0
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
    test_count += 1
    if test_count % 200 == 0:
        print("Przetworzono", test_count, "obrazkow do zbioru testowego")

image_probs_file_test = open("X_test.txt", mode="w")
for x in [str(x.tolist()) for x in class_probs_test]:
    image_probs_file_test.write(x+"\n")

labels_file_test = open("y_test.txt", mode="w")
for x in labels_test:
    labels_file_test.write(x+"\n")

for model in models:
    acc = model.score(class_probs_test, labels_test)
    print(type(model), " model accuracy: {:.2f}%".format(acc * 100))
    dump(model, str(type(model))[8:-2]+'.joblib')
