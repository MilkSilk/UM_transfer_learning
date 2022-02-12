import numpy as np
import tensorflow as tf

# Import the ResNet model and the preprocess_input function.
from keras.applications.resnet import ResNet50, preprocess_input

# Import the functions to load and transform images
from keras.preprocessing.image import load_img, img_to_array

# Create a Redidual-network already trained in the IMAGENET
ResNet50_model = ResNet50(weights='imagenet')

id_to_label_file = open("imagenet1000_clsidx_to_labels.txt", mode="r")
id_to_label = {}
for line in id_to_label_file:
    line = line.replace("{", "").replace(",\n", "").replace("'", "").split(":")
    id_to_label.update({int(line[0].strip()): line[1].strip()})
id_to_label_file.close()

breeds_file = open("breeds.txt", mode="r")
dataset_breeds = []
for breed in breeds_file:
    dataset_breeds.append(breed.strip())
breeds_file.close()

# Load the image in the size expected by the ResNet50_model
test_data_file = open("test.csv", mode="r")
test_imgs = []
for line in test_data_file:
    test_imgs.append(line.strip().split(","))
test_imgs = test_imgs[1:]
test_data_file.close()

correct_predictions = 0
false_predictions = 0

for image_data in test_imgs:
    img, label = image_data
    label = label.replace("_", " ")
    img = load_img(img, target_size=(224, 224))

    # Transform the image into an array
    img_array = img_to_array(img)
    img_preprocessed = preprocess_input(img_array)
    imgs_list = [np.array(img_preprocessed)]
    img_ready = np.asarray(imgs_list)

    # Pre-process the image according to IMAGENET standarts


    # Predicts
    tf.expand_dims(img_ready, axis=0)
    probs = ResNet50_model.predict(img_ready)

    # Find the position with the maximum probability value
    position_of_max = np.argmax(probs)
    if label in id_to_label[position_of_max]:
        correct_predictions += 1
    else:
        false_predictions += 1
    print(label, id_to_label[position_of_max])
print("correctly classified: ", correct_predictions/len(test_imgs))
print("wrongly classified: ", false_predictions/len(test_imgs))



