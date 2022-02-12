from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC

from joblib import dump

class_probs_train = []
image_probs_file = open("X_train.txt", mode="r")
for line in image_probs_file:
    class_probs_train.append([float(x) for x in line.strip().replace(" ", "")[1:-1].split(',')][150:268])

labels_train = []
labels_train_file = open("y_train.txt", mode="r")
for line in labels_train_file:
    labels_train.append(line[:-1])

knn_model = KNeighborsClassifier(n_jobs=-1, n_neighbors=3)
tree_model = DecisionTreeClassifier()
forest_model = RandomForestClassifier()
svm_model = SVC()

models = [knn_model, tree_model, forest_model, svm_model]

for model in models:
    model.fit(class_probs_train, labels_train)

class_probs_test = []
image_probs_file_test = open("X_test.txt", mode="r")
for line in image_probs_file_test:
    class_probs_test.append([float(x) for x in line.strip().replace(" ", "")[1:-1].split(',')][150:268])

labels_test = []
labels_file_test = open("y_test.txt", mode="r")
for line in labels_file_test:
    labels_test.append(line[:-1])

for model in models:
    acc = model.score(class_probs_test, labels_test)
    print(type(model), " model accuracy: {:.2f}%".format(acc * 100))
    # dump(model, str(type(model))[8:-2]+'.joblib')
