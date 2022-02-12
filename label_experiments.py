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
print(dataset_breeds)

counter = 0
for breed in dataset_breeds:
    for net_label in id_to_label.values():
        if breed in net_label:
            counter += 1
print(counter, len(dataset_breeds))
