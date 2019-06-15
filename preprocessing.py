import numpy as np
# For plotting 2-D projections
from sklearn.manifold import TSNE
# For plotting
import matplotlib.pyplot as plt
# For balancing the classes
from imblearn.over_sampling import SMOTE
# Random Forest classifier
from sklearn.ensemble import RandomForestClassifier
# To divide train and test
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix

# Fix the seed
np.random.seed(123456789)

# Read the data
data = []
file = open("./dataset/seismic-bumps.arff", "r")
read=False
for line in file:
    if read:
        data.append(line.strip().split(","))
    if "@data" in line and not read:
        read=True

# Parse categorical or string data to numerical
numerical_values = {"W": 0, "N": 1, "a": 1, "b": 2, "c": 3, "d": 4}
for row in data:
    row[0] = numerical_values[row[0]]
    row[1] = numerical_values[row[1]]
    row[2] = numerical_values[row[2]]
    row[7] = numerical_values[row[7]]

# Parse string to float
for i in range(len(data)):
    new_row = []
    for j in range(len(data[0])):
        new_row.append(float(data[i][j]))
    data[i] = new_row

# Split data and labels
data = np.array(data)
labels = data[:,-1]

# Divide into train and test
data_train, data_test, labels_train, labels_test = train_test_split(data,labels,test_size=0.3, train_size=0.7, random_state=123456789, stratify=labels)

# Balance the classes in train
sm = SMOTE(sampling_strategy=1,random_state = 123456789)
data_train_bal, labels_train_bal = sm.fit_resample(data_train,labels_train)

# Write test and train processed to files
test = open("./dataset/test.dat", "w")
test.write("@data\n")
for d in data_test:
    for e in d:
        test.write(str(e))
    test.write("\n")
test.write("@labels\n")
for l in labels_test:
    test.write(str(l) + "\n")
test.close()

train = open("./dataset/train.dat", "w")
train.write("@data\n")
for d in data_train_bal:
    for e in d:
        train.write(str(e))
    train.write("\n")
train.write("@labels\n")
for l in labels_train_bal:
    train.write(str(l) + "\n")
train.close()
