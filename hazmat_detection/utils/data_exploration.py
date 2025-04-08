import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import os

NUM_CLASSES = 15 # number of labels
classes = range(0, NUM_CLASSES)
data = pd.DataFrame()
empty = 0 # count number of empty inputs

# for each dataset
for dataset in ["train","valid","test"]:
    directory = f"dataset/{dataset}/labels"
    filenames = os.listdir(directory)
    classCount = {el: 0 for el in range(0,NUM_CLASSES)}
    total = 0
    for filename in filenames:
        with open(os.path.join(directory, filename), 'r') as file:
            lines = file.readlines()
            if len(lines) == 0:
                empty += 1
            else:
                for line in lines:
                    classCount[int(line.strip().split(" ")[0])] += 1
                    total += 1
    # scale values
    values = pd.array(list(classCount.values()))/total
    data[dataset] = values.copy()

print(f"There are {empty} empty lines")
# plot distribution of labels in training, validation and test set
plt.grid(alpha=0.5)
sns.lineplot(data=data)
plt.xlabel("Class")
plt.ylabel("Frequency")
plt.xticks(range(0,NUM_CLASSES))
plt.savefig("./utils/frequency_class.png")