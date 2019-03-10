import numpy as np
import pandas as pd
### DATA TRAIN ###
df = pd.DataFrame(pd.read_csv('TrainsetTugas1ML.csv').as_matrix().tolist())
label = []
[label.append(list(set(df[i].tolist()))) for i in range(1,9)]
for i in range(len(label)-1):
    for j in range(3):
        df = df.replace({label[i][j]: j})
df = df.replace({'>50K' : 0, '<=50K' : 1})
### MAKE A MODEL FOR TESTING ###
model = np.zeros((7,4,2))
for i in np.asarray(df):
    for j in range(1,len(i)-1):
        model[j-1][i[j]][i[8]] += 1
for i in model:
    i[3] = i.sum(axis = 0)
# print(model)
### DATA TEST ###
df = pd.DataFrame(pd.read_csv('TestsetTugas1ML.csv').as_matrix().tolist())
for i in range(len(label)-1):
    for j in range(3):
        df = df.replace({label[i][j]: j})
a = np.asarray(df)
fin = np.ones((len(a), 2))
for i in range(len(a)):
    for j in range(1, len(a[i])):
        for k in range(2):
            fin[i][k] *= model[j-1][a[i][j]][k] / model[j-1][3][k]
    for k in range(2):
        fin[i][k] *= model[j-1][3][k] / (model[j-1][3][0] + model[j-1][3][1])
dfFin = pd.DataFrame(np.argmax(fin, axis = 1)).replace({0 : '>50K', 1 : '<=50K'})
df[8] = dfFin
df.to_csv("Result.csv", encoding='utf-8', index=False, header=False)