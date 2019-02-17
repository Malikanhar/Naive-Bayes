# This program is created by Malik Anhar
import numpy as np
import pandas as pd

df = pd.DataFrame(pd.read_csv('TrainsetTugas1ML.csv').as_matrix().tolist())
dataTrain = np.asarray(df)
label = []
for i in range(1,9):
    label.append(list(set(df[i].tolist())))

# CREATING TABLE MODEL (for testing)
table_array = np.zeros((7,4,2))
sum_data = np.zeros((2))
for data in dataTrain:
    for j in range(1,len(data)-1):
        if(data[j] == label[j-1][0]):
            if(data[8] == label[7][0]):
                table_array[j-1][0][0] += 1
            else:
                table_array[j-1][0][1] += 1
        elif(data[j] == label[j-1][1]):
            if(data[8] == label[7][0]):
                table_array[j-1][1][0] += 1
            else:
                table_array[j-1][1][1] += 1
        else:
            if(data[8] == label[7][0]):
                table_array[j-1][2][0] += 1
            else:
                table_array[j-1][2][1] += 1
        if(data[8] == label[7][0]):
            table_array[j-1][3][0] += 1
        else:
            table_array[j-1][3][1] += 1
    if(data[8] == label[7][0]):
        sum_data[0] += 1
    else:
        sum_data[1] += 1
