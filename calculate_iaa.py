import pandas as pd
import numpy as np
from fleiss import fleissKappa

result = pd.read_excel('iaa_sample.xlsx',engine='openpyxl')
relation = {"no_relation":1,"org:members":2,"org:founded":3,"related":4,
            "product":5,"org:event":6,"org:field":7,"alternate":8,"location":9,"per:member_of":10,
            "per:title":11,"poh:type":12,"poh:start_date":13,"poh:end_date":14}
for i in range(len(result)):
    for j in range(7):
        result.iloc[i,j]=relation[result.iloc[i,j]]

result = result.to_numpy()
num_classes = int(np.max(result))

transformed_result = []
for i in range(len(result)):
    temp = np.zeros(num_classes)
    for j in range(len(result[i])):
        temp[int(result[i][j]-1)] += 1
    transformed_result.append(temp.astype(int).tolist())

kappa = fleissKappa(transformed_result,len(result[0]))
