import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
y_pred = ['a', 'b', 'b', 'b', 'c', 'c']#测试集
y_true = ['a', 'b', 'a', 'c', 'b', 'b']#真实
def F1_score (y_pred,y_true):
    res = {}
    qqq = {'TP': 0, 'FP': 0, 'FN': 0}
    for i in y_pred:
        res[i] = res.get(i, 0) + 1
    for i in y_true:
        res[i] = res.get(i, 0) + 1
    qqq = pd.DataFrame(qqq, index=list(res.keys()))
    for i in res.keys():
        for j in range(len(y_pred)):
            if (y_pred[j] == i) and (y_pred[j] == y_true[j]):
                qqq['TP'][i] = qqq['TP'][i] + 1
            if y_pred[j] == i:
                qqq['FN'][i] = qqq['FN'][i] + 1
            if y_true[j] == i:
                qqq['FP'][i] = qqq['FP'][i] + 1
    precision = []
    recall = []
    for i in res.keys():
        qqq['FN'][i] = qqq['FN'][i] - qqq['TP'][i]
        qqq['FP'][i] = qqq['FP'][i] - qqq['TP'][i]
        precision.append(qqq['TP'][i] / (qqq['TP'][i] + qqq['FP'][i]))
        recall.append(qqq['TP'][i] / (qqq['TP'][i] + qqq['FN'][i]))
    F1_score = [2 * precision[i] * recall[i] / (precision[i] + recall[i]) for i in range(len(precision)) if
                recall[i] != 0 or precision[i] != 0]
    F1_score = np.array(F1_score)
    print(np.sum(F1_score) / len(recall))
F1_score(y_pred,y_true)
sum=0
for i in range(1001):
    sum=sum+i
print(sum)

print(np.sum([i for i in range(1001)]))

df=pd.DataFrame(np.random.rand(10,5),columns=list("abcde"),index=range(1,11))
df.plot.bar()
df.plot.box()
df.plot.scatter(x='a',y='b')
df.plot.barh(stacked=True)
df.plot()
plt.show()
