import numpy as np
import matplotlib
matplotlib.use('TkAgg')  # 上一行直接导入matplotlib可能有问题，需要这两行
import matplotlib.pyplot as plt
from sklearn import linear_model, datasets

# 加载数据集
iris = datasets.load_iris()
X = iris.data[:, 2:4]  # 这里我们只取前两个作为特征
Y = iris.target

h = .02  # step size in the mesh

logreg = linear_model.LogisticRegression(C=1e5)

# 创建了一个分类器实例去拟合数据
logreg.fit(X, Y)

# 绘制决策边界。为此，我们将为每个对象分配一个颜色
# 网格中的点 [x_min, x_max]x[y_min, y_max].
x_min, x_max = X[:, 0].min() - .5, X[:, 0].max() + .5
y_min, y_max = X[:, 1].min() - .5, X[:, 1].max() + .5
xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
Z = logreg.predict(np.c_[xx.ravel(), yy.ravel()])

# 将结果放入颜色图中
Z = Z.reshape(xx.shape)
plt.figure(1, figsize=(4, 3))
plt.pcolormesh(xx, yy, Z, cmap=plt.cm.Paired)

# Plot also the training points
plt.scatter(X[:, 0], X[:, 1], c=Y, edgecolors='k', cmap=plt.cm.Paired)
plt.xlabel('Sepal length')
plt.ylabel('Sepal width')

plt.xlim(xx.min(), xx.max())
plt.ylim(yy.min(), yy.max())
plt.xticks(())
plt.yticks(())

plt.show()
# 下面显示的是iris数据集中的逻辑回归分类器决策边界。数据点根据其标签进行着色。