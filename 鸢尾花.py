# 鸢尾花支持向量机分类

# 导入库
import numpy as np
import sklearn
from sklearn.datasets import load_iris
from sklearn.svm import SVC
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

# 函数：绘制决策边界
def decision_regions(model, axis):
    x0, x1 = np.meshgrid(np.linspace(axis[0], axis[1], int((axis[1] - axis[0]) * 100)).reshape(-1, 1),
                         np.linspace(axis[2], axis[3], int((axis[3] - axis[2]) * 100)).reshape(-1, 1))
    X_new = np.c_[x0.ravel(), x1.ravel()]
    y_predict = model.predict(X_new)
    zz = y_predict.reshape(x0.shape)
    custom_cmap = ListedColormap(['#FFFFE0', '#FFF59D', '#98FB98'])
    plt.contourf(x0, x1, zz, cmap=custom_cmap)

# 导入鸢尾花数据集
iris = load_iris()

# 数据集由csv格式保存
fea = iris.data   # 4个特征：花萼长度,花萼宽度,花瓣长度,花瓣宽度
lab = iris.target    # 3种类别：山鸢尾(Iris Setosa)，变色鸢尾(Iris Versicolor)，维吉尼亚鸢尾(Iris Virginica)

print(fea)
print(lab)

# 取前两项特征，前两个类别
fea = fea[lab < 2, :2]
lab = lab[lab < 2]

# 分别画出类别为0和1的点集
plt.scatter(fea[lab == 0, 0], fea[lab == 0, 1], color='green');
plt.scatter(fea[lab == 1, 0], fea[lab == 1, 1], color='orange')
plt.show()

# 对数据进行标准化
standard_Scaler = sklearn.preprocessing.StandardScaler()
standard_Scaler.fit(fea)
X_standard = standard_Scaler.transform(fea)

# 使用线性SVM分类器
svc = SVC(kernel='linear', random_state=1)

# 训练svm
svc.fit(X_standard, lab)

# 绘制决策边界，规定x,y的范围
decision_regions(svc, axis=[-3, 3, -3, 3]);

# 绘制原始数据
plt.scatter(X_standard[lab == 0, 0], X_standard[lab == 0, 1], color='green');
plt.scatter(X_standard[lab == 1, 0], X_standard[lab == 1, 1], color='orange');
plt.show()



