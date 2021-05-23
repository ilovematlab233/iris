# 导入库函数
import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris

# 载入数据集
# 最先总是遇到梯度不下降的情况，将数据集中标签0换成-1
def create_data():
    iris = load_iris();
    df = pd.DataFrame(iris.data, columns=iris.feature_names);
    df['label'] = iris.target;
    df.columns = ['sepal length', 'sepal width', 'petal length', 'petal width', 'label'];
    data = np.array(df.iloc[:100, [0, 1, -1]]);
    for i in range(len(data)):
        if data[i,-1] == 0:
            data[i,-1] = -1
    # print(data)
    return data[:,:2], data[:,-1]

# 得到数据与标签
data, lab = create_data();
#print(data,lab)

# 初始化向量机参数
b = 0;
alpha = np.zeros((np.shape(data)[0],1))
# print(np.shape(data)[0])
# print('the alpha is: ',alpha)

# 设置常规参数
sla = 0.6   # 松弛变量
max_gen = 40    # 最大迭代次数
iter_num = 0    # 初始化迭代次数

while  max_gen > iter_num :

    # alpha优化次数,每迭代一次就初始化一次
    opti_num = 0

    # 对每个alpha进行优化
    for i in range(np.shape(data)[0]):

        # 输出目标值
        # 最先总是遇到维数不对的问题，查阅资料后用np.newaxis解决
        pre_i = np.sum(alpha * lab[:, np.newaxis] * data * data[i, :]) + b
        # print(pre)

        # 计算样本i的偏差
        Ei = pre_i - lab[i]

        # 判断是否满足KKT条件
        if (((lab[i] * Ei < -0.01) and (alpha[i] < sla)) or
                ((lab[i] * Ei > 0.01) and (alpha[i] > 0))):
            # 满足优化条件，选择一个j不等于i进行优化
            j = int(random.uniform(0, np.shape(data)[0]))
            # print(j)
            if j == i:
                temp = 1
                while temp:
                    j = int(random.uniform(0, np.shape(data)[0]))
                    if j != i:
                        temp = 0

            # 预测样本 j 的结果
            pre_j = np.sum(alpha * lab[:, np.newaxis] * data * data[j, :]) + b
            # print(pre_j)

            # 样本j误差
            Ej = pre_j - lab[j]

            # 更新上下限
            if (lab[i] != lab[j]):# 类标签不同，即异号
                L = max(0, alpha[j] - alpha[i])
                H = min(sla, alpha[j] - alpha[i] + sla)
            else:   # 类标签相同
                L = max(0, alpha[i] + alpha[j] - sla)
                H = min(sla, alpha[i] + alpha[j])

            # 上界与下界相等，不再优化
            if L == H:
                continue

            # 计算alphas[j]的最优修改量eta
            eta = np.sum(data[i, :] * data[i, :]) + np.sum(data[j, :] * data[j, :]) - \
                  2. * np.sum(data[i, :] * data[j, :])
            # !!!!一定要讨论eta<=0的情况！！不然会出现alpha[j]更新错误！！
            # 坑死我了
            if eta <= 0:
                continue

            # 复制原来的数据
            # 要使原来的数据与现在的数据不占相同内存！！
            alpha_pre_i = alpha[i].copy()
            alpha_pre_j = alpha[j].copy()

            # 更新 alpha(j)并限制范围
            alpha[j] = alpha_pre_j + lab[j] * (Ei - Ej) / eta
            if alpha[j] < L:
                alpha[j] = L
            elif alpha[j] > H:
                alpha[j] = H

            # alpha不再更新或者更新量极小时，退出
            if (0 < (alpha[j] - alpha_pre_j) < 0.0001 or (-0.0001 < alpha[j] - alpha_pre_j) < 0):
                continue

            # 更新alpha[i]
            alpha[i] = alpha_pre_i + lab[i] * lab[j] * (alpha_pre_j - alpha[j])

            # 计算参数 b
            bi = b - Ei - lab[i] * (alpha[i] - alpha_pre_i) * np.sum(data[i, :] * data[i, :]) - \
                 lab[j] * (alpha[j] - alpha_pre_j) * np.sum(data[i, :] * data[j, :])
            bj = b - Ej - lab[i] * (alpha[i] - alpha_pre_i) * np.sum(data[i, :] * data[j, :]) - \
                 lab[j] * (alpha[j] - alpha_pre_j) * np.sum(data[j, :] * data[j, :])

            # b 的更新条件
            if (0 < alpha[i] < sla):
                b = bi
            elif (0 < alpha[j] < sla):
                b = bj
            else:
                b = (bi + bj) / 2

            # 更新完成
            opti_num = opti_num + 1

    # 判断是否迭代了iter_num次
    if (opti_num == 0):
        iter_num += 1
    else:
        iter_num = 0

# 计算权值W
w = np.sum(alpha * lab[:, np.newaxis] * data, axis = 0)
# print('the w11 is: ', w )

# 计算预测结果
x = np.linspace(1, 10)
y = np.array([(-b - w[0] * x[i]) / w[1] for i in range(x.shape[0])])

# 显示结果
plt.scatter(data[lab == -1, 0], data[lab == -1, 1], color='red');
plt.scatter(data[lab == 1, 0], data[lab == 1, 1], color='green');
plt.plot(x, y)
plt.xlim(4,7.5)
plt.ylim(1,5)
plt.show()