import numpy as np
from sklearn import svm, model_selection

# 读取数据集
path = "SVM/data.txt"
data = np.loadtxt(path, encoding="UTF=8", dtype=int, delimiter=',', skiprows=1, converters={
    # 定义转换函数，把字符串转换为整数
    0:lambda s:{'青绿':0, '乌黑':1, '浅白':2}[s],
    1:lambda s:{'蜷缩':0, '稍蜷':1, '硬挺':2}[s],
    2:lambda s:{'浊响':0, '沉闷':1, '清脆':2}[s],
    3:lambda s:{'清晰':0, '稍糊':1, '模糊':2}[s],
    4:lambda s:{'凹陷':0, '稍凹':1, '平坦':2}[s],
    5:lambda s:{'硬滑':0, '软粘':1}[s],
    6:lambda s:{'是':1, '否':0}[s]
})

# 划分数据与标签
x,y = np.split(data, indices_or_sections=(6,), axis=1)
# 随机划分训练集和测试集
train_data, test_data, train_label, test_label = model_selection.train_test_split(x, y, random_state=1, train_size=0.7, test_size=0.3)

# 训练SVM分类器
classifier = svm.SVC(kernel='linear', C=1, gamma=10, decision_function_shape='ovo')
classifier.fit(train_data, train_label)

# 测试数据
train_score = classifier.score(train_data, train_label)
test_score = classifier.score(test_data, test_label)
print("训练集准确率：{}".format(train_score))
print("测试集准确率：{}".format(test_score))