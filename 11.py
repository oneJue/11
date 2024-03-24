import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler

# 读取数据
train = pd.read_csv('input/train.csv')
test = pd.read_csv('input/test.csv')

# 拆分训练数据和标签
X = train.iloc[:, 2:]
y = train['target']

# 数据标准化
scaler = StandardScaler()
X = scaler.fit_transform(X)

# 划分训练集和验证集
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

# 使用随机森林进行训练
model = RandomForestClassifier()
model.fit(X_train, y_train)

# 对测试集进行预测
test_X = test.iloc[:, 1:]
test_X = scaler.transform(test_X)  # 注意也要进行相同的标准化处理
preds = model.predict(test_X)

# 保存预测结果
preds = '\n'.join(map(str, preds))




import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import pandas as pd
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler

# 读取数据
train = pd.read_csv('input/train.csv')
test = pd.read_csv('input/test.csv')

# 拆分训练数据和标签
X = train.iloc[:, 2:].values
y = train['target'].values
test_X = test.iloc[:, 1:].values

# 数据标准化
scaler = StandardScaler()
X = scaler.fit_transform(X)
test_X = scaler.transform(test_X)

# 将数据转为torch.Tensor
X = torch.Tensor(X)
y = torch.Tensor(y)
test_X = torch.Tensor(test_X)

import pandas as pd
from sklearn.linear_model import LogisticRegression

# 读取训练集和测试集
train_data = pd.read_csv('input/train.csv')
test_data = pd.read_csv('input/test.csv')

# 提取特征和目标变量
train_features = train_data.iloc[:, 2:]
train_target = train_data['target']
test_features = test_data.iloc[:, 1:]

# 创建和训练逻辑回归模型
model = LogisticRegression()
model.fit(train_features, train_target)

# 在测试集上做预测
preds = model.predict(test_features)
preds = '\n'.join(map(str, preds))



# 定义MLP模型
class MLP(nn.Module):
    def __init__(self):
        super(MLP, self).__init__()
        self.layers = nn.Sequential(
            nn.Linear(200, 100),
            nn.ReLU(),
            nn.Dropout(0.5),  # 添加 Dropout 层
            nn.Linear(100, 50),
            nn.ReLU(),
            nn.Dropout(0.5),  # 添加 Dropout 层
            nn.Linear(50, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.layers(x)


# 交叉验证
kf = KFold(n_splits=5)

for fold, (train_index, val_index) in enumerate(kf.split(X)):
    X_train, X_val = X[train_index], X[val_index]
    y_train, y_val = y[train_index], y[val_index]

    # 训练模型
    model = MLP()
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    epochs = 200
    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad()
        output = model(X_train)
        loss = criterion(output.squeeze(), y_train)
        loss.backward()
        optimizer.step()

        model.eval()
        val_output = model(X_val)
        val_loss = criterion(val_output.squeeze(), y_val)
        print('Fold: {}, Epoch: {}/{}, Loss: {:.4f}, Val Loss: {:.4f}'.format(fold + 1, epoch + 1, epochs, loss.item(),
                                                                              val_loss.item()))

# 对测试集进行预测
model.eval()
with torch.no_grad():
    preds = model(test_X)
preds = (preds > 0.5).int().squeeze().numpy()

# 保存预测结果
preds = '\n'.join(map(str, preds))

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import pandas as pd
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler

# 读取数据
train = pd.read_csv('input/train.csv')
test = pd.read_csv('input/test.csv')

# 拆分训练数据和标签
X = train.iloc[:, 2:].values
y = train['target'].values
test_X = test.iloc[:, 1:].values

# 将标签转换为one-hot编码
num_classes = 2
y = nn.functional.one_hot(torch.tensor(y, dtype=torch.long), num_classes=num_classes).float()

# 数据标准化
scaler = StandardScaler()
X = scaler.fit_transform(X)
test_X = scaler.transform(test_X)

# 将数据转为torch.Tensor
X = torch.Tensor(X)
test_X = torch.Tensor(test_X)


# 定义MLP模型
class MLP(nn.Module):
    def __init__(self):
        super(MLP, self).__init__()
        self.layers = nn.Sequential(
            nn.Linear(200, 100),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(100, 50),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(50, num_classes),
            nn.Softmax(dim=1)  # 修改激活函数为Softmax
        )

    def forward(self, x):
        return self.layers(x)


# 交叉验证
kf = KFold(n_splits=5)

for fold, (train_index, val_index) in enumerate(kf.split(X)):
    X_train, X_val = X[train_index], X[val_index]
    y_train, y_val = y[train_index], y[val_index]

    # 训练模型
    model = MLP()
    criterion = nn.CrossEntropyLoss()  # 修改损失函数为交叉熵损失函数
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    epochs = 200
    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad()
        output = model(X_train)
        loss = criterion(output, y_train)
        loss.backward()
        optimizer.step()

        model.eval()
        val_output = model(X_val)
        val_loss = criterion(val_output, y_val)
        print('Fold: {}, Epoch: {}/{}, Loss: {:.4f}, Val Loss: {:.4f}'.format(fold + 1, epoch + 1, epochs, loss.item(),
                                                                              val_loss.item()))

# 对测试集进行预测
model.eval()
with torch.no_grad():
    preds = model(test_X)
preds = torch.argmax(preds, dim=1).numpy()  # 修改预测结果的处理方式

# 保存预测结果
preds = '\n'.join(map(str, preds))


print(a, b, c, sep=',')
print(res[i], end=' ')# end的默认值是'\n'，即换行
print("".join(res))



from collections import defaultdict

class ListNode:
    def __init__(self,val,next=None):
        self.val = val
        self.next = next


class TreeNode:
    def __init__(self, val=0, left=None, right=None):
        self.val = val
        self.left = left
        self.right = right

class Solution:
    def networkDelayTime(self, times: List[List[int]], n: int, k: int) -> int:
        """广度优先搜索"""
        # 建图 - 邻接表
        mp = [{} for i in range(n + 1)]
        for u, v, t in times:
            mp[u][v] = t
        print(mp)
        # 记录结点最早收到信号的时间
        r = [-1 for i in range(n + 1)]
        r[k] = 0
        # 队列中存放 [结点，收到信号时间]
        s = deque([[k, 0]])
        while s:
            cur, t = s.popleft()
            for u, v in mp[cur].items():

                art = t + v
                # 仅当结点未收到或收到时间比记录时间更早才更新并入队
                if r[u] == -1 or art < r[u]:
                    r[u] = art
                    s.append([u, art])
        minT = -1
        for i in range(1, n + 1):
            if r[i] == -1:
                return -1
            minT = max(minT, r[i])
        return minT



class Solution:
    def combinationSum(self, candidates: List[int], target: int) -> List[List[int]]:
        dp = [[] for _ in range(target+1)]
        dp[0] = [[]]

        for num in candidates:
            for i in range(1, target+1):
                if i-num >= 0:
                    for pre in dp[i-num]:
                        dp[i].append(pre+[num])
                        print(num,i,dp)
        return dp[-1]


def max_value(T, M, times, values):
    dp = [[0] * (T + 1) for _ in range(M + 1)]

    for i in range(1, M + 1):
        time, value = times[i - 1], values[i - 1]
        for j in range(1, T + 1):
            if j < time:
                dp[i][j] = dp[i - 1][j]
            else:
                dp[i][j] = max(dp[i - 1][j], dp[i - 1][j - time] + value)

    return dp[M][T]

T, M = map(int, input().split())
times = []
values = []
for _ in range(M):
    t, v = map(int, input().split())
    times.append(t)
    values.append(v)

print(max_value(T, M, times, values))        
        
        
        

    matrix = list(zip(*matrix))[::-1]  ##逆时针旋转90度 解包+反转
    matrix[:] = [row[::-1] for row in zip(*matrix)]  ##shun时针旋转90度


n = int(input())  # 这个根据题意设置，表示结点个数

edge = [[float('inf')] * n for i in range(n)]
# 初始化所有边权为无穷大

# 根据题意更新edge[i][j]
# 更新的时候，如果有无向图需要edge[i][j]=edge[j][i]这样设置，否则不用
## flod
# 三重循环 结束
for k in range(n):  # 以k为中转站
    for i in range(n):
        for j in range(n):
            edge[i][j] = min(edge[i][j], edge[i][k] + edge[k][j])
