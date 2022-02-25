# 高级实训-作业三

18342066 鲁沛

## TTransE

### 1. 原理介绍

#### Temporal Knowledge Graph (TKG)

在知识图谱的三元组表示基础上，添加了**时间**这一维度的考量，形成了新的四元组：(s, r, o, t)。

例如：(Barack Obama, visits, Ukraine, 2014-07-08)

 ![](img\1.png)



#### TTransE

由于多了时间维度的考量，TTransE 的得分函数变为：

 ![](\img\2.png" style="zoom:60%;" )

其余方面则与 TransE 没有较大区别，由于上次作业已经详细介绍了 TransE 的原理，因此本次不再赘述。



### 2. 模型细节

#### 数据预处理

首先将icews14 **训练数据集**中的实体、关系、时间都转化为 str 和 int 相对应的字典序，即赋予每个实体、关系、时间一个 id 来代表它，训练结束后生成的表征实体、关系、时间的向量也是和其 id 相对应的。

```python
entity_set = set()
relation_set = set()
time_set = set()
file_1 = "icews14\\icews_2014_train.txt"
file_2 = "icews14\\entity2id.txt"
file_3 = "icews14\\relation2id.txt"
file_4 = "icews14\\time2id.txt"

with open(file_1, 'r', encoding='utf-8') as f1:
    lines = f1.readlines()
    for line in lines:
        line = line.strip().split('\t')
        if len(line) != 4:
            continue
        entity_set.add(line[0])
        relation_set.add(line[1])
        entity_set.add(line[2])
        time_set.add(line[3])

i = 0
with open(file_2, 'w', encoding='utf-8') as f2:
    for entity in entity_set:
        text = entity + '\t' + str(i) + '\n'
        f2.write(text)
        i += 1

i = 0
with open(file_3, 'w', encoding='utf-8') as f3:
    for relation in relation_set:
        text = relation + '\t' + str(i) + '\n'
        f3.write(text)
        i += 1

i = 0
with open(file_4, 'w', encoding='utf-8') as f4:
    for time in time_set:
        text = time + '\t' + str(i) + '\n'
        f4.write(text)
        i += 1
```

这里会遇到一个问题：测试集中有些数据是在训练集中没有出现过的，因此训练结束后并没有得到这些新数据相对应的 id 和向量。因此在**测试数据集**中测试的时候如果遇到新的数据，则给其赋予新的 id，并直接生成一个随机向量来表征它。

```python
 # 对于 test 中未在 train 中出现过的 entity，relation 和 time，对其进行随机初始化
if quadruple[0] in entity2id:
    s_ = entity2id[quadruple[0]]
else:
    entity2id[quadruple[0]] = len_e
    s_ = len_e
    entity_dict[s_] = np.random.uniform(-6 / math.sqrt(50), 6 / math.sqrt(50), 50)
    len_e += 1

if quadruple[1] in relation2id:
    r_ = relation2id[quadruple[1]]
else:
    relation2id[quadruple[1]] = len_r
    r_ = len_r
    relation_dict[r_] = np.random.uniform(-6 / math.sqrt(50), 6 / math.sqrt(50), 50)
    len_r += 1

if quadruple[2] in entity2id:
    o_ = entity2id[quadruple[2]]
else:
    entity2id[quadruple[2]] = len_e
    o_ = len_e
    entity_dict[o_] = np.random.uniform(-6 / math.sqrt(50), 6 / math.sqrt(50), 50)
    len_e += 1

if quadruple[3] in time2id:
    t_ = time2id[quadruple[3]]
else:
    time2id[quadruple[3]] = len_t
    t_ = len_t
    time_dict[t_] = np.random.uniform(-6 / math.sqrt(50), 6 / math.sqrt(50), 50)
    len_t += 1

test_quadruple.append([s_, r_, o_, t_])
```



#### 数据读入

将通过数据预处理得到的 entity2id，relation2id，time2id 读入，并且读入训练数据



#### 距离函数

距离通过 L1 范数或 L2 范数来表征

```python
def distanceL2(s, r, o, t):
    # 为方便求梯度，去掉sqrt
    return np.sum(np.square(s + r + t - o))


def distanceL1(s, r, o, t):
    return np.sum(np.fabs(s + r + t - o))
```



#### 类的设计

```python
class TTransE:
    # 构造函数，确定一些超参数
    def __init__(self, entity_set, relation_set, time_set, quadruple_list,
                 embedding_dim=100, learning_rate=0.01, margin=1, L1=True)
    # 表征向量采用随机的方式初始化
    def emb_initialize(self)
    # 训练过程
    def train(self, epochs)
    # 得到负例
    def Corrupt(self, quadruple)
    # 向量更新函数
    def update_embeddings(self, Tbatch)
    # 计算 loss
    def hinge_loss(self, dist_correct, dist_corrupt)
```



#### 参数初始化

- embedding_dim：表征向量的维度，设置为50
- learning_rate：学习率，设定为0.01
- margin：用于计算损失函数，设置为1
- 距离函数：采用 L1 范数
- epoch：1000



#### 损失函数

和 TTransE 基本一致：
$$
L=\gamma+d(s+r+t,o)-d(s'+r+t,o')
$$


### 3. 实验结果

训练过程结束后，loss = 1169.334723892541

测试集上的测试结果：

```
entity hits@1: 0.05902041727100301
entity hits@3: 0.1621109003681803
entity hits@10: 0.3238313064822046
entity mean rank: 349.98850831194915
```

可以看到，结果并不是很好，可以在超参数的设置上进一步探索更优的实现。
