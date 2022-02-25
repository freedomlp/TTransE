import random
import math
import numpy as np
import copy
import time

entity2id = {}
relation2id = {}
time2id = {}

# 读入训练数据
def data_loader(file):
    file1 = file + "entity2id.txt"
    file2 = file + "relation2id.txt"
    file3 = file + "time2id.txt"
    file4 = file + "icews_2014_train.txt"

    with open(file1, 'r', encoding='utf-8') as f1, open(file2, 'r', encoding='utf-8') as f2, open(file3, 'r', encoding='utf-8') as f3:
        lines1 = f1.readlines()
        lines2 = f2.readlines()
        lines3 = f3.readlines()

        for line in lines1:
            line = line.strip().split('\t')
            if len(line) != 2:
                continue
            entity2id[line[0]] = int(line[1])

        for line in lines2:
            line = line.strip().split('\t')
            if len(line) != 2:
                continue
            relation2id[line[0]] = int(line[1])

        for line in lines3:
            line = line.strip().split('\t')
            if len(line) != 2:
                continue
            time2id[line[0]] = int(line[1])

    entity_set = set()   # 创建一个无序不重复元素集
    relation_set = set()
    time_set = set()
    quadruple_list = []

    with open(file4, 'r', encoding='utf-8') as f:
        content = f.readlines()
        for line in content:
            quadruple = line.strip().split("\t")
            if len(quadruple) != 4:
                continue

            # e.g. South Korea  Criticize or denounce   North Korea 2014-05-13
            s_ = entity2id[quadruple[0]]
            r_ = relation2id[quadruple[1]]
            o_ = entity2id[quadruple[2]]
            t_ = time2id[quadruple[3]]

            quadruple_list.append([s_, r_, o_, t_])  # 三元组由头实体、尾实体、关系的 id 组成

            entity_set.add(s_)
            relation_set.add(r_)
            entity_set.add(o_)
            time_set.add(t_)
    return entity_set, relation_set, time_set, quadruple_list


def distanceL2(s, r, o, t):
    # 为方便求梯度，去掉sqrt
    return np.sum(np.square(s + r + t - o))


def distanceL1(s, r, o, t):
    return np.sum(np.fabs(s + r + t - o))


class TTransE:
    def __init__(self, entity_set, relation_set, time_set, quadruple_list,
                 embedding_dim=100, learning_rate=0.01, margin=1, L1=True):
        self.embedding_dim = embedding_dim
        self.learning_rate = learning_rate
        self.margin = margin
        self.entity = entity_set
        self.relation = relation_set
        self.time = time_set
        self.quadruple_list = quadruple_list
        self.L1 = L1
        self.loss = 0

    def emb_initialize(self):
        relation_dict = {}
        entity_dict = {}
        time_dict = {}

        # 通过随机的方式生成向量
        for relation in self.relation:
            r_emb_temp = np.random.uniform(-6 / math.sqrt(self.embedding_dim),
                                           6 / math.sqrt(self.embedding_dim),
                                           self.embedding_dim)
            relation_dict[relation] = r_emb_temp / np.linalg.norm(r_emb_temp, ord=2)

        for entity in self.entity:
            e_emb_temp = np.random.uniform(-6 / math.sqrt(self.embedding_dim),
                                           6 / math.sqrt(self.embedding_dim),
                                           self.embedding_dim)
            entity_dict[entity] = e_emb_temp / np.linalg.norm(e_emb_temp, ord=2)

        for time in self.time:
            e_emb_temp = np.random.uniform(-6 / math.sqrt(self.embedding_dim),
                                           6 / math.sqrt(self.embedding_dim),
                                           self.embedding_dim)
            time_dict[time] = e_emb_temp / np.linalg.norm(e_emb_temp, ord=2)
        # 将 relation，entity，time 更新为为 id 和向量对应的字典
        self.relation = relation_dict
        self.entity = entity_dict
        self.time = time_dict

    def train(self, epochs):
        nbatches = 400
        batch_size = len(self.quadruple_list) // nbatches
        print("batch size: ", batch_size)
        for epoch in range(epochs):
            start = time.time()
            self.loss = 0

            for k in range(nbatches):
                Sbatch = random.sample(self.quadruple_list, batch_size)
                Tbatch = []
                for quadruple in Sbatch:
                    corrupted_quadruple = self.Corrupt(quadruple)
                    Tbatch.append((quadruple, corrupted_quadruple))

                self.update_embeddings(Tbatch)

            end = time.time()
            print("epoch: ", epoch, "cost time: %s" % (round((end - start), 3)))
            print("loss: ", self.loss)

            # 保存临时结果
            if epoch % 10 == 0:
                with open("res\\entity_temp.txt", 'w', encoding='utf-8') as f_e:
                    for e in self.entity.keys():
                        f_e.write(str(e) + "\t")
                        f_e.write(str(list(self.entity[e])))
                        f_e.write("\n")
                with open("res\\relation_temp.txt", 'w', encoding='utf-8') as f_r:
                    for r in self.relation.keys():
                        f_r.write(str(r) + "\t")
                        f_r.write(str(list(self.relation[r])))
                        f_r.write("\n")
                with open("res\\time_temp.txt", 'w', encoding='utf-8') as f_t:
                    for t in self.time.keys():
                        f_t.write(str(t) + "\t")
                        f_t.write(str(list(self.time[t])))
                        f_t.write("\n")
                with open("res\\result_temp.txt", 'a', encoding='utf-8') as f_s:
                    f_s.write("epoch: %d\tloss: %s\n" % (epoch, self.loss))

        print("写入文件...")
        with open("res\\entity_50dim_batch400.txt", 'w', encoding='utf-8') as f1:
            for e in self.entity.keys():
                f1.write(str(e) + "\t")
                f1.write(str(list(self.entity[e])))
                f1.write("\n")

        with open("res\\relation_50dim_batch400.txt", 'w', encoding='utf-8') as f2:
            for r in self.relation.keys():
                f2.write(str(r) + "\t")
                f2.write(str(list(self.relation[r])))
                f2.write("\n")
        with open("res\\time_50dim_batch400.txt", 'w', encoding='utf-8') as f3:
            for t in self.time.keys():
                f3.write(str(t) + "\t")
                f3.write(str(list(self.time[t])))
                f3.write("\n")
        print("写入完成")

    # 通过随机替换头实体或尾实体得到负例
    def Corrupt(self, quadruple):
        corrupted_quadruple = copy.deepcopy(quadruple)
        seed = random.random()
        if seed > 0.5:
            # 替换head
            head = quadruple[0]
            rand_head = head
            while (rand_head == head):
                rand_head = random.randint(0, len(self.entity) - 1)
            corrupted_quadruple[0] = rand_head

        else:
            # 替换tail
            tail = quadruple[2]
            rand_tail = tail
            while (rand_tail == tail):
                rand_tail = random.randint(0, len(self.entity) - 1)
            corrupted_quadruple[2] = rand_tail
        return corrupted_quadruple

    def update_embeddings(self, Tbatch):
        # 不要每次都深拷贝整个字典，只拷贝当前 Tbatch 中出现的四元组对应的向量
        entity_updated = {}
        relation_updated = {}
        time_updated = {}
        for quadruple, corrupted_quadruple in Tbatch:
            # 取原始的vector计算梯度
            s_correct = self.entity[quadruple[0]]
            o_correct = self.entity[quadruple[2]]

            s_corrupt = self.entity[corrupted_quadruple[0]]
            o_corrupt = self.entity[corrupted_quadruple[2]]

            relation = self.relation[quadruple[1]]
            time = self.time[quadruple[3]]


            if quadruple[0] in entity_updated.keys():
                pass
            else:
                entity_updated[quadruple[0]] = copy.copy(self.entity[quadruple[0]])
            if quadruple[2] in entity_updated.keys():
                pass
            else:
                entity_updated[quadruple[2]] = copy.copy(self.entity[quadruple[2]])
            if corrupted_quadruple[0] in entity_updated.keys():
                pass
            else:
                entity_updated[corrupted_quadruple[0]] = copy.copy(self.entity[corrupted_quadruple[0]])
            if corrupted_quadruple[2] in entity_updated.keys():
                pass
            else:
                entity_updated[corrupted_quadruple[2]] = copy.copy(self.entity[corrupted_quadruple[2]])
            if quadruple[1] in relation_updated.keys():
                pass
            else:
                relation_updated[quadruple[1]] = copy.copy(self.relation[quadruple[1]])
            if quadruple[3] in time_updated.keys():
                pass
            else:
                time_updated[quadruple[3]] = copy.copy(self.time[quadruple[3]])


            if self.L1:
                dist_correct = distanceL1(s_correct, relation, o_correct, time)
                dist_corrupt = distanceL1(s_corrupt, relation, o_corrupt, time)
            else:
                dist_correct = distanceL2(s_correct, relation, o_correct, time)
                dist_corrupt = distanceL2(s_corrupt, relation, o_corrupt, time)

            err = self.hinge_loss(dist_correct, dist_corrupt)

            # err > 0时才更新参数
            if err > 0:
                self.loss += err
                grad_pos = 2 * (s_correct + relation + time - o_correct)
                grad_neg = 2 * (s_corrupt + relation + time - o_corrupt)
                if self.L1:
                    for i in range(len(grad_pos)):
                        if (grad_pos[i] > 0):
                            grad_pos[i] = 1
                        else:
                            grad_pos[i] = -1

                    for i in range(len(grad_neg)):
                        if (grad_neg[i] > 0):
                            grad_neg[i] = 1
                        else:
                            grad_neg[i] = -1

                # 梯度求导参考 https://blog.csdn.net/weixin_42348333/article/details/89598144
                entity_updated[quadruple[0]] -= self.learning_rate * grad_pos
                entity_updated[quadruple[2]] -= (-1) * self.learning_rate * grad_pos

                entity_updated[corrupted_quadruple[0]] -= (-1) * self.learning_rate * grad_neg
                entity_updated[corrupted_quadruple[2]] -= self.learning_rate * grad_neg

                relation_updated[quadruple[1]] -= self.learning_rate * grad_pos
                relation_updated[quadruple[1]] -= (-1) * self.learning_rate * grad_neg

                time_updated[quadruple[3]] -= self.learning_rate * grad_pos
                time_updated[quadruple[3]] -= (-1) * self.learning_rate * grad_neg


        # batch norm
        for i in entity_updated.keys():
            entity_updated[i] /= np.linalg.norm(entity_updated[i])
            self.entity[i] = entity_updated[i]
        for i in relation_updated.keys():
            relation_updated[i] /= np.linalg.norm(relation_updated[i])
            self.relation[i] = relation_updated[i]
        for i in time_updated.keys():
            time_updated[i] /= np.linalg.norm(time_updated[i])
            time_updated[i] /= np.linalg.norm(time_updated[i])
            self.time[i] = time_updated[i]
        return

    def hinge_loss(self, dist_correct, dist_corrupt):
        return max(0, dist_correct - dist_corrupt + self.margin)

if __name__ == '__main__':
    file = "icews14\\"
    print("load file...")
    entity_set, relation_set, time_set, quadruple_list = data_loader(file)
    print("Complete load. entity : %d , relation : %d , time : %d , quadruple : %d" % (len(entity_set), len(relation_set), len(time_set), len(quadruple_list)))
    TTransE = TTransE(entity_set, relation_set, time_set, quadruple_list, embedding_dim=50, learning_rate=0.01, margin=1, L1=True)
    TTransE.emb_initialize()
    TTransE.train(epochs=1001)