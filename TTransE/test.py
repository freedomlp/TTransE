import numpy as np
import math
import operator
import json
from TTransE import data_loader, entity2id, relation2id, time2id


def dataloader(entity_file, relation_file, time_file, test_file):
    entity_dict = {}
    relation_dict = {}
    time_dict = {}
    test_quadruple = []

    with open(entity_file, 'r', encoding='utf-8') as e_f:
        lines = e_f.readlines()
        for line in lines:
            entity, embedding = line.strip().split('\t')
            embedding = np.array(json.loads(embedding))
            entity_dict[int(entity)] = embedding

    with open(relation_file, 'r', encoding='utf-8') as r_f:
        lines = r_f.readlines()
        for line in lines:
            relation, embedding = line.strip().split('\t')
            embedding = np.array(json.loads(embedding))
            relation_dict[int(relation)] = embedding

    with open(time_file, 'r', encoding='utf-8') as t_f:
        lines = t_f.readlines()
        for line in lines:
            time, embedding = line.strip().split('\t')
            embedding = np.array(json.loads(embedding))
            time_dict[int(time)] = embedding

    with open(test_file, 'r', encoding='utf-8') as test_f:
        lines = test_f.readlines()
        len_e = len(entity2id)
        len_r = len(relation2id)
        len_t = len(time2id)
        for line in lines:
            quadruple = line.strip().split('\t')
            if len(quadruple) != 4:
                continue

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
    return entity_dict, relation_dict, time_dict, test_quadruple


def distance(s, r, o, t):
    return np.linalg.norm(s + r + t - o)


class Test:
    def __init__(self, entity_dict, relation_dict, time_dict, test_quadruple, train_quadruple, isFit=True):
        self.entity_dict = entity_dict
        self.relation_dict = relation_dict
        self.time_dict = time_dict
        self.test_quadruple = test_quadruple
        self.train_quadruple = train_quadruple
        print("len entity_dict: ", len(self.entity_dict), "  len relation_dict: ", len(self.relation_dict),
              "  len time_dict: ", len(self.time_dict))
        print("len train_quadruple: ", len(self.train_quadruple), "  len test_quadruple: ", len(self.test_quadruple))
        self.isFit = isFit

        self.hits1 = 0
        self.hits3 = 0
        self.hits10 = 0
        self.mean_rank = 0

    def rank(self):
        hits_1 = 0
        hits_3 = 0
        hits_10 = 0
        rank_sum = 0
        step = 1
        for quadruple in self.test_quadruple:
            rank_head_dict = {}
            rank_tail_dict = {}

            for entity in self.entity_dict.keys():
                if self.isFit:
                    if [entity, quadruple[1], quadruple[2], quadruple[3]] not in self.train_quadruple:
                        s_emb = self.entity_dict[entity]
                        r_emb = self.relation_dict[quadruple[1]]
                        o_emb = self.entity_dict[quadruple[2]]
                        t_emb = self.time_dict[quadruple[3]]
                        rank_head_dict[entity] = distance(s_emb, r_emb, o_emb, t_emb)
                else:
                    s_emb = self.entity_dict[entity]
                    r_emb = self.relation_dict[quadruple[1]]
                    o_emb = self.entity_dict[quadruple[2]]
                    t_emb = self.time_dict[quadruple[3]]
                    rank_head_dict[entity] = distance(s_emb, r_emb, o_emb, t_emb)

                if self.isFit:
                    if [quadruple[0], quadruple[1], entity, quadruple[3]] not in self.train_quadruple:
                        s_emb = self.entity_dict[quadruple[0]]
                        r_emb = self.relation_dict[quadruple[1]]
                        o_emb = self.entity_dict[entity]
                        t_emb = self.time_dict[quadruple[3]]
                        rank_tail_dict[entity] = distance(s_emb, r_emb, o_emb, t_emb)
                else:
                    s_emb = self.entity_dict[quadruple[0]]
                    r_emb = self.relation_dict[quadruple[1]]
                    o_emb = self.entity_dict[entity]
                    t_emb = self.time_dict[quadruple[3]]
                    rank_tail_dict[entity] = distance(s_emb, r_emb, o_emb, t_emb)

            rank_head_sorted = sorted(rank_head_dict.items(), key=operator.itemgetter(1))
            rank_tail_sorted = sorted(rank_tail_dict.items(), key=operator.itemgetter(1))

            # rank_sum and hits
            for i in range(len(rank_head_sorted)):
                if quadruple[0] == rank_head_sorted[i][0]:
                    if i == 0:
                        hits_1 += 1
                    if i < 3:
                        hits_3 += 1
                    if i < 10:
                        hits_10 += 1
                    rank_sum = rank_sum + i + 1
                    break

            for i in range(len(rank_tail_sorted)):
                if quadruple[2] == rank_tail_sorted[i][0]:
                    if i == 0:
                        hits_1 += 1
                    if i < 3:
                        hits_3 += 1
                    if i < 10:
                        hits_10 += 1
                    rank_sum = rank_sum + i + 1
                    break

            step += 1
            if step % 200 == 0:
                print("step: ", step)
                print("hits@1: ", hits_1 / (2 * step))
                print("hits@3: ", hits_3 / (2 * step))
                print("hits@10: ", hits_10 / (2 * step))
        self.hits1 = hits_1 / (2 * len(self.test_quadruple))
        self.hits3 = hits_3 / (2 * len(self.test_quadruple))
        self.hits10 = hits_10 / (2 * len(self.test_quadruple))
        self.mean_rank = rank_sum / (2 * len(self.test_quadruple))


if __name__ == '__main__':
    _, _, _, train_quadruple = data_loader("icews14\\")
    entity_dict, relation_dict, time_dict, test_quadruple = dataloader("res\\entity_50dim_batch400.txt",
                                                                       "res\\relation_50dim_batch400.txt",
                                                                       "res\\time_50dim_batch400.txt",
                                                                       "icews14\\icews_2014_test.txt")

    test = Test(entity_dict, relation_dict, time_dict, test_quadruple, train_quadruple, isFit=False)
    test.rank()
    f = open("res\\result.txt", 'w', encoding='utf-8')
    f.write("entity hits@1: " + str(test.hits1) + '\n')
    f.write("entity hits@3: " + str(test.hits3) + '\n')
    f.write("entity hits@10: " + str(test.hits10) + '\n')
    f.write("entity mean rank: " + str(test.mean_rank) + '\n')
    f.close()
