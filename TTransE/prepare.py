# 将训练数据和测试数据中的 entity，relation，time 分别赋予一个整数id

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