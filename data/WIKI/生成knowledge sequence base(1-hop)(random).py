import random
#A 读取训练集，并按时间戳递增的顺序将数据分组。
def read_and_group_by_timestamp(file_path):
    data_by_timestamp = {}

    with open(file_path, 'r') as file:
        for line in file:
            parts = line.strip().split('\t')
            #print(parts)
            if len(parts) < 5:
                continue  # Skip invalid lines

            head_entity, relation, tail_entity, timestamp_index, _ = parts
            #print(timestamp_index)
            head_entity = int(head_entity)
            relation = int(relation)
            tail_entity = int(tail_entity)
            timestamp_index = int(timestamp_index)

            if timestamp_index not in data_by_timestamp:
                data_by_timestamp[timestamp_index] = []

            data_by_timestamp[timestamp_index].append((head_entity, relation, tail_entity, timestamp_index, timestamp_index//10))
    return data_by_timestamp

file_path = './train.txt'
knowledge_by_timestamp = read_and_group_by_timestamp(file_path)


#B 创建一个所有实体的list。
entities_no_duplicates = list(range(12554))


#C 根据A的结果，归纳B中每一个实体所对应的一阶知识集合。
correct_result_dict = {}
a = 0
for entity in entities_no_duplicates:
    a = a+1
    if a % 500 == 0:
        print("已经处理了"+str(a)+"条数据")
    entity_triples_corrected = {}
    for time, triples in knowledge_by_timestamp.items():
        matching_triples = []
        for triple in triples:
            if entity in triple[:1] or entity in triple[2:3]:
                matching_triples.append(triple)

        if matching_triples:
            entity_triples_corrected[time] = matching_triples

        else:
            entity_triples_corrected[time] = [[12554, 24, 12554, time, time//10]]

        entity_triples_corrected[time] = random.choice(entity_triples_corrected[time])

    entity_triples_corrected = [entity_triples_corrected[key] for key in sorted(entity_triples_corrected.keys())]
    correct_result_dict[entity] = entity_triples_corrected

print(len(correct_result_dict))

#D 将C的结果进行存储。
import json

with open('./Training_specific_knowledge_sequence(1-hop)(random).json', 'w') as file3:
    # 将字典以JSON格式保存到文件中
    json.dump(correct_result_dict, file3)

















