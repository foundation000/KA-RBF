import json

# A 以文本读模式打开文件
with open('./Training_specific_knowledge_sequence(1-hop)(random).json', 'r') as file:
    # 从文件中加载字典
    train_entity_knowledge_by_timestamp = json.load(file)
correct_result_dict = train_entity_knowledge_by_timestamp

'''
# 一个correct_result_dict的例子。
correct_result_dict = {
    10289: {0: [[10289, 9, 10290], [10290, 9, 10289]], 1: [[10289, 9, 11235]], 2: [[10290, 30, 10289]]},
    8429: {0: [[8429, 9, 8468], [8429, 2, 8468], [8429, 94, 868]], 1: [[8429, 29, 12], [8429, 49, 8468], [8429, 9, 12334]]},
    9384: {0: [[9384, 9, 9385]], 2: [[9384, 14, 9385]]},
    10290: {0: [[10289, 3, 10290]]},
    9385: {0: [[9384, 9, 9385]], 2: [[9384, 14, 9385]]},
    8468: {0: [[83687, 10, 8468], [8468, 120, 84]], 1: [[8429, 14, 8468]]},
    11235: {0: [[11235, 10, 8468], [8468, 120, 11235]], 1: [[11235, 14, 8468]]}
}
'''

knowledge_list = []
file_path = "./train.txt"
with open(file_path, "r") as file:
    for line in file:
        parts = line.strip().split('\t')
        if len(parts) < 5:
            continue
        head_entity, relation, tail_entity, timestamp, nothing = parts
        knowledge_list.append((int(head_entity), int(relation), int(tail_entity), int(timestamp), int(nothing)))

'''
# 一个knowledge_list的例子。
knowledge_list = [
    (10289, 9, 10290, 3, 0),
    (9384, 9, 9385, 1, 0),
    (8429, 9, 8468, 2, 0),
    (10289, 10, 11235, 1, 1)
]
'''

print("该数据集中的训练样本个数为：" + str(len(knowledge_list)))
counts_head_entity = 0
counts_tail_entity = 0
counts_relation = 0



# B 开始统计
a = 0
for head_entity, relation, tail_entity, time, key in knowledge_list:
    a = a + 1
    if a % 500 == 0:
        print("已经处理了"+str(a)+"条数据")

    time_points_head = correct_result_dict[str(head_entity)]

    count_head_entity = 0
    count_head_entity_1_1 = 0
    count_head_entity_2_1 = 0

    for time_point_triples_head in time_points_head:
        if int(time_point_triples_head[3]) < time:
            if tail_entity in time_point_triples_head[:1] or tail_entity in time_point_triples_head[2:]:
                count_head_entity += 1

            if count_head_entity != 0:
                break

            for triple_head in [time_point_triples_head]:
                count_head_entity_1 = 0
                count_head_entity_2 = 0
                if head_entity in triple_head[:1] and triple_head[2] != 12554:
                    time_points_head_1 = correct_result_dict[str(triple_head[2])]
                    for time_point_triples_head_1 in time_points_head_1:
                        if int(time_point_triples_head_1[3]) < time:
                            if tail_entity in time_point_triples_head_1[:1] or tail_entity in time_point_triples_head_1[2:]:
                                count_head_entity_1 += 1

                if count_head_entity_1 != 0:
                    count_head_entity_1_1 += 1
                    break

                if head_entity in triple_head[2:] and triple_head[0] != 12554:
                    time_points_head_2 = correct_result_dict[str(triple_head[0])]
                    for time_points_triples_head_2 in time_points_head_2:
                        if int(time_points_triples_head_2[3]) < time:
                            if tail_entity in time_points_triples_head_2[:1] or tail_entity in time_points_triples_head_2[2:]:
                                count_head_entity_2 += 1

                if count_head_entity_2 != 0:
                    count_head_entity_2_1 += 1
                    break

            if count_head_entity_1_1 != 0 or count_head_entity_2_1 != 0:
                break

    if count_head_entity != 0 or count_head_entity_1_1 != 0 or count_head_entity_2_1 != 0:
        counts_head_entity += 1


    time_points_tail = correct_result_dict[str(tail_entity)]

    count_tail_entity = 0
    count_tail_entity_1_1 = 0
    count_tail_entity_2_1 = 0

    for time_point_triples_tail in time_points_tail:
        if int(time_point_triples_tail[3]) < time:
            if head_entity in time_point_triples_tail[:1] or head_entity in time_point_triples_tail[2:]:
                count_tail_entity += 1

            if count_tail_entity != 0:
                break

            for triple_tail in [time_point_triples_tail]:
                count_tail_entity_1 = 0
                count_tail_entity_2 = 0
                if tail_entity in triple_tail[:1] and triple_tail[2] != 12554:
                    time_points_tail_1 = correct_result_dict[str(triple_tail[2])]
                    for time_point_triples_tail_1 in time_points_tail_1:
                        if int(time_point_triples_tail_1[3]) < time:
                            if head_entity in time_point_triples_tail_1[:1] or head_entity in time_point_triples_tail_1[2:]:
                                count_tail_entity_1 += 1

                if count_tail_entity_1 != 0:
                    count_tail_entity_1_1 += 1
                    break

                if tail_entity in triple_tail[2:] and triple_tail[0] != 12554:
                    time_points_tail_2 = correct_result_dict[str(triple_tail[0])]
                    for time_points_triples_tail_2 in time_points_tail_2:
                        if int(time_points_triples_tail_2[3]) < time:
                            if head_entity in time_points_triples_tail_2[:1] or head_entity in time_points_triples_tail_2[2:]:
                                count_tail_entity_2 += 1
                if count_tail_entity_2 != 0:
                    count_tail_entity_2_1 += 1
                    break
            if count_tail_entity_1_1 != 0 or count_tail_entity_2_1 != 0:
                break

    if count_tail_entity != 0 or count_tail_entity_1_1 != 0 or count_tail_entity_2_1 != 0:
        counts_tail_entity += 1


    count_head_relation = 0
    count_head_relation_1_1 = 0
    count_tail_relation = 0
    count_tail_relation_1_1 = 0

    for time_point_triples_relation_head in time_points_head:
        if int(time_point_triples_relation_head[3]) < time:
            if relation == time_point_triples_relation_head[1]:
                count_head_relation += 1

            if count_head_relation != 0:
                break

            for triple in [time_point_triples_relation_head]:
                count_head_relation_1 = 0
                count_tail_relation_1 = 0
                if triple[0] != 12554 and triple[2] != 12554:
                    time_points_head_relation1 = correct_result_dict[str(triple[0])]
                    time_points_tail_relation1 = correct_result_dict[str(triple[2])]

                    for time_points_head_relation_triples1 in time_points_head_relation1:
                        if int(time_points_head_relation_triples1[3]) < time:
                            if relation == time_points_head_relation_triples1[1]:
                                count_head_relation_1 += 1

                        if count_head_relation_1 != 0:
                            break

                    for time_points_tail_relation_triples1 in time_points_tail_relation1:
                        if int(time_points_tail_relation_triples1[3]) < time:
                            if relation == time_points_tail_relation_triples1[1]:
                                count_tail_relation_1 += 1

                        if count_tail_relation_1 != 0:
                            break

                if count_head_relation_1 != 0 or count_tail_relation_1 != 0:
                    count_head_relation_1_1 += 1
                    break
            if count_head_relation_1_1 != 0:
                break

    if count_head_relation != 0 or count_head_relation_1_1 != 0:
        counts_relation += 1

    else:

        for time_point_triples_relation_tail in time_points_tail:
            if int(time_point_triples_relation_tail[3]) < time:
                if relation == time_point_triples_relation_tail[1]:
                    count_tail_relation += 1

                if count_tail_relation != 0:
                    break

                for triple in [time_point_triples_relation_tail]:
                    count_head_relation_1 = 0
                    count_tail_relation_1 = 0
                    if triple[0] != 12554 and triple[2] != 12554:
                        time_points_head_relation2 = correct_result_dict[str(triple[0])]
                        time_points_tail_relation2 = correct_result_dict[str(triple[2])]

                        for time_points_head_relation_triples2 in time_points_head_relation2:
                            if int(time_points_head_relation_triples2[3]) < time:
                                if relation == time_points_head_relation_triples2[1]:
                                    count_head_relation_1 += 1

                            if count_head_relation_1 != 0:
                                break

                        for time_points_tail_relation_triples2 in time_points_tail_relation2:
                            if int(time_points_tail_relation_triples2[3]) < time:
                                if relation == time_points_tail_relation_triples2[1]:
                                    count_tail_relation_1 += 1

                            if count_tail_relation_1 != 0:
                                break

                    if count_head_relation_1 != 0 or count_tail_relation_1 != 0:
                        count_tail_relation_1_1 += 1
                        break

                if count_tail_relation_1_1 != 0:
                    break

        if count_tail_relation != 0 or count_tail_relation_1_1 != 0:
            counts_relation += 1

print("分别针对训练集中的每一条样本，已知当前时间戳下的头实体和关系，在头实体过去的知识序列中，尾实体会出现的样本数为：" + str(counts_head_entity))
print("分别针对训练集中的每一条样本，已知当前时间戳下的尾实体和关系，在尾实体过去的知识序列中，头实体会出现的样本数为：" + str(counts_tail_entity))
print("分别针对训练集中的每一条样本，已知当前时间戳下的头实体和尾实体，在头实体和尾实体过去的知识序列中，关系会出现的样本数为：" + str(counts_relation))













