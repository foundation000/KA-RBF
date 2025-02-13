import re
import copy
import time
import pickle
import json
import torch
from 模型 import Encoder


def read_knowledge_sequence(inAdd):
    print("-----Reading Training_specific_knowledge_sequence.json from " + inAdd + "/-----")
    with open(inAdd + "/Training_specific_knowledge_sequence(1-hop)(random).json", 'r') as file:
        # 从文件中加载字典
        Training_specific_knowledge_sequence = json.load(file)
        return Training_specific_knowledge_sequence

def readTestTriples(inAdd):
    test2id = {}
    fileName = "/test.txt"
    print("-----Reading Test Triples from " + inAdd + fileName + "-----")
    count = 0
    test2id["h"] = []
    test2id["r"] = []
    test2id["t"] = []
    test2id["T"] = []
    inputData = open(inAdd + fileName)
    line = inputData.readline()
    numOfTestTriple = int(re.findall(r"\d+", line)[0])
    line = inputData.readline()
    while line and line not in ["\n", "\r\n", "\r"]:
        reR = re.findall(r"\d+", line)
        if reR:
            tmpHead = int(re.findall(r"\d+", line)[0])
            tmpTail = int(re.findall(r"\d+", line)[2])
            tmpRelation = int(re.findall(r"\d+", line)[1])
            tmpTime = int(re.findall(r"\d+", line)[3])
            test2id["h"].append(tmpHead)
            test2id["r"].append(tmpRelation)
            test2id["t"].append(tmpTail)
            test2id["T"].append(tmpTime)
            count += 1
        else:
            print("error in " + fileName + " at Line " + str(count + 2))
        line = inputData.readline()
    inputData.close()

    if count == numOfTestTriple:
        print("number of test triples: " + str(numOfTestTriple))

    else:
        print("count: " + str(count))
        print("expected number of test triples:" + str(numOfTestTriple))
        print("error in " + fileName)

    return test2id, numOfTestTriple

def generateBatchEntity_knowledge_sequence(BatchEntity, BatchTime, Training_specific_knowledge_sequence, numOfEntity, numOfRelation):
    BatchEntity_Training_specific_knowledge_sequence_list = []

    for Entity, Time in zip(BatchEntity, BatchTime):
        Entity_Training_specific_knowledge_sequence_copy = Training_specific_knowledge_sequence[str(Entity)]
        Entity_Training_specific_knowledge_sequence = copy.deepcopy(Entity_Training_specific_knowledge_sequence_copy)
        i = 0
        for knowledge in Entity_Training_specific_knowledge_sequence:
            if knowledge[3] >= Time:
                Entity_Training_specific_knowledge_sequence[i][0] = numOfEntity
                Entity_Training_specific_knowledge_sequence[i][1] = numOfRelation
                Entity_Training_specific_knowledge_sequence[i][2] = numOfEntity
            i = i + 1
        BatchEntity_Training_specific_knowledge_sequence_list.append(Entity_Training_specific_knowledge_sequence)

    return BatchEntity_Training_specific_knowledge_sequence_list

def loading_model(inAdd, numOfEntity, numOfRelation, numOfTimestamp):
    print("-----loading Model（KF-RBFA）-----")

    model = Encoder(numOfEntity=numOfEntity + 1, numOfRelation=numOfRelation + 1, numOfTimestamp=numOfTimestamp,
                    numOfTime_domain=numOfTimestamp // 10 + 1, Dimension=40,
                    num_layers=4, heads=2, dropout=0,
                    forward_expansion=2, gamma=0.2, lambda_reg=0.1)

    model.load_state_dict(torch.load(inAdd + "/KF-RBFA.pkl", map_location=torch.device('cpu')))

    print("-----loading Complete-----")
    return model

def Model_Sorting(model, Heads_test_specific_knowledge_sequence, Tails_test_specific_knowledge_sequence, test_Heads,
                  test_Relations, test_Tails, numOfEntity, numOfRelation):
    print("-----Sorting Started at " + time.strftime('%m-%d-%Y %H:%M:%S', time.localtime(time.time())) + "-----")

    attn_weight_heads = []
    attn_weight_tails = []

    filtered_Heads_test_knowledge_sequence = []
    filtered_Tails_test_knowledge_sequence = []

    for Head_test_specific_knowledge_sequence, Tail_test_specific_knowledge_sequence, test_Relation, test_Head, test_Tail in zip(torch.tensor(Heads_test_specific_knowledge_sequence), torch.tensor(Tails_test_specific_knowledge_sequence), torch.tensor(test_Relations),torch.tensor(test_Heads), torch.tensor(test_Tails)):

        attn_weights1, attn_weights2, loss = model(Head_test_specific_knowledge_sequence.unsqueeze(0), Tail_test_specific_knowledge_sequence.unsqueeze(0), test_Relation.unsqueeze(0), test_Head.unsqueeze(0), test_Tail.unsqueeze(0), None)

        attn_weight_heads.append(attn_weights1.tolist()[0][0])
        attn_weight_tails.append(attn_weights2.tolist()[0][0])

    mum = 0

    for Head_test_specific_knowledge_sequence_list, Tail_test_specific_knowledge_sequence_list, test_head, test_tail, attn_weight_head, attn_weight_tail in zip(
        Heads_test_specific_knowledge_sequence, Tails_test_specific_knowledge_sequence, test_Heads, test_Tails, attn_weight_heads, attn_weight_tails):


        # 使用列表推导式同时过滤两个列表
        A = [(a, b) for a, b in zip(Head_test_specific_knowledge_sequence_list, attn_weight_head) if numOfEntity not in a]
        if A:
            filtered_Head_specific_knowledge_sequence, filtered_attn_weight_head = zip(*A)
            filtered_Head_specific_knowledge_sequence = list(filtered_Head_specific_knowledge_sequence)
            filtered_attn_weight_head = list(filtered_attn_weight_head)
        else:
            filtered_Head_specific_knowledge_sequence = [[numOfEntity, numOfRelation, numOfEntity]]
            filtered_attn_weight_head = [1]

        B = [(a, b) for a, b in zip(Tail_test_specific_knowledge_sequence_list, attn_weight_tail) if numOfEntity not in a]
        if B:
            filtered_Tail_specific_knowledge_sequence, filtered_attn_weight_tail = zip(*B)
            filtered_Tail_specific_knowledge_sequence = list(filtered_Tail_specific_knowledge_sequence)
            filtered_attn_weight_tail = list(filtered_attn_weight_tail)
        else:
            filtered_Tail_specific_knowledge_sequence = [[numOfEntity, numOfRelation, numOfEntity]]
            filtered_attn_weight_tail = [1]

        # 使用 zip 结合两个列表并根据 attn_weight 的元素排序
        combined_Head = list(zip(filtered_attn_weight_head, filtered_Head_specific_knowledge_sequence))
        combined_Head.sort(reverse=True)  # 指定降序排序
        combined_Tail = list(zip(filtered_attn_weight_tail, filtered_Tail_specific_knowledge_sequence))
        combined_Tail.sort(reverse=True)  # 指定降序排序

        # 解压排序后的列表
        filtered_attn_weight_sorted_head, filtered_head_specific_knowledge_sequence_sorted = zip(*combined_Head)
        filtered_attn_weight_sorted_tail, filtered_tail_specific_knowledge_sequence_sorted = zip(*combined_Tail)

        # 将结果转换为列表形式
        filtered_head_specific_knowledge_sequence_sorted_list = list(filtered_head_specific_knowledge_sequence_sorted)
        filtered_tail_specific_knowledge_sequence_sorted_list = list(filtered_tail_specific_knowledge_sequence_sorted)

        # 汇总
        filtered_Heads_test_knowledge_sequence.append(filtered_head_specific_knowledge_sequence_sorted_list)
        filtered_Tails_test_knowledge_sequence.append(filtered_tail_specific_knowledge_sequence_sorted_list)

        mum = mum + 1

        if mum % 5000 == 0:
            print(str(mum) + " test triples sorting processed!")

    print("-----Sorting Ended at " + time.strftime('%m-%d-%Y %H:%M:%S', time.localtime(time.time())) + "-----")
    return filtered_Heads_test_knowledge_sequence, filtered_Tails_test_knowledge_sequence

def readid2entity(inAdd):
    print("-----Reading entity2id.txt from " + inAdd + "/-----")
    count = 0
    id2entity = {}
    inputData = open(inAdd + "/entity2id.txt", encoding='utf-8')
    line = inputData.readline()
    numOfEntity = int(re.findall(r"\d+", line)[0])
    line = inputData.readline()
    while line and line not in ["\n", "\r\n", "\r"]:
        #reR = re.search(r"^(.+?)\t(\d+)\t", line)     #YAGO用这行
        reR = re.search(r"^(.+?)\t(\d+)$", line)             #WIKI用这行
        if reR:
            entity = reR.group(1)
            Eid = reR.group(2)
            id2entity[int(Eid)] = entity
            count += 1
            line = inputData.readline()
        else:
            print("error in entity2id.txt at line " + str(count + 2))
            line = inputData.readline()
    inputData.close()
    if count == numOfEntity:
        return id2entity
    else:
        print("error in entity2id.txt")
        return

def readid2relation(inAdd):
    print("-----Reading relation2id.txt from " + inAdd + "/-----")
    count = 0
    id2relation = {}
    inputData = open(inAdd + "/relation2id.txt", encoding='utf-8')
    line = inputData.readline()
    numOfRelation = int(re.findall(r"\d+", line)[0])
    line = inputData.readline()
    while line and line not in ["\n", "\r\n", "\r"]:
        reR = re.search(r"(.+)\t(\d+)", line)
        if reR:
            relation = reR.group(1)
            Rid = int(reR.group(2))

            id2relation[Rid] = relation
            line = inputData.readline()
            count += 1
        else:
            print("error in relation2id.txt at line " + str(count + 2))
            line = inputData.readline()
    inputData.close()
    if count == numOfRelation:
        return id2relation
    else:
        print("error in relation2id.txt")
        return

def mapping(Id2entity, Id2relation, Heads_test_knowledge_sequence, Tails_test_knowledge_sequence, numOfEntity, numOfRelation):
    print("-----Start Mapping-----")
    new_Heads_test_knowledge_sequence = []
    new_Tails_test_knowledge_sequence = []

    for Head_test_knowledge_sequence in Heads_test_knowledge_sequence:
        new_Head_test_knowledge_sequence = []
        if Head_test_knowledge_sequence != [[numOfEntity, numOfRelation, numOfEntity]]:
            for knowledge in Head_test_knowledge_sequence:
                new_knowledge = []
                new_knowledge.append(Id2entity[knowledge[0]])
                new_knowledge.append(Id2relation[knowledge[1]])
                new_knowledge.append(Id2entity[knowledge[2]])
                new_knowledge.append(knowledge[3])
                new_Head_test_knowledge_sequence.append(new_knowledge)
            new_Head_test_knowledge_sequence_sorted = sorted(new_Head_test_knowledge_sequence, key=lambda x: x[-1])
            new_Heads_test_knowledge_sequence.append(new_Head_test_knowledge_sequence_sorted)
        else:
            new_Heads_test_knowledge_sequence.append([[]])

    for Tail_test_knowledge_sequence in Tails_test_knowledge_sequence:
        new_Tail_test_knowledge_sequence = []
        if Tail_test_knowledge_sequence != [[numOfEntity, numOfRelation, numOfEntity]]:
            for knowledge in Tail_test_knowledge_sequence:
                new_knowledge = []
                new_knowledge.append(Id2entity[knowledge[0]])
                new_knowledge.append(Id2relation[knowledge[1]])
                new_knowledge.append(Id2entity[knowledge[2]])
                new_knowledge.append(knowledge[3])
                new_Tail_test_knowledge_sequence.append(new_knowledge)
            new_Tail_test_knowledge_sequence_sorted = sorted(new_Tail_test_knowledge_sequence, key=lambda x: x[-1])
            new_Tails_test_knowledge_sequence.append(new_Tail_test_knowledge_sequence_sorted)
        else:
            new_Tails_test_knowledge_sequence.append([[]])

    print("-----Mapping Complete-----")
    return new_Heads_test_knowledge_sequence, new_Tails_test_knowledge_sequence

def LLM(b):
    a = b

if __name__ == '__main__':
        inAdd = "./data/ICEWS18"
        numOfEntity = 23033
        numOfRelation = 256
        numOfTimestamp = 240

        # （1）读取知识序列文件
        Training_specific_knowledge_sequence = read_knowledge_sequence(inAdd)

        # （2）读取测试集
        test2id, numOfTestTriple = readTestTriples(inAdd)

        # （3）读取测试集中头尾实体所对应的知识序列
        print("-----Generate Test Heads knowledge_sequence-----")
        Heads_test_specific_knowledge_sequence = generateBatchEntity_knowledge_sequence(test2id["h"], test2id["T"],
        Training_specific_knowledge_sequence,
        numOfEntity, numOfRelation)
        print("-----Generate Complete-----")
        print("-----Generate Test Tails knowledge_sequence-----")
        Tails_test_specific_knowledge_sequence = generateBatchEntity_knowledge_sequence(test2id["t"], test2id["T"],
        Training_specific_knowledge_sequence,
        numOfEntity, numOfRelation)
        print("-----Generate Complete-----")

        # （4）加载训练好的KF-RBFA模型
        KF_RBFA = loading_model(inAdd, numOfEntity, numOfRelation, numOfTimestamp)

        # （5）根据训练好的KF-RBFA模型对（4）中的知识序列进行清洗与排序
        Filtered_Heads_test_knowledge_sequence, Filtered_Tails_test_knowledge_sequence = Model_Sorting(KF_RBFA, Heads_test_specific_knowledge_sequence, Tails_test_specific_knowledge_sequence, test2id["h"],
                      test2id["r"], test2id["t"], numOfEntity, numOfRelation)

        # （6）对排序后的结果进行截断（前20个）
        Filtered_Heads_test_knowledge_sequence = [sublist[:20] for sublist in Filtered_Heads_test_knowledge_sequence]
        Filtered_Tails_test_knowledge_sequence = [sublist[:20] for sublist in Filtered_Tails_test_knowledge_sequence]

        # （7）读取实体和关系的索引映射dict
        Id2entity = readid2entity(inAdd)
        Id2relation = readid2relation(inAdd)

        # （8）将截断后的结果映射为具有现实意义的实体和关系
        New_Heads_test_knowledge_sequence, New_Tails_test_knowledge_sequence = mapping(Id2entity, Id2relation, Filtered_Heads_test_knowledge_sequence, Filtered_Tails_test_knowledge_sequence, numOfEntity, numOfRelation)

        # （9）将现实意义的知识序列存储在本地
        print("-----Starting Saving-----")
        with open(inAdd + '/New_Heads_test_knowledge_sequence.json', 'w') as file:
            json.dump(New_Heads_test_knowledge_sequence, file)

        with open(inAdd + '/New_Tails_test_knowledge_sequence.json', 'w') as file:
            json.dump(New_Tails_test_knowledge_sequence, file)
        print("-----Saving Complete-----")

        # （9）将(8)的结果填充到预定义的Prompt中，并传入给大模型，返回大模型的生成结果并进行存储
