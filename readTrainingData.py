import re
import json
import torch

class readData:                                      #需要传入10个参数
    def __init__(self, inAdd, train2id, headRelation2Tail, tailRelation2Head, entity2id, id2entity, relation2id, id2relation, nums):
        self.inAdd = inAdd                           #数据（知识图谱）目录

        self.train2id = train2id                     #空字典train2id = {}

        self.headRelation2Tail = headRelation2Tail   #空字典headRelation2Tail = {}
        self.tailRelation2Head = tailRelation2Head   #空字典tailRelation2Head = {}

        self.nums = nums                             #nums = [0, 0, 0]

        self.entity2id = entity2id                   #空字典entity2id = {}
        self.id2entity = id2entity                   #id2entity = {}

        self.relation2id = relation2id               #空字典relation2id = {}
        self.id2relation = id2relation               #空字典id2relation = {}

        self.Training_specific_knowledge_sequence = None

        self.numOfTriple = 0
        self.numOfEntity = 0
        self.numOfRelation = 0

        self.trainTriple = None


        self.readTrain2id()
        print("number of triples: " + str(self.numOfTriple))

        #self.readEntity2id()
        #print("number of entities: " + str(self.numOfEntity))

        #self.readRelation2id()
        #print("number of relations: " + str(self.numOfRelation))

        self.read_knowledge_sequence()
        print("number of Training_specific_knowledge_sequence: " + str(len(self.Training_specific_knowledge_sequence)))

        self.nums[0] = self.numOfTriple
        self.nums[1] = self.numOfEntity
        self.nums[2] = self.numOfRelation

    def out(self):
        return self.Training_specific_knowledge_sequence, self.trainTriple

    def readTrain2id(self):
        print("-----Reading train.txt from " + self.inAdd + "/-----")
        count = 0
        inputData = open(self.inAdd + "/train.txt")
        line = inputData.readline()
        self.numOfTriple = int(re.findall(r"\d+", line)[0])
        self.train2id["h"] = []
        self.train2id["t"] = []
        self.train2id["r"] = []
        self.train2id["T"] = []
        self.trainTriple = torch.ones(self.numOfTriple, 4)
        line = inputData.readline()
        while line and line not in ["\n", "\r\n", "\r"]:
            reR = re.findall(r"\d+", line)
            if reR:
                tmpHead = int(re.findall(r"\d+", line)[0])
                tmpRelation = int(re.findall(r"\d+", line)[1])
                tmpTail = int(re.findall(r"\d+", line)[2])
                tmpTime = int(re.findall(r"\d+", line)[3])

                self.train2id["h"].append(tmpHead)
                self.train2id["r"].append(tmpRelation)
                self.train2id["t"].append(tmpTail)
                self.train2id["T"].append(tmpTime)

                self.trainTriple[count, 0] = tmpHead
                self.trainTriple[count, 1] = tmpRelation
                self.trainTriple[count, 2] = tmpTail
                self.trainTriple[count, 3] = tmpTime

                if tmpHead not in self.headRelation2Tail.keys():
                    self.headRelation2Tail[tmpHead] = {}
                    self.headRelation2Tail[tmpHead][tmpRelation] = {}
                    self.headRelation2Tail[tmpHead][tmpRelation][tmpTail] = []
                    self.headRelation2Tail[tmpHead][tmpRelation][tmpTail].append(tmpTime)
                else:
                    if tmpRelation not in self.headRelation2Tail[tmpHead].keys():
                        self.headRelation2Tail[tmpHead][tmpRelation] = {}
                        self.headRelation2Tail[tmpHead][tmpRelation][tmpTail] = []
                        self.headRelation2Tail[tmpHead][tmpRelation][tmpTail].append(tmpTime)
                    else:
                        if tmpTail not in self.headRelation2Tail[tmpHead][tmpRelation].keys():
                            self.headRelation2Tail[tmpHead][tmpRelation][tmpTail] = []
                            self.headRelation2Tail[tmpHead][tmpRelation][tmpTail].append(tmpTime)
                        else:
                            if tmpTime not in self.headRelation2Tail[tmpHead][tmpRelation][tmpTail]:
                                self.headRelation2Tail[tmpHead][tmpRelation][tmpTail].append(tmpTime)



                if tmpTail not in self.tailRelation2Head.keys():
                    self.tailRelation2Head[tmpTail] = {}
                    self.tailRelation2Head[tmpTail][tmpRelation] = {}
                    self.tailRelation2Head[tmpTail][tmpRelation][tmpHead] = []
                    self.tailRelation2Head[tmpTail][tmpRelation][tmpHead].append(tmpTime)
                else:
                    if tmpRelation not in self.tailRelation2Head[tmpTail].keys():
                        self.tailRelation2Head[tmpTail][tmpRelation] = {}
                        self.tailRelation2Head[tmpTail][tmpRelation][tmpHead] = []
                        self.tailRelation2Head[tmpTail][tmpRelation][tmpHead].append(tmpTime)
                    else:
                        if tmpHead not in self.tailRelation2Head[tmpTail][tmpRelation].keys():
                            self.tailRelation2Head[tmpTail][tmpRelation][tmpHead] = []
                            self.tailRelation2Head[tmpTail][tmpRelation][tmpHead].append(tmpTime)
                        else:
                            if tmpTime not in self.tailRelation2Head[tmpTail][tmpRelation][tmpHead]:
                                self.tailRelation2Head[tmpTail][tmpRelation][tmpHead].append(tmpTime)



                count += 1
                line = inputData.readline()
            else:
                print("error in train.txt at Line " + str(count + 2))
                line = inputData.readline()
        inputData.close()
        if count == self.numOfTriple:
            self.trainTriple.long()
            return
        else:
            print("error in train.txt")
            return

    def readEntity2id(self):
        print("-----Reading entity2id.txt from " + self.inAdd + "/-----")
        count = 0
        inputData = open(self.inAdd + "/entity2id.txt", encoding='utf-8')
        line = inputData.readline()
        self.numOfEntity = int(re.findall(r"\d+", line)[0])
        line = inputData.readline()
        while line and line not in ["\n", "\r\n", "\r"]:
            #reR = re.search(r"^(.+?)\t(\d+)\t", line) # YAGO用这行
            reR = re.search(r"^(.+?)\t(\d+)$", line) #WIKI和ICEWS8用这行
            if reR:
                entity = reR.group(1)
                Eid = reR.group(2)
                self.entity2id[entity] = int(Eid)
                self.id2entity[int(Eid)] = entity
                count += 1
                line = inputData.readline()
            else:
                print("error in entity2id.txt at line " + str(count + 2))
                line = inputData.readline()
        inputData.close()
        if count == self.numOfEntity:
            return
        else:
            print("error in entity2id.txt")
            return

    def readRelation2id(self):
        print("-----Reading relation2id.txt from " + self.inAdd + "/-----")
        count = 0
        inputData = open(self.inAdd + "/relation2id.txt", encoding='utf-8')
        line = inputData.readline()
        self.numOfRelation = int(re.findall(r"\d+", line)[0])
        line = inputData.readline()
        while line and line not in ["\n", "\r\n", "\r"]:
            reR = re.search(r"(.+)\t(\d+)", line)
            if reR:
                relation = reR.group(1)
                Rid = int(reR.group(2))
                self.relation2id[relation] = Rid
                self.id2relation[Rid] = relation
                line = inputData.readline()
                count += 1
            else:
                print("error in relation2id.txt at line " + str(count + 2))
                line = inputData.readline()
        inputData.close()
        if count == self.numOfRelation:
            return
        else:
            print("error in relation2id.txt")
            return

    def read_knowledge_sequence(self):
        print("-----Reading Training_specific_knowledge_sequence.json from " + self.inAdd + "/-----")
        with open(self.inAdd + "/Training_specific_knowledge_sequence(1-hop)(random).json", 'r') as file:
            # 从文件中加载字典
            self.Training_specific_knowledge_sequence = json.load(file)
