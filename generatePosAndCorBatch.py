import torch
import copy
from torch.utils.data import Dataset

class dataset(Dataset):

    def __init__(self, numOfTriple):
        self.tripleList = torch.LongTensor(range(numOfTriple))
        self.numOfTriple = numOfTriple

    def __len__(self):
        return self.numOfTriple

    def __getitem__(self, item):
        return self.tripleList[item]

class generateBatches:

    def __init__(self, batch, train2id, positiveBatch, corruptedBatch, numOfEntity, headRelation2Tail, tailRelation2Head):
        self.batch = batch
        self.train2id = train2id
        self.positiveBatch = positiveBatch
        self.corruptedBatch = corruptedBatch
        self.numOfEntity = numOfEntity
        self.headRelation2Tail = headRelation2Tail
        self.tailRelation2Head = tailRelation2Head

        self.generatePosAndCorBatch()

    def generatePosAndCorBatch(self):
        self.positiveBatch["h"] = []
        self.positiveBatch["r"] = []
        self.positiveBatch["t"] = []
        self.positiveBatch["T"] = []

        self.corruptedBatch["h"] = []
        self.corruptedBatch["r"] = []
        self.corruptedBatch["t"] = []
        self.corruptedBatch["T"] = []
        for tripleId in self.batch:
            tmpHead = self.train2id["h"][tripleId]
            tmpRelation = self.train2id["r"][tripleId]
            tmpTail = self.train2id["t"][tripleId]
            tmpTime = self.train2id["T"][tripleId]
            self.positiveBatch["h"].append(tmpHead)
            self.positiveBatch["r"].append(tmpRelation)
            self.positiveBatch["t"].append(tmpTail)
            self.positiveBatch["T"].append(tmpTime)
            if torch.rand(1).item() >= 0.5:
                tmpCorruptedHead = torch.FloatTensor(1).uniform_(0, self.numOfEntity).long().item()
                while tmpCorruptedHead in self.tailRelation2Head[tmpTail][tmpRelation] or tmpCorruptedHead == tmpHead:
                    tmpCorruptedHead = torch.FloatTensor(1).uniform_(0, self.numOfEntity).long().item()
                tmpHead = tmpCorruptedHead
            else:
                tmpCorruptedTail = torch.FloatTensor(1).uniform_(0, self.numOfEntity).long().item()
                while tmpCorruptedTail in self.headRelation2Tail[tmpHead][tmpRelation] or tmpCorruptedTail == tmpTail:
                    tmpCorruptedTail = torch.FloatTensor(1).uniform_(0, self.numOfEntity).long().item()
                tmpTail = tmpCorruptedTail
            self.corruptedBatch["h"].append(tmpHead)
            self.corruptedBatch["r"].append(tmpRelation)
            self.corruptedBatch["t"].append(tmpTail)
            self.corruptedBatch["T"].append(tmpTime)
        for aKey in self.positiveBatch:
            self.positiveBatch[aKey] = torch.LongTensor(self.positiveBatch[aKey])
        for aKey in self.corruptedBatch:
            self.corruptedBatch[aKey] = torch.LongTensor(self.corruptedBatch[aKey])

class generate_knowledge_sequence:
    def __init__(self, Training_specific_knowledge_sequence_1, BatchEntity, BatchTime, numOfEntity_1, numOfRelation_1):

        self.Training_specific_knowledge_sequence = Training_specific_knowledge_sequence_1
        self.BatchEntity = BatchEntity
        self.BatchTime = BatchTime
        self.numOfEntity_2 = numOfEntity_1
        self.numOfRelation_2 = numOfRelation_1
        self.BatchEntity_Training_specific_knowledge_sequence = self.generateBatchEntity_knowledge_sequence()


    def generateBatchEntity_knowledge_sequence(self):
        BatchEntity_Training_specific_knowledge_sequence_list = []
        for Entity, Time in zip(self.BatchEntity, self.BatchTime):

            Entity = Entity.item()
            Time = Time.item()
            Entity_Training_specific_knowledge_sequence_1 = self.Training_specific_knowledge_sequence[str(Entity)]
            Entity_Training_specific_knowledge_sequence = copy.deepcopy(Entity_Training_specific_knowledge_sequence_1)
            i = 0
            for knowledge in Entity_Training_specific_knowledge_sequence:

                if knowledge[3] >= Time:
                    Entity_Training_specific_knowledge_sequence[i][0] = self.numOfEntity_2
                    Entity_Training_specific_knowledge_sequence[i][1] = self.numOfRelation_2
                    Entity_Training_specific_knowledge_sequence[i][2] = self.numOfEntity_2

                i = i + 1

            BatchEntity_Training_specific_knowledge_sequence_list.append(Entity_Training_specific_knowledge_sequence)

        BatchEntity_Training_specific_knowledge_sequence = torch.tensor(BatchEntity_Training_specific_knowledge_sequence_list)

        return BatchEntity_Training_specific_knowledge_sequence


