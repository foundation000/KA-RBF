import re
import os
import time
import json
import torch
import pickle
from 模型 import Encoder
import torch.optim as optim
import torch.distributed as dist
from readTrainingData import readData
from torch.utils.data import DataLoader
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler
from generatePosAndCorBatch import generateBatches, dataset, generate_knowledge_sequence

class trainModel:
    def __init__(self):

        #数据加载参数
        self.inAdd = "./data/ICEWS14"                 #输入数据（知识图谱）目录
        self.outAdd = "./data/ICEWS14"                #训练结束后模型的输出目录
        self.lossAdd = "./data/ICEWS14"               #训练loss的输出目录
        self.preAdd = "./data/ICEWS14/outputData"     #现有的预训练地址
        self.preOrNot = False                      #是否基于self.preAdd中的现有嵌入继续培训

        #模型参数
        self.Dimension = 40                         #实体，关系，时间戳以及时间域的维度
        self.numOfTriple = 0
        self.numOfEntity = 0
        self.numOfRelation = 0
        self.numOfTime_domain = 10                  #时间域的个数
        self.num_layers = 4                         #堆叠多少层TransformerBlock
        self.heads = 2                              #TransformerBlock中注意力的头数
        self.dropout = 0                            #dropout的比率
        self.forward_expansion = 2                  #TransformerBlock中,在FFN中第一个全连接上升特征数的倍数
        self.mask = None
        self.gamma = 0.2                            #高斯函数的宽度, 过大的gamma会使核函数变得非常尖锐，只有非常近的样本对会有较高的相似度；而过小的gamma会使得核函数变得平坦，几乎所有的样本对都具有相似的高相似度。
        self.lambda_reg = 0.1                       #正则化项的权重超参数

        # 训练参数
        self.numOfEpochs = 200                      #Epoch的次数
        self.outputFreq  = 5                        #每5个Epoch输出下验证集的学习结果
        self.numOfBatches = 1500                    #Batch有数量
        self.learningRate = 0.01  # 0.01            #SGD优化器的学习速率
        self.weight_decay = 0.005  # 0.005  0.02    #SGD优化器的权重衰减率

        self.patience = 5                           #在self.patience 5次之后验证结果没有改善时，更改学习率和权重衰减
        self.earlyStopPatience = 5                  #改变学习率和权重衰减self.earlyStopPatience次数后，停止训练并输出学习结果
        self.bestAvFiMR = None

        #if torch.cuda.is_available():              #看是否有可用GPU、
            #self.device = torch.device("cuda:0")
        #else:
            #self.device = torch.device("cpu")

        #数据加载产生的中间参数，包括生成负样本所需的参数
        self.train2id = {}
        self.trainTriple = None

        self.entity2id = {}
        self.id2entity = {}

        self.relation2id = {}
        self.id2relation = {}

        self.Training_specific_knowledge_sequence = None

        self.nums = [0, 0, 0]                       #用来保存（三元组数目，实体数目，关系数目）

        self.headRelation2Tail = {}
        self.tailRelation2Head = {}

        self.positiveBatch = {}
        self.corruptedBatch = {}

        self.entityEmbedding = None
        self.relationEmbedding = None

        # 读验证集需要的参数
        self.validate2id = {}
        self.numOfValidateTriple = 0                

        # 读测试集需要的参数
        self.test2id = {}
        self.numOfTestTriple = 0

        # 分布式学习参数
        #print(torch.__version__)
        #print(torch.version.cuda)
        #print(torch.cuda.is_available())
        self.rank = int(os.environ["RANK"])
        self.local_rank = int(os.environ["LOCAL_RANK"])
        torch.cuda.set_device(self.rank % torch.cuda.device_count())
        dist.init_process_group(backend="nccl")
        self.device = torch.device("cuda", self.local_rank)

        self.start()
        self.train()
        self.end()

    def start(self):
        print("-----Training Started at " + time.strftime('%m-%d-%Y %H:%M:%S',time.localtime(time.time())) + "-----")
        print("input address: " + self.inAdd)
        print("output address: " +self.outAdd)
        print("entity dimension: " + str(self.Dimension))
        print("relation dimension: " + str(self.Dimension))
        print("number of epochs: " + str(self.numOfEpochs))
        print("output training results every " + str(self.outputFreq) + " epochs")
        print("number of batches: " + str(self.numOfBatches))
        print("learning rate: " + str(self.learningRate))
        print("weight decay: " + str(self.weight_decay))
        print("gamma: " + str(self.gamma))
        print("lambda_reg: " + str(self.lambda_reg))
        print("is a continued learning: " + str(self.preOrNot))
        if self.preOrNot:
            print("pre-trained result address: " + self.preAdd)
        print("device: " + str(self.device))
        print("patience: " + str(self.patience))
        print("early stop patience: " + str(self.earlyStopPatience))
        print(f"[init] == local rank: {self.local_rank}, global rank: {self.rank} ==")

    def end(self):
        print("-----Training Finished at " + time.strftime('%m-%d-%Y %H:%M:%S',time.localtime(time.time())) + "-----")

    def train(self):

        read = readData(self.inAdd, self.train2id, self.headRelation2Tail, self.tailRelation2Head,
                      self.entity2id, self.id2entity, self.relation2id, self.id2relation, self.nums)
        self.Training_specific_knowledge_sequence, self.trainTriple = read.out()   #这里返回的是一个tensor，Size为([训练集数量, 4])，其中维度4指的是（h，r，t, T）。
        self.numOfTriple = 323895 #self.nums[0]
        self.numOfEntity = 12499 #self.nums[1]
        self.numOfRelation = 261 #self.nums[2]

        # 调用dataset，输入参数是三元组的数量
        dataSet = dataset(self.numOfTriple)
        batchSize = int(self.numOfTriple / self.numOfBatches)
        sampler = DistributedSampler(dataSet, shuffle=True)    # 创建分布式采样器
        dataLoader = DataLoader(dataSet, batchSize, num_workers=8, pin_memory=True, sampler=sampler)
        #dataLoader = DataLoader(dataSet, batchSize, pin_memory=True)

        # 读验证集
        #self.readValidateTriples()
        # 读测试集
        self.readTestTriples()

        # 实例化模型
        model = Encoder(numOfEntity=self.numOfEntity + 1, numOfRelation=self.numOfRelation + 1, numOfTimestamp=len(set(self.train2id["T"])), numOfTime_domain=len(set(self.train2id["T"]))//self.numOfTime_domain + 1, Dimension=self.Dimension,
                        num_layers=self.num_layers, heads=self.heads, dropout=self.dropout, forward_expansion=self.forward_expansion, gamma=self.gamma, lambda_reg=self.lambda_reg)

        model.to(self.device)

        model = DDP(model, device_ids=[self.local_rank], output_device=self.local_rank) #包装模型进行并行处理

        #self.bestAvFiMR = self.Fast_validate(model)

        if self.preOrNot:       #判断是或否基于预训练，默认是不基于
            self.preRead(model)

        # 定义优化器
        optimizer = optim.SGD(model.parameters(), lr=self.learningRate, weight_decay=self.weight_decay)

        loss_history = []
        patienceCount = 0

        for epoch in range(self.numOfEpochs):
            # 更新采样器状态
            dataLoader.sampler.set_epoch(epoch)

            # 每一次Epoch时，损失置为0
            epochLoss = 0

            # batch是一个向量,长度是三元组总数除以numOfBatches，也就是batchSize。形式还都是tensor的形式，向量中每一个数字代表三元组的序号。
            for batch in dataLoader:

                self.positiveBatch = {}
                self.corruptedBatch = {}

                #根据batch，产生对应的正样本和负样本
                generateBatches(batch, self.train2id, self.positiveBatch, self.corruptedBatch, self.numOfEntity,
                                self.headRelation2Tail, self.tailRelation2Head)

                # 优化器梯度清0
                optimizer.zero_grad()

                # 只提取正样本，positiveBatchHead和batch的shape一样
                positiveBatchHead = self.positiveBatch["h"]
                positiveBatchRelation = self.positiveBatch["r"]
                positiveBatchTail = self.positiveBatch["t"]
                positiveBatchTime = self.positiveBatch["T"]

                # 生成positiveBatchHead，positiveBatchTail，所对应的过去知识序列
                positiveBatchHead_Training_specific_knowledge_sequence = generate_knowledge_sequence(self.Training_specific_knowledge_sequence, positiveBatchHead, positiveBatchTime, self.numOfEntity, self.numOfRelation).BatchEntity_Training_specific_knowledge_sequence
                positiveBatchTail_Training_specific_knowledge_sequence = generate_knowledge_sequence(self.Training_specific_knowledge_sequence, positiveBatchTail, positiveBatchTime, self.numOfEntity, self.numOfRelation).BatchEntity_Training_specific_knowledge_sequence
                #print(positiveBatchHead_Training_specific_knowledge_sequence)


                # 输入数据传入GPU中
                positiveBatchHead = positiveBatchHead.to(self.device)
                positiveBatchRelation = positiveBatchRelation.to(self.device)
                positiveBatchTail = positiveBatchTail.to(self.device)

                positiveBatchHead_Training_specific_knowledge_sequence = positiveBatchHead_Training_specific_knowledge_sequence.to(self.device)
                positiveBatchTail_Training_specific_knowledge_sequence = positiveBatchTail_Training_specific_knowledge_sequence.to(self.device)

                #model.Entity_embedding.weight.data[self.numOfEntity] = torch.zeros(self.Dimension)
                #model.Relation_embedding.weight.data[self.numOfRelation] = torch.zeros(self.Dimension)

                # 前向传播
                attn_weights1, attn_weights2, batchLoss = model(positiveBatchHead_Training_specific_knowledge_sequence, positiveBatchTail_Training_specific_knowledge_sequence,
                            positiveBatchRelation, positiveBatchHead, positiveBatchTail, self.mask)

                batchLoss.backward()       # 反向传播计算当前梯度
                optimizer.step()           # 更新权重
                epochLoss += batchLoss     # 全部batch的损失累加为一个epoch的损失

            if dist.get_rank() == 0:
                print("epoch " + str(epoch) + ": , loss: " + str(epochLoss))
                loss_history.append(epochLoss.item())
            
            '''
            if epoch % self.outputFreq == 0:

                tmpAvFiMR = self.Fast_validate(model)

                if tmpAvFiMR < self.bestAvFiMR:
                    print("best averaged raw mean rank: " + str(self.bestAvFiMR) + " -> " + str(tmpAvFiMR))
                    patienceCount = 0
                    self.bestAvFiMR = tmpAvFiMR
                else:
                    patienceCount += 1
                    print("early stop patience: " + str(self.earlyStopPatience) + ", patience count: " + str(patienceCount) + ", current rank: " + str(tmpAvFiMR) + ", best rank: " + str(self.bestAvFiMR))
                    if patienceCount == self.patience:
                        if self.earlyStopPatience == 1:
                            break
                        print("learning rate: " + str(self.learningRate) + " -> " + str(self.learningRate / 2))
                        print("weight decay: " + str(self.weight_decay) + " -> " + str(self.weight_decay * 2))
                        self.learningRate = self.learningRate/2
                        self.weight_decay = self.weight_decay*2
                        optimizer = optim.SGD(model.parameters(), lr=self.learningRate, weight_decay=self.weight_decay)
                        patienceCount = 0
                        self.earlyStopPatience -= 1

            '''
            
        
        torch.save(model.module.state_dict(), self.outAdd + '/KF-RBFA.pkl')
        print("模型保存成功")
        self.test(model)
        with open(self.lossAdd + '/loss_historyt.json', 'w') as file:
            # 将loss写入 JSON 文件
            json.dump(loss_history, file)
        print("loss保存成功")
        #plt.plot(loss_history)
        #plt.title('Loss over iterations')
        #plt.xlabel('Iteration')
        #plt.ylabel('Loss')
        #plt.grid(True)
        #plt.show()

    def readValidateTriples(self):
        fileName = "/valid.txt"
        print("-----Reading valid Triples from " + self.inAdd + fileName + "-----")
        count = 0
        self.validate2id["h"] = []
        self.validate2id["r"] = []
        self.validate2id["t"] = []
        self.validate2id["T"] = []
        inputData = open(self.inAdd + fileName)
        line = inputData.readline()
        self.numOfValidateTriple = int(re.findall(r"\d+", line)[0])
        line = inputData.readline()
        while line and line not in ["\n", "\r\n", "\r"]:
            reR = re.findall(r"\d+", line)
            if reR:
                tmpHead = int(re.findall(r"\d+", line)[0])
                tmpTail = int(re.findall(r"\d+", line)[2])
                tmpRelation = int(re.findall(r"\d+", line)[1])
                tmpTime = int(re.findall(r"\d+", line)[3])
                self.validate2id["h"].append(tmpHead)
                self.validate2id["r"].append(tmpRelation)
                self.validate2id["t"].append(tmpTail)
                self.validate2id["T"].append(tmpTime)
                count += 1
            else:
                print("error in " + fileName + " at Line " + str(count + 2))
            line = inputData.readline()
        inputData.close()

        if count == self.numOfValidateTriple:
            print("number of validate triples: " + str(self.numOfValidateTriple))

        else:
            print("count: " + str(count))
            print("expected number of validate triples:" + str(self.numOfValidateTriple))
            print("error in " + fileName)

    def Fast_validate(self, model):

        print("Fast_Valid Started")

        attn_weight_heads = []
        attn_weight_tails = []

        numOfValidateTriple_first_Head = 0

        MR_tail = []
        
        #validate_Heads = torch.tensor(self.validate2id["h"][:10000])
        validate_Heads = torch.tensor(self.validate2id["h"])
        validate_Tails = torch.tensor(self.validate2id["t"])
        validate_Relations = torch.tensor(self.validate2id["r"])
        validate_Times = torch.tensor(self.validate2id["T"])

        Heads_validate_specific_knowledge_sequence = generate_knowledge_sequence(self.Training_specific_knowledge_sequence,
                                                                             validate_Heads, validate_Times,
                                                                             self.numOfEntity,
                                                                             self.numOfRelation).BatchEntity_Training_specific_knowledge_sequence
        Tails_validate_specific_knowledge_sequence = generate_knowledge_sequence(self.Training_specific_knowledge_sequence,
                                                                             validate_Tails, validate_Times,
                                                                             self.numOfEntity,
                                                                             self.numOfRelation).BatchEntity_Training_specific_knowledge_sequence

        Heads_validate_specific_knowledge_sequence_list = Heads_validate_specific_knowledge_sequence.tolist()

        for Head_validate_specific_knowledge_sequence, Tail_validate_specific_knowledge_sequence, validate_Relation, validate_Head, validate_Tail in zip(Heads_validate_specific_knowledge_sequence, Tails_validate_specific_knowledge_sequence, validate_Relations, validate_Heads, validate_Tails):

            attn_weights1, attn_weights2, loss = model(Head_validate_specific_knowledge_sequence.unsqueeze(0).to(self.device),
                                                       Tail_validate_specific_knowledge_sequence.unsqueeze(0).to(self.device), validate_Relation.unsqueeze(0).to(self.device),
                                                       validate_Head.unsqueeze(0).to(self.device), validate_Tail.unsqueeze(0).to(self.device), None)

            attn_weight_heads.append(attn_weights1.tolist()[0][0])
            attn_weight_tails.append(attn_weights2.tolist()[0][0])

        for Head_validate_specific_knowledge_sequence_list, attn_weight_head, validate_tail in zip(
                Heads_validate_specific_knowledge_sequence_list, attn_weight_heads, validate_Tails.tolist()):

            # 使用列表推导式过滤列表
            A = [(a, b) for a, b in zip(Head_validate_specific_knowledge_sequence_list, attn_weight_head) if self.numOfEntity not in a]
            if A:
                filtered_Head_specific_knowledge_sequence, filtered_attn_weight_head = zip(*A)
                filtered_Head_specific_knowledge_sequence = list(filtered_Head_specific_knowledge_sequence)
                filtered_attn_weight_head = list(filtered_attn_weight_head)
            else:
                numOfValidateTriple_first_Head = numOfValidateTriple_first_Head + 1
                filtered_Head_specific_knowledge_sequence = [[self.numOfEntity, self.numOfRelation, self.numOfEntity]]
                filtered_attn_weight_head = [1]

            # 再次使用 zip 结合两个列表并根据 attn_weight 的元素排序
            combined_Head = list(zip(filtered_attn_weight_head, filtered_Head_specific_knowledge_sequence))
            combined_Head.sort(reverse=True)  # 指定降序排序

            # 解压排序后的列表
            filtered_attn_weight_sorted_head, filtered_head_specific_knowledge_sequence_sorted = zip(*combined_Head)

            # 将结果转换为列表形式
            filtered_head_specific_knowledge_sequence_sorted_list = list(filtered_head_specific_knowledge_sequence_sorted)

            # 预测尾实体的MR
            for i in range(len(filtered_head_specific_knowledge_sequence_sorted_list)):
                if validate_tail in filtered_head_specific_knowledge_sequence_sorted_list[i][:1] or validate_tail in filtered_head_specific_knowledge_sequence_sorted_list[i][2:3]:
                    MR_tail.append(i+1)
                    break

        return sum(MR_tail) / len(MR_tail)

        print("Fast_Valid End")

    def readTestTriples(self):
        fileName = "/test.txt"
        print("-----Reading Test Triples from " + self.inAdd + fileName + "-----")
        count = 0
        self.test2id["h"] = []
        self.test2id["r"] = []
        self.test2id["t"] = []
        self.test2id["T"] = []
        inputData = open(self.inAdd + fileName)
        line = inputData.readline()
        self.numOfTestTriple = int(re.findall(r"\d+", line)[0])
        line = inputData.readline()
        while line and line not in ["\n", "\r\n", "\r"]:
            reR = re.findall(r"\d+", line)
            if reR:
                tmpHead = int(re.findall(r"\d+", line)[0])
                tmpTail = int(re.findall(r"\d+", line)[2])
                tmpRelation = int(re.findall(r"\d+", line)[1])
                tmpTime = int(re.findall(r"\d+", line)[3])
                self.test2id["h"].append(tmpHead)
                self.test2id["r"].append(tmpRelation)
                self.test2id["t"].append(tmpTail)
                self.test2id["T"].append(tmpTime)
                count += 1
            else:
                print("error in " + fileName + " at Line " + str(count + 2))
            line = inputData.readline()
        inputData.close()

        if count == self.numOfTestTriple:
            print("number of test triples: " + str(self.numOfTestTriple))

        else:
            print("count: " + str(count))
            print("expected number of test triples:" + str(self.numOfTestTriple))
            print("error in " + fileName)

    def test(self, model):

        print("-----Test Started at " + time.strftime('%m-%d-%Y %H:%M:%S', time.localtime(time.time())) + "-----")

        attn_weight_heads = []
        attn_weight_tails = []

        numOfTestTriple_first_Head = 0
        numOfTestTriple_first_Tail = 0

        MR_tail = []
        hit_1_tail = 0
        hit_3_tail = 0
        hit_10_tail = 0

        MR_head = []
        hit_1_head = 0
        hit_3_head = 0
        hit_10_head = 0

        test_Heads = torch.tensor(self.test2id["h"])
        test_Tails = torch.tensor(self.test2id["t"])
        test_Relations = torch.tensor(self.test2id["r"])
        test_Times = torch.tensor(self.test2id["T"])

        Heads_test_specific_knowledge_sequence = generate_knowledge_sequence(self.Training_specific_knowledge_sequence,
                                                                             test_Heads, test_Times,
                                                                             self.numOfEntity,
                                                                             self.numOfRelation).BatchEntity_Training_specific_knowledge_sequence
        Tails_test_specific_knowledge_sequence = generate_knowledge_sequence(self.Training_specific_knowledge_sequence,
                                                                             test_Tails, test_Times,
                                                                             self.numOfEntity,
                                                                             self.numOfRelation).BatchEntity_Training_specific_knowledge_sequence

        for Head_test_specific_knowledge_sequence, Tail_test_specific_knowledge_sequence, test_Relation, test_Head, test_Tail in zip(Heads_test_specific_knowledge_sequence, Tails_test_specific_knowledge_sequence, test_Relations, test_Heads, test_Tails):

            attn_weights1, attn_weights2, loss = model(Head_test_specific_knowledge_sequence.unsqueeze(0).to(self.device),
                                                       Tail_test_specific_knowledge_sequence.unsqueeze(0).to(self.device), test_Relation.unsqueeze(0).to(self.device),
                                                       test_Head.unsqueeze(0).to(self.device), test_Tail.unsqueeze(0).to(self.device), None)

            attn_weight_heads.append(attn_weights1.tolist()[0][0])
            attn_weight_tails.append(attn_weights2.tolist()[0][0])

        mum = 0
        for Head_test_specific_knowledge_sequence_list, Tail_test_specific_knowledge_sequence_list, attn_weight_head, attn_weight_tail, test_head, test_tail in zip(
                Heads_test_specific_knowledge_sequence.tolist(), Tails_test_specific_knowledge_sequence.tolist(), attn_weight_heads,
                attn_weight_tails, test_Heads.tolist(), test_Tails.tolist()):

            # 使用列表推导式同时过滤两个列表
            A = [(a, b) for a, b in zip(Head_test_specific_knowledge_sequence_list, attn_weight_head) if self.numOfEntity not in a]
            if A:
                filtered_Head_specific_knowledge_sequence, filtered_attn_weight_head = zip(*A)
                filtered_Head_specific_knowledge_sequence = list(filtered_Head_specific_knowledge_sequence)
                filtered_attn_weight_head = list(filtered_attn_weight_head)
            else:
                numOfTestTriple_first_Head = numOfTestTriple_first_Head + 1
                filtered_Head_specific_knowledge_sequence = [[self.numOfEntity, self.numOfRelation, self.numOfEntity]]
                filtered_attn_weight_head = [1]

            B = [(a, b) for a, b in zip(Tail_test_specific_knowledge_sequence_list, attn_weight_tail) if self.numOfEntity not in a]
            if B:
                filtered_Tail_specific_knowledge_sequence, filtered_attn_weight_tail = zip(*B)
                filtered_Tail_specific_knowledge_sequence = list(filtered_Tail_specific_knowledge_sequence)
                filtered_attn_weight_tail = list(filtered_attn_weight_tail)
            else:
                numOfTestTriple_first_Tail = numOfTestTriple_first_Tail + 1
                filtered_Tail_specific_knowledge_sequence = [[self.numOfEntity, self.numOfRelation, self.numOfEntity]]
                filtered_attn_weight_tail = [1]

            # 再次使用 zip 结合两个列表并根据 attn_weight 的元素排序
            combined_Head = list(zip(filtered_attn_weight_head, filtered_Head_specific_knowledge_sequence))
            combined_Head.sort(reverse=True)  # 指定降序排序
            combined_Tail = list(zip(filtered_attn_weight_tail, filtered_Tail_specific_knowledge_sequence))
            combined_Tail.sort(reverse=True)  # 指定降序排序

            # 解压排序后的列表
            filtered_attn_weight_sorted_head, filtered_head_specific_knowledge_sequence_sorted = zip(*combined_Head)
            filtered_attn_weight_sorted_tail, filtered_tail_specific_knowledge_sequence_sorted = zip(*combined_Tail)

            # 将结果转换为列表形式
            filtered_attn_weight_sorted_head_list = list(filtered_attn_weight_sorted_head)
            filtered_head_specific_knowledge_sequence_sorted_list = list(filtered_head_specific_knowledge_sequence_sorted)

            filtered_attn_weight_sorted_tail_list = list(filtered_attn_weight_sorted_tail)
            filtered_tail_specific_knowledge_sequence_sorted_list = list(filtered_tail_specific_knowledge_sequence_sorted)

            # 预测尾实体
            for i in range(len(filtered_head_specific_knowledge_sequence_sorted_list)):
                if test_tail in filtered_head_specific_knowledge_sequence_sorted_list[i][:1] or test_tail in filtered_head_specific_knowledge_sequence_sorted_list[i][2:3]:
                    MR_tail.append(i+1)
                    break

            head_specific_knowledge_sequence_sorted_1 = filtered_head_specific_knowledge_sequence_sorted_list[:1]
            for knowledge_1 in head_specific_knowledge_sequence_sorted_1:
                if test_tail in knowledge_1[:1] or test_tail in knowledge_1[2:3]:
                    hit_1_tail = hit_1_tail + 1
                    break

            head_specific_knowledge_sequence_sorted_3 = filtered_head_specific_knowledge_sequence_sorted_list[:3]
            for knowledge_3 in head_specific_knowledge_sequence_sorted_3:
                if test_tail in knowledge_3[:1] or test_tail in knowledge_3[2:3]:
                    hit_3_tail = hit_3_tail + 1
                    break

            head_specific_knowledge_sequence_sorted_10 = filtered_head_specific_knowledge_sequence_sorted_list[:10]
            for knowledge_10 in head_specific_knowledge_sequence_sorted_10:
                if test_tail in knowledge_10[:1] or test_tail in knowledge_10[2:3]:
                    hit_10_tail = hit_10_tail + 1
                    break

            # 预测头实体
            for i in range(len(filtered_tail_specific_knowledge_sequence_sorted_list)):
                if test_head in filtered_tail_specific_knowledge_sequence_sorted_list[i][:1] or test_head in filtered_tail_specific_knowledge_sequence_sorted_list[i][2:3]:
                    MR_head.append(i+1)
                    break

            tail_specific_knowledge_sequence_sorted_1 = filtered_tail_specific_knowledge_sequence_sorted_list[:1]
            for knowledge_1 in tail_specific_knowledge_sequence_sorted_1:
                if test_head in knowledge_1[:1] or test_head in knowledge_1[2:3]:
                    hit_1_head = hit_1_head + 1
                    break

            tail_specific_knowledge_sequence_sorted_3 = filtered_tail_specific_knowledge_sequence_sorted_list[:3]
            for knowledge_3 in tail_specific_knowledge_sequence_sorted_3:
                if test_head in knowledge_3[:1] or test_head in knowledge_3[2:3]:
                    hit_3_head = hit_3_head + 1
                    break

            tail_specific_knowledge_sequence_sorted_10 = filtered_tail_specific_knowledge_sequence_sorted_list[:10]
            for knowledge_10 in tail_specific_knowledge_sequence_sorted_10:
                if test_head in knowledge_10[:1] or test_head in knowledge_10[2:3]:
                    hit_10_head = hit_10_head + 1
                    break

            mum = mum + 1

            if mum % 1000 == 0:
                print(str(mum) + " test triples processed!")


        print("-----Result of Entity Link Prediction (Tail entity)-----")
        print("|  numOfTestTriple |" + str(self.numOfTestTriple) + "  |")
        print("|  MR_tail_len |" + str(len(MR_tail)) + "  |")
        print("|  numOfTestTriple_first_Head |" + str(numOfTestTriple_first_Head) + "  |")
        print("|  Mean Rank  |" + str(sum(MR_tail) / len(MR_tail)) + "  |")
        print("|  HR @ 1  |" + str(hit_1_tail) + "  |" + str(hit_1_tail / self.numOfTestTriple) + "  |" + str(hit_1_tail / len(MR_tail)) + "  |")
        print("|  HR @ 3  |" + str(hit_3_tail) + "  |" + str(hit_3_tail / self.numOfTestTriple) + "  |" + str(hit_3_tail / len(MR_tail)) + "  |")
        print("|  HR @ 10 |" + str(hit_10_tail) + "  |" + str(hit_10_tail / self.numOfTestTriple) + "  |" + str(hit_10_tail / len(MR_tail)) + "  |")

        print("-----Result of Entity Link Prediction (Head entity)-----")
        print("|  numOfTestTriple |" + str(self.numOfTestTriple) + "  |")
        print("|  MR_head_len |" + str(len(MR_head)) + "  |")
        print("|  numOfTestTriple_first_Tail |" + str(numOfTestTriple_first_Tail) + "  |")
        print("|  Mean Rank  |" + str(sum(MR_head) / len(MR_head)) + "  |")
        print("|  HR @ 1  |" + str(hit_1_head) + "  |" + str(hit_1_head / self.numOfTestTriple) + "  |" + str(hit_1_head / len(MR_head)) + "  |")
        print("|  HR @ 3  |" + str(hit_3_head) + "  |" + str(hit_3_head / self.numOfTestTriple) + "  |" + str(hit_3_head / len(MR_head)) + "  |")
        print("|  HR @ 10 |" + str(hit_10_head) + "  |" + str(hit_10_head / self.numOfTestTriple) + "  |" + str(hit_10_head / len(MR_head)) + "  |")

        print("-----Test Ended at " + time.strftime('%m-%d-%Y %H:%M:%S', time.localtime(time.time())) + "-----")

    def preRead(self, model):
        print("-----Reading Pre-Trained Results from " + self.preAdd + "/model_parameters.pth")
        # 加载之前保存的参数
        model.load_state_dict(torch.load(self.preAdd + "/model_parameters.pth"))

    def Read(self, output_path):
        # 以二进制读取模式打开文件
        with open(output_path, 'rb') as f:
            # 从文件中加载模型
            Model = pickle.load(f)
            return Model

if __name__ == '__main__':
    train = trainModel()

