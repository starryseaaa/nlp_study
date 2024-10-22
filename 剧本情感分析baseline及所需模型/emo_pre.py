from tqdm import tqdm #Python进度条，可以在长循环中添加一个进度提示信息，用户只需封装任意的迭代器tqdm(iterator)
import pandas as pd  #Python的一个数据分析库 “关系”或“标签”数据的处理工作简单化
import os#用于固定某些参数值，返回一个新的函数
from functools import partial
import numpy as np #数学函数处理以及高效的多维数组对象
import time

import torch #用于计算机视觉和自然语言处理等领域
import torch.nn as nn  #nn模块定义了构建神经网络的类和方法
import torch.nn.functional as F  #functional模块包含了一些用于构建神经网络的函数，比如激活函数和损失函数

from torch.utils.data import DataLoader #DataLoader提供了一个可迭代的对象，支持自动批处理、打乱数据和并行加载数据等功能。
from torch.utils.data.dataset import Dataset #dataset所有数据集类的基类，用户可以通过继承这个类来自定义自己的数据集。
from transformers import BertPreTrainedModel, BertTokenizer, BertConfig, BertModel  #用于处理BERT模型的类，包括预训练模型、分词器、配置和模型本身。
from functools import partial
from transformers import AdamW, get_linear_schedule_with_warmup #AdamW是Adam优化器的一个变体，get_linear_schedule_with_warmup用于创建学习率调度器

import os
os.environ["CUDA_VISIBLE_DEVICES"] = '0'#os.environ 是一个字典对象，它代表了当前操作系统的环境变量，通过设置或修改这个字典中的键值对，你可以更改或添加环境变量#，这些变量可以被操作系统及其运行的程序所使用。
#只有编号为0的GPU设备对当前进程可见
with open(r'E:\文件\下载\train_dataset_v2.tsv', 'r', encoding='utf-8') as handler:
    lines = handler.read().split('\n')[1:-1]

    data = list()
    for line in tqdm(lines):
        sp = line.split('\t')
        if len(sp) != 4:  #就是一行需要有四列元素，不等于四就是有缺失值，需要报错
            print("Error: ", sp)
            continue
        data.append(sp)
#数据处理
train = pd.DataFrame(data) #使用Pandas将处理后的数据转换为DataFrame，并命名列
train.columns = ['id', 'content', 'character', 'emotions']  #训练集打上列标签

test = pd.read_csv(r'E:\文件\下载\test_dataset.tsv', sep='\t')
submit = pd.read_csv(r'E:\文件\下载\submit_example.tsv', sep='\t')
train = train[train['emotions'] != '']

train['text'] = train[ 'content'].astype(str)  +'角色: ' + train['character'].astype(str)
test['text'] = test['content'].astype(str) + ' 角色: ' + test['character'].astype(str)  #都转换成str类型
#将训练集和测试集中的content和character列合并，中间加上"角色:"作为分隔符，生成新的text列。
train['emotions'] = train['emotions'].apply(lambda x: [int(_i) for _i in x.split(',')])  #情绪数据转换成数值向量
##将emotions列中的情绪数据从字符串转换为整数列表
train[['love', 'joy', 'fright', 'anger', 'fear', 'sorrow']] = train['emotions'].values.tolist() #自定义情绪标签 用emotion数值向量中每个数值打分
test[['love', 'joy', 'fright', 'anger', 'fear', 'sorrow']] =[0,0,0,0,0,0] #测试集都初始化0
##将情绪数据拆分为六个独立的列（love, joy, fright, anger, fear, sorrow），这些列代表不同情绪标签的得分
train.to_csv(r'E:\data_process\train.csv',columns=['id', 'content', 'character','text','love', 'joy', 'fright', 'anger', 'fear', 'sorrow'],
            sep='\t',
            index=False)
#分别保存为CSV文件
test.to_csv(r'E:\data_process\test.csv',columns=['id', 'content', 'character','text','love', 'joy', 'fright', 'anger', 'fear', 'sorrow'],
            sep='\t',
            index=False)
#定义dataset
target_cols=['love', 'joy', 'fright', 'anger', 'fear', 'sorrow']
class RoleDataset(Dataset): #RoleDataset类继承自Dataset，用于加载和处理数据
    def __init__(self, tokenizer, max_len, mode='train'): #初始化时，根据mode参数（'train'或'test'）读取相应的CSV文件
        super(RoleDataset, self).__init__()
        if mode == 'train':   #分开测试集和训练集
            self.data = pd.read_csv(r'E:\data_process\train.csv',sep='\t')
        else:
            self.data = pd.read_csv(r'E:\data_process\test.csv',sep='\t')
        self.texts=self.data['text'].tolist()   #字符串又转换成列表的形式？
        self.labels=self.data[target_cols].to_dict('records') #从数据中提取text列作为输入文本，target_cols指定的列作为标签
        self.tokenizer = tokenizer
        self.max_len = max_len #接收一个分词器（tokenizer）和一个最大长度（max_len），用于后续文本的分词和截断

    # __getitem__和__len__方法来创建一个自定义的PyTorch数据集
    def __getitem__(self, index):
        text=str(self.texts[index])
        label=self.labels[index]

        encoding=self.tokenizer.encode_plus(text,
                                            add_special_tokens=True,
                                            max_length=self.max_len,
                                            return_token_type_ids=True,
                                            pad_to_max_length=True,
                                            return_attention_mask=True,
                                            return_tensors='pt',)

        sample = {
            'texts': text,
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten()  #用于将多维数组转换为一维数组
        }

        for label_col in target_cols:
            sample[label_col] = torch.tensor(label[label_col], dtype=torch.int64)
        return sample

    def __len__(self):
        return len(self.texts)
#创建dataloader  打乱 然后批处理
def create_dataloader(dataset, batch_size, mode='train'):
    shuffle = True if mode == 'train' else False

    if mode == 'train':
        data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)
    else:
        data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)
    return data_loader

# roberta #分词器和模型
PRE_TRAINED_MODEL_NAME='hfl/chinese-roberta-wwm-ext'
#tokenizer = BertTokenizer.from_pretrained(PRE_TRAINED_MODEL_NAME)
#base_model = BertModel.from_pretrained(PRE_TRAINED_MODEL_NAME)  # 加载预训练模型

tokenizer =  BertTokenizer.from_pretrained("F:\python_jupyter\pythonProject\hflchinese-roberta-wwm-ext")
base_model = BertModel.from_pretrained("F:\python_jupyter\pythonProject\hflchinese-roberta-wwm-ext")
class EmotionClassifier(nn.Module):  #构建神经网络模型定义情感分类模型，继承自nn.Module，并在初始化时接收情感类别的数量和BERT模型。然后，它为每个情感类别定义了一个线性层来输出分类结果
    def __init__(self, n_classes, bert):
        super(EmotionClassifier, self).__init__() #n_classes（情感类别的数量）和bert（一个BERT模型实例）。
        self.bert = bert#将传入的BERT模型实例保存为类的属性
        self.out_love = nn.Linear(self.bert.config.hidden_size, n_classes)
        self.out_joy = nn.Linear(self.bert.config.hidden_size, n_classes)
        self.out_fright = nn.Linear(self.bert.config.hidden_size, n_classes)
        self.out_anger = nn.Linear(self.bert.config.hidden_size, n_classes)
        self.out_fear = nn.Linear(self.bert.config.hidden_size, n_classes)
        self.out_sorrow = nn.Linear(self.bert.config.hidden_size, n_classes)#这些线性层都将BERT的输出（隐藏状态的大小）映射到情感类别的数量上

    def forward(self, input_ids, attention_mask):#input_ids（输入文本的token IDs）和attention_mask（用于指示哪些token是有效的，哪些应该被忽略）。
        _, pooled_output = self.bert( #通过BERT模型传递输入，并接收两个返回值（但您只保留了第二个，即pooled_output）
            input_ids=input_ids,
            attention_mask=attention_mask,
            return_dict = False
        )
        love = self.out_love(pooled_output)
        joy = self.out_joy(pooled_output)
        fright = self.out_fright(pooled_output)
        anger = self.out_anger(pooled_output)
        fear = self.out_fear(pooled_output)
        sorrow = self.out_sorrow(pooled_output)  #通过相应的线性层传递pooled_output，并保存结果
        return {
            'love': love, 'joy': joy, 'fright': fright,
            'anger': anger, 'fear': fear, 'sorrow': sorrow,
        }

EPOCHS=1  #epoch轮次 因为比较慢就用了->等于0/1时迭代，之后停止迭代
weight_decay=0.005#、权重衰减、。正则化技术，用于防止模型过拟合。
data_path='data'#数据路径、
warmup_proportion=0.0#用于调整学习率，热身比例
batch_size=64#每次迭代中用于训练模型的数据样本数量、批量大小
lr = 2e-5#用于控制模型在每次迭代中更新权重的步长 学习率
max_len = 128#最大序列长度对于文本数据，这通常意味着将文本截断或填充到指定的长度

warm_up_ratio = 0

trainset = RoleDataset(tokenizer, max_len, mode='train')
train_loader = create_dataloader(trainset, batch_size, mode='train')

valset = RoleDataset(tokenizer, max_len, mode='test')
valid_loader = create_dataloader(valset, batch_size, mode='test')

model = EmotionClassifier(n_classes=4, bert=base_model)
model.cuda()

optimizer = AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
total_steps = len(train_loader) * EPOCHS


scheduler = get_linear_schedule_with_warmup(
  optimizer,
  num_warmup_steps=warm_up_ratio*total_steps,
  num_training_steps=total_steps
)

criterion = nn.CrossEntropyLoss().cuda()
#定义训练过程。在每个epoch中，遍历训练数据加载器，计算损失，进行反向传播，更新模型参数，并打印训练日志。
def do_train(model, date_loader, criterion, optimizer, scheduler, metric=None):
    model.train()
    global_step = 0
    tic_train = time.time()
    log_steps = 100
    for epoch in range(EPOCHS):
        losses = []
        for step, sample in enumerate(train_loader):
            input_ids = sample["input_ids"].cuda()
            attention_mask = sample["attention_mask"].cuda()

            outputs = model(input_ids=input_ids, attention_mask=attention_mask)

            loss_love = criterion(outputs['love'], sample['love'].cuda())
            loss_joy = criterion(outputs['joy'], sample['joy'].cuda())
            loss_fright = criterion(outputs['fright'], sample['fright'].cuda())
            loss_anger = criterion(outputs['anger'], sample['anger'].cuda())
            loss_fear = criterion(outputs['fear'], sample['fear'].cuda())
            loss_sorrow = criterion(outputs['sorrow'], sample['sorrow'].cuda())
            loss = loss_love + loss_joy + loss_fright + loss_anger + loss_fear + loss_sorrow

            losses.append(loss.item())

            loss.backward()

#             nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()

            global_step += 1

            if global_step % log_steps == 0:
                print("global step %d, epoch: %d, batch: %d, loss: %.5f, speed: %.2f step/s, lr: %.10f"
                      % (global_step, epoch, step, np.mean(losses), global_step / (time.time() - tic_train),
                         float(scheduler.get_last_lr()[0])))

#第一处print
do_train(model, train_loader, criterion, optimizer, scheduler) #调用自定义的函数？
#模型预测
from collections import defaultdict

model.eval()
#评估部分遍历验证数据加载器，对每个批次的数据进行预测，并将预测结果存储在test_pred字典中。只打印了第一个批次的预测结果并中断了循环（break），
test_pred = defaultdict(list)
for step, batch in tqdm(enumerate(valid_loader)):
    b_input_ids = batch['input_ids'].cuda()
    attention_mask = batch["attention_mask"].cuda()
    logists = model(input_ids=b_input_ids, attention_mask=attention_mask)
    for col in target_cols:
        out2 = torch.argmax(logists[col], axis=1)
        test_pred[col].append(out2.cpu().numpy())

    print(test_pred)
    break


def predict(model, test_loader):
    val_loss = 0#
    test_pred = defaultdict(list)# 创建一个默认字典来存储每个标签的预测结果
    model.eval()#将模型设置为评估模式
    for step, batch in tqdm(enumerate(test_loader)):#使用tqdm显示进度条，遍历测试数据加载器
        b_input_ids = batch['input_ids'].cuda()#将输入ID移至GPU
        attention_mask = batch["attention_mask"].cuda() #将注意力掩码移到GPU
        with torch.no_grad():# 禁用梯度计算，以节省内存和计算资源
            logists = model(input_ids=b_input_ids, attention_mask=attention_mask)# 进行前向传播
            for col in target_cols: # 遍历所有目标列（情感标签）
                out2 = torch.argmax(logists[col], axis=1)# 对每个标签的预测结果取最大值所在的索引
                test_pred[col].extend(out2.cpu().numpy().tolist()) # 将预测结果转换为列表并扩展到相应的字典中

    return test_pred  # 返回包含所有标签预测结果的字典
# 读取提交文件的示例
submit = pd.read_csv(r'E:\文件\下载\submit_example.tsv', sep='\t')
test_pred = predict(model, valid_loader)# 使用验证数据加载器和模型进行预测

print(test_pred['love'][:10])# 打印'love'标签的前10个预测结果
print(len(test_pred['love']))# 打印'love'标签预测结果的总数
# 准备将所有标签的预测结果合并
label_preds = []   # 创建一个空列表来存储所有标签的预测结果
for col in target_cols:# 遍历所有目标列
    preds = test_pred[col] # 获取当前标签的预测结果
    label_preds.append(preds)  # 将预测结果添加到列表中
print(len(label_preds[0]))
sub = submit.copy()# 复制提交文件的示例
sub['emotion'] = np.stack(label_preds, axis=1).tolist()# 将所有标签的预测结果合并为一个二维数组，并转换为列表的列表形式
sub['emotion'] = sub['emotion'].apply(lambda x: ','.join([str(i) for i in x]))# 将二维列表转换为字符串形式，每个元素之间用逗号分隔
sub.to_csv('baseline_{}.tsv'.format(PRE_TRAINED_MODEL_NAME.split('/')[-1]), sep='\t', index=False)# 将结果保存到新的TSV文件中
sub.head()# 打印提交文件的前几行以检查