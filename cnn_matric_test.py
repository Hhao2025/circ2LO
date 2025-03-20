import torch
import torch.nn as nn
import random
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import os
import numpy as np
#from cuml.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
# 定义一个生成器函数，用于逐批加载数据
def batch_generator(data_folder, batch_size):

        filenames = [os.path.join(data_folder, f) for f in os.listdir(data_folder) if f.endswith('.pt')]
        total_files = len(filenames)
        random.shuffle(filenames)
   # while True:  # 这是一个无限循环的生成器
        for i in range(0, total_files, batch_size):
            batch_files = filenames[i:i+batch_size]
            batch_data = []
            batch_labels = []
            print(i)
            for file_path in batch_files:
                tensor = torch.load(file_path)

               # print(f'Epoch {epoch + 1}/{num_epochs}, Loss: {loss.item()}, Val Loss: {val_loss}')
                # 假设每个文件包含一个样本，这里需要根据实际情况调整
                batch_data.append(tensor)
                # 假设正样本文件名包含'pos'，负样本文件名包含'neg'
                if 'pos' in file_path:
                    batch_labels.append(1)
                else:
                    batch_labels.append(0)
            #yield AugmentedTensorDataset(torch.tensor(np.array(batch_data)),torch.tensor(np.array(batch_labels)))
            yield torch.tensor(np.array(batch_data)), torch.tensor(np.array(batch_labels))
           # yield train_dataset



def load_all_tensors_to_matrix(folder_path):
    tensors = []
    i=0
    batch_labels = []
    for filename in os.listdir(folder_path):
        if filename.endswith('.pt'):  # 检查文件扩展名是否为 '.pt'
            file_path = os.path.join(folder_path, filename)
            tensor = torch.load(file_path)
            tensors.append(tensor)
            i=i+1
            #print(i)
            if 'pos' in filename:
                batch_labels.append(1)
            else:
                batch_labels.append(0)
    return torch.tensor(np.array(tensors)),torch.tensor(np.array(batch_labels))
class ResidualBlock(nn.Module):
    def __init__(self, channels):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv1d(channels, channels, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm1d(channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv1d(channels, channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm1d(channels)

    def forward(self, x):
        residual = x
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += residual
        out = self.relu(out)
        return out


# 定义数据增强函数
def augment_data(sequence, max_shift=3):
    # 随机打乱序列
    if np.random.rand() < 0.5:
        sequence = np.random.permutation(sequence)

    # 窗口滑动
    shift = np.random.randint(0, max_shift)
    sequence = np.roll(sequence, shift)

    # 添加噪声
    noise = np.random.normal(0, 0.1, sequence.shape)
    sequence = sequence + noise

    return sequence


# 定义数据增强的Dataset类
class AugmentedTensorDataset(TensorDataset):
    def __init__(self, *tensors):
        super(AugmentedTensorDataset, self).__init__(*tensors)

    def __getitem__(self, index):
        x, y = super().__getitem__(index)
        x_augmented = augment_data(x.numpy())
        return torch.tensor(x_augmented, dtype=torch.float32), y
# 定义卷积网络模型
class RNAConvNet(nn.Module):
    def __init__(self, embedding_dim=2560):
        super(RNAConvNet, self).__init__()


        self.conv1 = nn.Conv1d(202, 64, kernel_size=7, padding=3)
        self.bn1 = nn.BatchNorm1d(64)
        #self.relu = nn.ReLU(inplace=True)
        self.leaky_relu = nn.LeakyReLU(negative_slope=0.01)  # 使用LeakyReLU
        self.dropout1 = nn.Dropout(0.7)
        #self.res_block = ResidualBlock(64)
        # 添加更多的残差块
        # self.res_blocks = nn.Sequential(
        #     ResidualBlock(64),
        #     ResidualBlock(64),
        #     ResidualBlock(64),
        #     ResidualBlock(64)
        # )
        self.conv2 = nn.Conv1d(64, 128, kernel_size=5, padding=2)
        self.bn2 = nn.BatchNorm1d(128)
       # self.dropout2 = nn.Dropout(0.7)
        self.conv3 = nn.Conv1d(128, 256, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm1d(256)
        #self.dropout3 = nn.Dropout(0.7)  # 添加Dropout层
        self.dropout = nn.Dropout()
        self.pool = nn.AdaptiveMaxPool1d(1)

        self.fc = nn.Linear(256, 2)

    def forward(self, x):
        #x = x.permute(0, 2, 1)  # 调整维度从[N, 202, 2560]到[N, 2560, 202]
        #x = x.unsqueeze(1)
        #x = self.relu(self.conv1(x))
        x = self.leaky_relu(self.conv1(x))  # 使用LeakyReLU
        #x = self.dropout1(x)
        #x = x.unsqueeze(1)  # 增加一个通道维度
        #x = self.relu(self.bn1(self.conv1(x)))
        #x = self.res_blocks(x)
        #x = self.relu(self.bn2(self.conv2(x)))
        x = self.leaky_relu(self.conv2(x))  # 使用LeakyReLU
        #x = self.dropout2(x)
        #x = self.relu(self.bn3(self.conv3(x)))
        x = self.leaky_relu(self.conv3(x))  # 使用LeakyReLU
        x = self.dropout(x)
        x = self.pool(x)
        x = x.view(x.size(0), -1)
        #x = self.dropout(x)  # 应用Dropout
        x = self.fc(x)
        return x


#y_train=y_train.unsqueeze(1)
#y_train1 = y_train # 将标签转换为长整型
#y_test=torch.tensor(y_test1).int()
#y_test=y_test.unsqueeze(1)
#y_test=y_test


# 数据增强创建数据集和数据加载器
#  train_dataset = AugmentedTensorDataset(X_train, y_train)
#  train_loader = DataLoader(dataset=train_dataset, batch_size=64, shuffle=True)
# test_dataset = TensorDataset(X_test1, y_test)
# test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)
# 创建数据集和数据加载器
# train_dataset = TensorDataset(X_train, y_train)
# train_loader = DataLoader(dataset=train_dataset, batch_size=64, shuffle=True)
# test_dataset = TensorDataset(X_test1, y_test)
# test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)


#train_generator = batch_generator(train_data_folder, train_batch_size)
# test_dataset = TensorDataset(X_test1, y_test)
# test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)



# 修改训练模型函数以使用生成器
def train_model_nogen(model, criterion, optimizer, dataloader, num_epochs, early_stopping_patience):
    model.train()
    min_val_loss = float('inf')
    patience_counter = 0
    for epoch in range(num_epochs):
        for inputs, labels in dataloader:
            inputs, labels = inputs.long().cuda(), labels.long().cuda()
            inputs=inputs.float()
            #labels=labels.float()
            optimizer.zero_grad()
            #inputs=inputs.float()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
        val_loss=10
        val_loss = evaluate_model(model, dataloader, criterion)  # 计算验证集损失
        print(f'Epoch {epoch+1}/{num_epochs}, Loss: {loss.item()}, Val Loss: {val_loss}')
        # 早停策略
        if val_loss < min_val_loss:
            min_val_loss = val_loss
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= early_stopping_patience:
                print("Early stopping triggered.")
                break
# 训练模型
def train_model(model, criterion, optimizer, dataloader, num_epochs, early_stopping_patience):
    model.train()
    min_val_loss = float('inf')
    patience_counter = 0
    for epoch in range(num_epochs):
        train_generator = batch_generator(train_data_folder, train_batch_size)
        #train_loader1 = DataLoader(dataset=train_generator, batch_size=train_batch_size, shuffle=True)
        #train_dataset = AugmentedTensorDataset(X_train, y_train)
        #train_dataset = batch_generator(train_data_folder, train_batch_size)
        #train_loader1 = DataLoader(dataset=train_dataset, batch_size=64, shuffle=True)
        for inputs, labels in train_generator:
            # train_dataset = AugmentedTensorDataset(inputs, labels)
            # train_loader1 = DataLoader(dataset=train_dataset, batch_size=64, shuffle=True)
            #for inputs, labels in train_loader1:
              inputs, labels = inputs.cuda(), labels.cuda()
              optimizer.zero_grad()
              outputs = model(inputs)
              loss = criterion(outputs, labels)
              loss.backward()
              optimizer.step()
        # 在每个epoch后更新学习率
        scheduler.step()
        val_loss = evaluate_model(model, dataloader, criterion)  # 计算验证集损失
        #val_loss = evaluate_model(model, dataloader, criterion)  # 计算验证集损失
        print(f'Epoch {epoch+1}/{num_epochs}, Loss: {loss.item()}, Val Loss: {val_loss}')
        # 早停策略
        if val_loss < min_val_loss:
            min_val_loss = val_loss
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= early_stopping_patience:
                print("Early stopping triggered.")

                break

# 评估模型并计算损失
def evaluate_model(model, dataloader, criterion):
    model.eval()
    total_loss = 0
    correct = 0
    total = 0
    all_preds = []
    all_labels = []
    with torch.no_grad():
        for inputs, labels in dataloader:
            inputs, labels = inputs.cuda(), labels.long().cuda()
            inputs=inputs.float()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            total_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            all_preds.extend(predicted.tolist())
            all_labels.extend(labels.tolist())
    accuracy = 100 * correct / total
    print(f'Accuracy of the network on the var-train data: {accuracy}%')
    avg_loss = total_loss / len(dataloader)
    # 计算精确度
    precision = precision_score(all_labels, all_preds)
    print(f'Precision: {precision}')
    # 计算召回率
    recall = recall_score(all_labels, all_preds)
    print(f'recall: {recall}')
    # 计算F1评分
    f1 = f1_score(all_labels, all_preds)
    print(f'F1-score: {f1}')

    # 计算马修相关系数
    mcc = matthews_corrcoef(all_labels, all_preds)
    print(f'MCC: {mcc}')
    return avg_loss

from sklearn.metrics import precision_score, f1_score, matthews_corrcoef, recall_score


def evaluate_model1(model, dataloader):
    model.eval()
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for inputs, labels in dataloader:
            inputs, labels = inputs.cuda(), labels.long().cuda()  # 将数据移动到GPU
            inputs = inputs.float()
            outputs = model(inputs)
            _, predicted = torch.max(outputs.data, 1)
            all_preds.extend(predicted.tolist())
            all_labels.extend(labels.tolist())

    # 计算准确率
    correct = sum([pred == label for pred, label in zip(all_preds, all_labels)])
    accuracy = 100 * correct / len(all_labels)
    print(f'Accuracy of the network on the test data: {accuracy}%')

    # 计算精确度
    precision = precision_score(all_labels, all_preds)
    print(f'Precision: {precision}')
    # 计算召回率
    recall = recall_score(all_labels, all_preds)
    print(f'recall: {recall}')
    # 计算F1评分
    f1 = f1_score(all_labels, all_preds)
    print(f'F1-score: {f1}')

    # 计算马修相关系数
    mcc = matthews_corrcoef(all_labels, all_preds)
    print(f'MCC: {mcc}')
# 实例化模型
model = RNAConvNet().cuda()

# 损失函数和优化器
criterion = nn.CrossEntropyLoss()
#optimizer = optim.Adam(model.parameters(), lr=0.001)
#optimizer = optim.Adam(model.parameters(), lr=0.0001)  # 降低学习率
#optimizer = optim.Adam(model.parameters(), lr=0.0001, weight_decay=1e-5)
# 使用AdamW优化器
optimizer = optim.AdamW(model.parameters(), lr=0.004, weight_decay=1e-5)

# 设置学习率衰减
# 这里的scheduler将在每10个epoch后将学习率乘以0.1
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)
# 假设我们有一些训练数据
# X_train: 训练数据，维度为[batch_size, embedding_dim]
# y_train: 训练标签，维度为[batch_size]
# 这里只是示例，实际数据需要您提供
#X_train_s = torch.randn(64, 2560)  # 假设的批量大小为64
#y_train1_s = torch.randint(0, 2, (64,))  # 假设的标签
#无位置特征的向量化表示
# posFile='data/nnj_circRNA_newseq.fea'
# negFile='data/nnj_lncRNA_newseq.fea'
#kmer的向量化表示
# posFile='data/tmp1.fea'
# negFile='data/tmp2.fea'
# pos1 = np.loadtxt(posFile, dtype=str)
# neg1 = np.loadtxt(negFile, dtype=str)

# neg1 = load_all_tensors_to_matrix('./data/embedding/nnj/mat/neg/')
# X_pos_train1, X_pos_test1 = train_test_split(pos1, random_state=1, train_size=0.8)
# X_neg_train1, X_neg_test1 = train_test_split(neg1, random_state=1, train_size=0.8)# 创建TensorDataset和DataLoader
# X_train1= np.vstack((X_pos_train1, X_neg_train1))
# X_test1 = np.vstack((X_pos_test1, X_neg_test1))
# y_train1 = np.hstack((np.ones(len(X_pos_train1), int), np.zeros(len(X_neg_train1), int)))
# y_test1 = np.hstack((np.ones(len(X_pos_test1), int), np.zeros(len(X_neg_test1), int)))
# X_train1 = X_train1.astype(float)
# X_test1 = X_test1.astype(float)
# X_train=torch.tensor(X_train1)
# X_test1=torch.tensor(X_test1)
# y_train=torch.tensor(y_train1)
# y_test=torch.tensor(y_test1)
# train_dataset = TensorDataset(X_train, y_train)
# train_loader = DataLoader(dataset=train_dataset, batch_size=64, shuffle=True)
# test_dataset = TensorDataset(X_test1, y_test)
# test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

# 训练和评估
num_epochs = 100
early_stopping_patience = 50 # 早停的耐心值
# 创建训练数据的生成器
train_data_folder = './data/embedding/nnj/mat/train'  # 假设您的数据在这个文件夹中
train_batch_size = 64*2 # 根据您的内存限制调整批次大小
#train_model_nogen(model, criterion, optimizer, train_loader, num_epochs, early_stopping_patience)
x_var,y_var=load_all_tensors_to_matrix('./data/embedding/nnj/mat/test/')
# x_var = x_var.astype(float)
# X_vartrain=torch.tensor(x_var)
# y_vartrain = torch.tensor(y_var)
var_dataset = TensorDataset(x_var, y_var)
var_loader = DataLoader(dataset=var_dataset, batch_size=64, shuffle=True)
train_model(model, criterion, optimizer, var_loader, num_epochs, early_stopping_patience)
#model_path = './data/embedding/nnj/model/model_noembedding.pth'
model_path = './data/embedding/nnj/model/model_cnn1219.pth'
# 保存模型的状态字典
torch.save(model.state_dict(), model_path)
#evaluate_model1(model, test_loader)import torch
import torch.nn as nn
import random
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import os
import numpy as np
#from cuml.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
# 定义一个生成器函数，用于逐批加载数据
def batch_generator(data_folder, batch_size):

        filenames = [os.path.join(data_folder, f) for f in os.listdir(data_folder) if f.endswith('.pt')]
        total_files = len(filenames)
        random.shuffle(filenames)
   # while True:  # 这是一个无限循环的生成器
        for i in range(0, total_files, batch_size):
            batch_files = filenames[i:i+batch_size]
            batch_data = []
            batch_labels = []
            print(i)
            for file_path in batch_files:
                tensor = torch.load(file_path)

               # print(f'Epoch {epoch + 1}/{num_epochs}, Loss: {loss.item()}, Val Loss: {val_loss}')
                # 假设每个文件包含一个样本，这里需要根据实际情况调整
                batch_data.append(tensor)
                # 假设正样本文件名包含'pos'，负样本文件名包含'neg'
                if 'pos' in file_path:
                    batch_labels.append(1)
                else:
                    batch_labels.append(0)
            #yield AugmentedTensorDataset(torch.tensor(np.array(batch_data)),torch.tensor(np.array(batch_labels)))
            yield torch.tensor(np.array(batch_data)), torch.tensor(np.array(batch_labels))
           # yield train_dataset



def load_all_tensors_to_matrix(folder_path):
    tensors = []
    i=0
    batch_labels = []
    for filename in os.listdir(folder_path):
        if filename.endswith('.pt'):  # 检查文件扩展名是否为 '.pt'
            file_path = os.path.join(folder_path, filename)
            tensor = torch.load(file_path)
            tensors.append(tensor)
            i=i+1
            #print(i)
            if 'pos' in filename:
                batch_labels.append(1)
            else:
                batch_labels.append(0)
    return torch.tensor(np.array(tensors)),torch.tensor(np.array(batch_labels))
class ResidualBlock(nn.Module):
    def __init__(self, channels):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv1d(channels, channels, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm1d(channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv1d(channels, channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm1d(channels)

    def forward(self, x):
        residual = x
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += residual
        out = self.relu(out)
        return out


# 定义数据增强函数
def augment_data(sequence, max_shift=3):
    # 随机打乱序列
    if np.random.rand() < 0.5:
        sequence = np.random.permutation(sequence)

    # 窗口滑动
    shift = np.random.randint(0, max_shift)
    sequence = np.roll(sequence, shift)

    # 添加噪声
    noise = np.random.normal(0, 0.1, sequence.shape)
    sequence = sequence + noise

    return sequence


# 定义数据增强的Dataset类
class AugmentedTensorDataset(TensorDataset):
    def __init__(self, *tensors):
        super(AugmentedTensorDataset, self).__init__(*tensors)

    def __getitem__(self, index):
        x, y = super().__getitem__(index)
        x_augmented = augment_data(x.numpy())
        return torch.tensor(x_augmented, dtype=torch.float32), y
# 定义卷积网络模型
class RNAConvNet(nn.Module):
    def __init__(self, embedding_dim=2560):
        super(RNAConvNet, self).__init__()


        self.conv1 = nn.Conv1d(202, 64, kernel_size=7, padding=3)
        self.bn1 = nn.BatchNorm1d(64)
        #self.relu = nn.ReLU(inplace=True)
        self.leaky_relu = nn.LeakyReLU(negative_slope=0.01)  # 使用LeakyReLU
        self.dropout1 = nn.Dropout(0.7)
        #self.res_block = ResidualBlock(64)
        # 添加更多的残差块
        # self.res_blocks = nn.Sequential(
        #     ResidualBlock(64),
        #     ResidualBlock(64),
        #     ResidualBlock(64),
        #     ResidualBlock(64)
        # )
        self.conv2 = nn.Conv1d(64, 128, kernel_size=5, padding=2)
        self.bn2 = nn.BatchNorm1d(128)
       # self.dropout2 = nn.Dropout(0.7)
        self.conv3 = nn.Conv1d(128, 256, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm1d(256)
        #self.dropout3 = nn.Dropout(0.7)  # 添加Dropout层
        self.dropout = nn.Dropout()
        self.pool = nn.AdaptiveMaxPool1d(1)

        self.fc = nn.Linear(256, 2)

    def forward(self, x):
        #x = x.permute(0, 2, 1)  # 调整维度从[N, 202, 2560]到[N, 2560, 202]
        #x = x.unsqueeze(1)
        #x = self.relu(self.conv1(x))
        x = self.leaky_relu(self.conv1(x))  # 使用LeakyReLU
        #x = self.dropout1(x)
        #x = x.unsqueeze(1)  # 增加一个通道维度
        #x = self.relu(self.bn1(self.conv1(x)))
        #x = self.res_blocks(x)
        #x = self.relu(self.bn2(self.conv2(x)))
        x = self.leaky_relu(self.conv2(x))  # 使用LeakyReLU
        #x = self.dropout2(x)
        #x = self.relu(self.bn3(self.conv3(x)))
        x = self.leaky_relu(self.conv3(x))  # 使用LeakyReLU
        x = self.dropout(x)
        x = self.pool(x)
        x = x.view(x.size(0), -1)
        #x = self.dropout(x)  # 应用Dropout
        x = self.fc(x)
        return x


#y_train=y_train.unsqueeze(1)
#y_train1 = y_train # 将标签转换为长整型
#y_test=torch.tensor(y_test1).int()
#y_test=y_test.unsqueeze(1)
#y_test=y_test


# 数据增强创建数据集和数据加载器
#  train_dataset = AugmentedTensorDataset(X_train, y_train)
#  train_loader = DataLoader(dataset=train_dataset, batch_size=64, shuffle=True)
# test_dataset = TensorDataset(X_test1, y_test)
# test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)
# 创建数据集和数据加载器
# train_dataset = TensorDataset(X_train, y_train)
# train_loader = DataLoader(dataset=train_dataset, batch_size=64, shuffle=True)
# test_dataset = TensorDataset(X_test1, y_test)
# test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)


#train_generator = batch_generator(train_data_folder, train_batch_size)
# test_dataset = TensorDataset(X_test1, y_test)
# test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)



# 修改训练模型函数以使用生成器
def train_model_nogen(model, criterion, optimizer, dataloader, num_epochs, early_stopping_patience):
    model.train()
    min_val_loss = float('inf')
    patience_counter = 0
    for epoch in range(num_epochs):
        for inputs, labels in dataloader:
            inputs, labels = inputs.long().cuda(), labels.long().cuda()
            inputs=inputs.float()
            #labels=labels.float()
            optimizer.zero_grad()
            #inputs=inputs.float()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
        val_loss=10
        val_loss = evaluate_model(model, dataloader, criterion)  # 计算验证集损失
        print(f'Epoch {epoch+1}/{num_epochs}, Loss: {loss.item()}, Val Loss: {val_loss}')
        # 早停策略
        if val_loss < min_val_loss:
            min_val_loss = val_loss
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= early_stopping_patience:
                print("Early stopping triggered.")
                break
# 训练模型
def train_model(model, criterion, optimizer, dataloader, num_epochs, early_stopping_patience):
    model.train()
    min_val_loss = float('inf')
    patience_counter = 0
    for epoch in range(num_epochs):
        train_generator = batch_generator(train_data_folder, train_batch_size)
        #train_loader1 = DataLoader(dataset=train_generator, batch_size=train_batch_size, shuffle=True)
        #train_dataset = AugmentedTensorDataset(X_train, y_train)
        #train_dataset = batch_generator(train_data_folder, train_batch_size)
        #train_loader1 = DataLoader(dataset=train_dataset, batch_size=64, shuffle=True)
        for inputs, labels in train_generator:
            # train_dataset = AugmentedTensorDataset(inputs, labels)
            # train_loader1 = DataLoader(dataset=train_dataset, batch_size=64, shuffle=True)
            #for inputs, labels in train_loader1:
              inputs, labels = inputs.cuda(), labels.cuda()
              optimizer.zero_grad()
              outputs = model(inputs)
              loss = criterion(outputs, labels)
              loss.backward()
              optimizer.step()
        # 在每个epoch后更新学习率
        scheduler.step()
        val_loss = evaluate_model(model, dataloader, criterion)  # 计算验证集损失
        #val_loss = evaluate_model(model, dataloader, criterion)  # 计算验证集损失
        print(f'Epoch {epoch+1}/{num_epochs}, Loss: {loss.item()}, Val Loss: {val_loss}')
        # 早停策略
        if val_loss < min_val_loss:
            min_val_loss = val_loss
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= early_stopping_patience:
                print("Early stopping triggered.")

                break

# 评估模型并计算损失
def evaluate_model(model, dataloader, criterion):
    model.eval()
    total_loss = 0
    correct = 0
    total = 0
    with torch.no_grad():
        for inputs, labels in dataloader:
            inputs, labels = inputs.cuda(), labels.long().cuda()
            inputs=inputs.float()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            total_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    accuracy = 100 * correct / total
    print(f'Accuracy of the network on the var-train data: {accuracy}%')
    avg_loss = total_loss / len(dataloader)
    return avg_loss

from sklearn.metrics import precision_score, f1_score, matthews_corrcoef, recall_score


def evaluate_model1(model, dataloader):
    model.eval()
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for inputs, labels in dataloader:
            inputs, labels = inputs.cuda(), labels.long().cuda()  # 将数据移动到GPU
            inputs = inputs.float()
            outputs = model(inputs)
            _, predicted = torch.max(outputs.data, 1)
            all_preds.extend(predicted.tolist())
            all_labels.extend(labels.tolist())

    # 计算准确率
    correct = sum([pred == label for pred, label in zip(all_preds, all_labels)])
    accuracy = 100 * correct / len(all_labels)
    print(f'Accuracy of the network on the test data: {accuracy}%')

    # 计算精确度
    precision = precision_score(all_labels, all_preds)
    print(f'Precision: {precision}')
    # 计算召回率
    recall = recall_score(all_labels, all_preds)
    print(f'recall: {recall}')
    # 计算F1评分
    f1 = f1_score(all_labels, all_preds)
    print(f'F1-score: {f1}')

    # 计算马修相关系数
    mcc = matthews_corrcoef(all_labels, all_preds)
    print(f'MCC: {mcc}')
# 实例化模型
model = RNAConvNet().cuda()

# 损失函数和优化器
criterion = nn.CrossEntropyLoss()
#optimizer = optim.Adam(model.parameters(), lr=0.001)
#optimizer = optim.Adam(model.parameters(), lr=0.0001)  # 降低学习率
#optimizer = optim.Adam(model.parameters(), lr=0.0001, weight_decay=1e-5)
# 使用AdamW优化器
optimizer = optim.AdamW(model.parameters(), lr=0.004, weight_decay=1e-5)

# 设置学习率衰减
# 这里的scheduler将在每10个epoch后将学习率乘以0.1
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)
# 假设我们有一些训练数据
# X_train: 训练数据，维度为[batch_size, embedding_dim]
# y_train: 训练标签，维度为[batch_size]
# 这里只是示例，实际数据需要您提供
#X_train_s = torch.randn(64, 2560)  # 假设的批量大小为64
#y_train1_s = torch.randint(0, 2, (64,))  # 假设的标签
#无位置特征的向量化表示
# posFile='data/nnj_circRNA_newseq.fea'
# negFile='data/nnj_lncRNA_newseq.fea'
#kmer的向量化表示
# posFile='data/tmp1.fea'
# negFile='data/tmp2.fea'
# pos1 = np.loadtxt(posFile, dtype=str)
# neg1 = np.loadtxt(negFile, dtype=str)

# neg1 = load_all_tensors_to_matrix('./data/embedding/nnj/mat/neg/')
# X_pos_train1, X_pos_test1 = train_test_split(pos1, random_state=1, train_size=0.8)
# X_neg_train1, X_neg_test1 = train_test_split(neg1, random_state=1, train_size=0.8)# 创建TensorDataset和DataLoader
# X_train1= np.vstack((X_pos_train1, X_neg_train1))
# X_test1 = np.vstack((X_pos_test1, X_neg_test1))
# y_train1 = np.hstack((np.ones(len(X_pos_train1), int), np.zeros(len(X_neg_train1), int)))
# y_test1 = np.hstack((np.ones(len(X_pos_test1), int), np.zeros(len(X_neg_test1), int)))
# X_train1 = X_train1.astype(float)
# X_test1 = X_test1.astype(float)
# X_train=torch.tensor(X_train1)
# X_test1=torch.tensor(X_test1)
# y_train=torch.tensor(y_train1)
# y_test=torch.tensor(y_test1)
# train_dataset = TensorDataset(X_train, y_train)
# train_loader = DataLoader(dataset=train_dataset, batch_size=64, shuffle=True)
# test_dataset = TensorDataset(X_test1, y_test)
# test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

# 训练和评估
num_epochs = 100
early_stopping_patience = 30 # 早停的耐心值
# 创建训练数据的生成器
train_data_folder = './data/embedding/nnj/mat/train'  # 假设您的数据在这个文件夹中
train_batch_size = 64*2 # 根据您的内存限制调整批次大小
#train_model_nogen(model, criterion, optimizer, train_loader, num_epochs, early_stopping_patience)
x_var,y_var=load_all_tensors_to_matrix('./data/embedding/nnj/mat/test/')
# x_var = x_var.astype(float)
# X_vartrain=torch.tensor(x_var)
# y_vartrain = torch.tensor(y_var)
var_dataset = TensorDataset(x_var, y_var)
var_loader = DataLoader(dataset=var_dataset, batch_size=64, shuffle=True)
train_model(model, criterion, optimizer, var_loader, num_epochs, early_stopping_patience)
#model_path = './data/embedding/nnj/model/model_noembedding.pth'
model_path = './data/embedding/nnj/model/model_cnn1219.pth'
# 保存模型的状态字典
torch.save(model.state_dict(), model_path)
#evaluate_model1(model, test_loader)