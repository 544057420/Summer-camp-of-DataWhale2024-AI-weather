# Baseline操作步骤与优化方向

## 1. 定义数据集, 建立起训练数据和标签之间的关系；定义数据加载器(DataLoader)， 方便取数据进行训练

### 优化Dataloader
    
预取因子：`prefetch_factor`参数可以设置为一个大于1的值，以允许DataLoader在当前批次处理时预加载额外的数据批次
    
持久化工作进程：设置`persistent_workers=True`可以使得工作进程在数据加载完毕后不立即销毁，而是等待下一个`epoch`继续使用，这可以减少进程创建和销毁的开销。
    
内存映射：如果数据集太大无法全部加载到内存中，可以考虑使用内存映射技术，即在需要时从磁盘读取数据。
    
减少数据转换的重复计算：在初始化数据集时，对数据进行一次性的预处理，如归一化和转换，然后在后续的迭代中不再重复这些操作。
    
使用pin_memory：当设置`pin_memory=True`时，DataLoader会将数据加载到CUDA的固定内存中，这可以加速数据从CPU到GPU的传输。
    
合理设置batch_size：根据可用的内存和GPU资源，合理设置每个批次的大小，以避免内存不足的问题。
    
使用shuffle和sampler：合理使用`shuffle`参数和自定义的`sampler`可以优化数据的读取顺序，有时可以提高训练效率。
    
避免数据转换中的重复计算：在数据集中预先应用ToTensor和Normalize等转换，避免在每个epoch中重复这些操作。
    
使用drop_last：如果数据集的最后一个批次不完整，使用drop_last=True可以丢弃它，以保持批次大小的一致性。
    
直接使用数据切片：在某些情况下，直接使用数据集的切片可能比使用DataLoader更高效，尤其是在数据已经全部加载到内存中的情况下

## 2. 定义模型, 利用PyTorch搭建网络，根据输入输出数据维度实例化模型

### 加深网络深度：原baseline只有一个卷积层（conv）对于特征提取和学习的能力不足

LSTM类<https://easyai.tech/ai-definition/lstm/>

Transformer与Attention类<https://easyai.tech/ai-definition/attention/>

CNN类<https://easyai.tech/blog/illustrated-10-cnn-architectures/><https://easyai.tech/ai-definition/rnn/>

门控循环单元（GRU - Gated Recurrent Unit）<>

个人不推荐GNN类模型，其训练算力消耗与可解释性方面存在缺陷。但有超越Mamba架构的[Test-Time-Training layers，TTT模型]<https://mp.weixin.qq.com/s/Z8BVt7g6rnuAFzoca1fjfg>经过减枝后可能有更好表现。

### 数据压缩感知：
  <https://github.com/datawhalechina/awesome-compression>
  模型剪枝
  
  模型量化
  
  神经网络架构搜索
  
  知识蒸馏

## 3. 定义损失函数, 优化器, 训练周期, 训练模型并保存模型参数
  
  *许多部署场景下Adam最优*

### 常用的损失函数：

1. **均方误差（Mean Squared Error, MSE）**
   - 用于回归问题，计算预测值与真实值之间差的平方的平均值。

2. **交叉熵损失（Cross-Entropy Loss）**
   - 用于分类问题，特别是多分类问题，衡量的是模型输出的概率分布与真实标签的概率分布之间的差异。

3. **二元交叉熵损失（Binary Cross-Entropy Loss）**
   - 用于二分类问题，计算模型输出为1的概率与实际标签为1的概率之间的差异。

4. **Hinge损失（Hinge Loss）**
   - 用于SVM（支持向量机）分类问题，衡量的是分类间隔的大小。

5. **Categorical Cross-Entropy Loss**
   - 用于多分类问题，是交叉熵损失的一种，适用于标签为one-hot编码的情况。

6. **Softmax损失**
   - 通常与Categorical Cross-Entropy Loss结合使用，用于多分类问题，将模型输出归一化为概率分布。

7. **Dice损失（Dice Loss）**
   - 常用于医学图像分割，衡量的是预测的分割图与真实分割图之间的重叠度。

8. **IoU损失（Intersection over Union Loss）**
   - 用于目标检测和图像分割，衡量的是预测的边界框与真实边界框之间的重叠度。

9. **Focal Loss**
   - 用于解决类别不平衡问题，通过减少易分类样本的权重来增加模型对难分类样本的关注。

10. **Triplet损失**
    - 用于度量学习，确保一个样本与其对应的锚点更接近，而与负样本更远离。

### 常用的优化器：

1. **SGD（随机梯度下降，Stochastic Gradient Descent）**
   - 最基础的优化算法，通过随机选择样本来更新权重。

2. **Momentum**
   - 在SGD的基础上增加了动量项，有助于加速梯度下降过程并减少震荡。

3. **Nesterov Accelerated Gradient（NAG）**
   - 对Momentum的改进，考虑了动量项的梯度。

4. **Adam（Adaptive Moment Estimation）**
   - 结合了Momentum和RMSprop的特点，自适应调整每个参数的学习率。

5. **RMSprop（Root Mean Square Propagation）**
   - 对SGD的改进，通过平方根的累积平均来调整学习率。

6. **Adagrad（Adaptive Gradient Algorithm）**
   - 对每个参数自适应地调整学习率，对于稀疏数据特别有效。

7. **Adadelta**
   - Adagrad的改进版本，解决了学习率随着时间降低的问题。

8. **Adamax**
   - Adam的一个变种，使用无限范数来提高数值稳定性。

9. **Nadam（Nesterov-accelerated Adaptive Moment Estimation）**
   - 结合了NAG和Adam的特点，适用于处理大规模数据集。

10. **AMSGrad**
    - Adam的变种，解决了Adam在非平稳目标问题上的收敛性问题。

## 4. 模型加载及推理(模型预测)，输入测试数据输出要提交的文件

* 思考题：  

** 观察数据的组成结合时间序列分析, 你认为本次赛题数据中有哪些值得改进的地方？  

含有缺失数据，需要使用插值或使用模型预测缺失值，但是存在误差积累的风险。


** 本次赛题你认为是否是时间序列预测问题, 并给出相应的理由  
任务要求预测未来72小时的逐小时气象要素，特别是累积降水量，这明显是一个时间序列预测任务。  
数据包括历史时段的多个气象要素，这些要素随时间变化，并且对未来的降水量有影响，符合时间序列分析的需求。
