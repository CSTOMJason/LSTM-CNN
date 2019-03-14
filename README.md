LSTM+Multi_CNN处理文本情感
========================
  参考这篇使用`CNN网络处理文本分类`问题论文(https://arxiv.org/abs/1408.5882) 在卷积的输出结果中添加一个LSTM网络
  ------------
      CNN功能:通过卷积操作，抽取局部特征图，多层卷积能过抽取更高维的特征信息
      LSTM功能：RNN网络的升级版，每层LSTM网络的参数共享，添加了`遗忘门(forget gate)`，`输入门(input gate)`对历史信息进行一个summary
 实验结果导读
 ============
   实验目标
   ------
       使用Google开源深度学习`Tensorflow`框架建立`LSTM+Multi_CNN`对文本情感的分析
   实验准备
   -------
       在`Kaggle`上获取Sentiment.csv的数据集
   数据清洗+训练数据和测试数据划分
   ------
       getdata.py调用input_data函数对数据进行清洗和数据的划分
   实验超参数
   ---------
       lr(学习效率）
       batch（minibatch的系数)
       hidden_units(LSTM网络单元数)
       keep_prob(dropout系数）
       beta（正则化惩罚系数）
       embed_dim(embedding layer的输出维度)
       filters(不同的卷积核的数量)
   模型训练结果
   -----------
   
   
       
       
 
      
  
