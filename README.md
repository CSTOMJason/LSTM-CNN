LSTM+Mui_CNN处理文本情感:
========================
  参考这篇使用`CNN网络处理文本分类`问题论文(https://arxiv.org/abs/1408.5882) 在卷积的输出结果中添加一个LSTM网络\<br>
      #CNN功能:通过卷积操作，抽取局部特征图，多层卷积能过抽取更高维的特征信息]<br>
      #LSTM功能：RNN网络的升级版，每层LSTM网络的参数共享，添加了`遗忘门(forget gate)`，`输入门(input gate)`对历史信息进行一个summary\<br>
      
  
