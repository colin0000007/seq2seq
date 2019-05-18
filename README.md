# seq2seq

## 1.参考的项目
(1) google官方的tensorflow nmt 
(2) 知乎一篇实现seq2seq的文章，只做了基础参考，我改进了很多地方，代码也规范一些
https://zhuanlan.zhihu.com/p/27608348

## 2.项目结构 
Tasks_chatbot 聊天机器人的例子，没有数据 
Tasks_nmt 机器翻译的例子，没有数据 
data 只包含了测试数据，数据source为字母序列，target为字母序列反转 
modelFile 模型文件存放位置 
referenceCode 参考的知乎那篇文章的代码，可以不用看 
seq2seqV1 改进知乎那篇文章的代码，但是还是有点乱，具体改进什么可以看看我的另一个项目 https://github.com/colin0000007/seq2seq-easy 
seq2seqV2 讲seq2seq模型封装为2个类，BasicSeq2Seq包括了bi-rnn,beam search，AttentionSeq2Seq只是多了attention
utils 数据加载和预处理工具

## 3. 使用
最核心的2个类：
(1) BasicSeq2SeqModel 
(2)AttentionSeqSeqModel 
这2个类对模型进行了封装，具体的使用直接参照 
TestBasicSeq2Seq 
TestAttentionSeq2Seq 

## 4.实现过程中的问题
直接看我的博客，我更新了一些实现过程中的细节问题，踩的坑。 
https://blog.csdn.net/qq_37667364
