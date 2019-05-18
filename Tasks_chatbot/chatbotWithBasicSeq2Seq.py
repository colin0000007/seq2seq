#encoding=UTF-8
import tensorflow as tf
from seq2seqV2.BasicSeq2seqModel import BasicSeq2SeqModel
#使用beam search必须添加这个导入
from tensorflow.contrib.seq2seq.python.ops import beam_search_ops
from utils.dataPreprocessing import source_seq_list_2_ids
from utils.dataPreprocessing import load_data
import pickle
import numpy as np
import pkuseg
'''
中文对话
'''
#模型训练
def train():
    source_data_path = "D:\\nlp语料\\各类中文对话语料\\qingyun-11w\\sources.txt"
    target_data_path = "D:\\nlp语料\\各类中文对话语料\\qingyun-11w\\targets.txt"
    model_path = "../modelFile/chatbot/model_basic/model.ckpt"
    src_encoding = "utf-8"
    tgt_encoding = "utf-8"
    source_split_char = " "
    target_split_char = " "
    smwf = 2 #source 最小词频
    tmwf = 2 #target最小词频
    batch_size = 50
    epochs = 40
    dataInfoObj, gen = load_data(source_data_path,target_data_path,source_split_char,target_split_char,source_minimum_word_frequency=smwf,target_minimum_word_frequency=tmwf,batch_size=batch_size,epochs=epochs,source_encoding=src_encoding,target_encoding=tgt_encoding)
    #保存数据集的一些信息
    f = open("../modelFile/chatbot/model_basic/model.dataInfoObj","wb")
    pickle.dump(dataInfoObj,f)
    f.close()
    #超参数开始
    src_embedding_size = 200
    tgt_embedding_size = 200
    '''
      encoder是否双向
                注意:使用bidirectional，encoder rnn的num_units变为decoder的一半，这是为了能够保证encoder_states和decoder的输入shape能对应上
    '''
    is_encoder_bidirectional = True
    rnn_layer_size = 4
    rnn_num_units = 256
    cell_type = "LSTM"
    lr = 0.001
    decoding_method = "beamSearch"
    #训练
    model = BasicSeq2SeqModel(src_vocab_size=dataInfoObj.source_vocab_size,tgt_time_step=dataInfoObj.target_max_len,tgt_vocab_size=dataInfoObj.target_vocab_size,start_token_id=dataInfoObj.target_token_2_id['<s>'],end_toekn_id=dataInfoObj.target_token_2_id['</s>'])
    model.train(model_path, gen, src_embedding_size, tgt_embedding_size, is_encoder_bidirectional,rnn_layer_size, rnn_num_units, cell_type, lr,decoding_method=decoding_method,beam_width=10)
    
#模型测试
def test():
    dataInfoObj = pickle.load(open("../modelFile/chatbot/model_basic/model.dataInfoObj","rb"))
    model_path = "../modelFile/chatbot/model_basic/model.ckpt"
    model = BasicSeq2SeqModel(model_path=model_path)
    #预测
    input = None
    seg = pkuseg.pkuseg()
    
    while input != 'over':
        input = input("human:")
        #需要对输入分词
        source_batch,seq_len = source_seq_list_2_ids(dataInfoObj,[seg.cut(input)])
        answer_logits = model.predict(source_batch, seq_len)
        end_token_id = dataInfoObj.target_token_2_id['</s>']
        answer = [[dataInfoObj.target_token_list[index] for index in seq if index != end_token_id] for seq in answer_logits]
        print("robot:","".join(answer[i]))
if __name__=="__main__":
    train()
    #test()
