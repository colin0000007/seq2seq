#encoding=UTF-8
import tensorflow as tf
from seq2seqV2.BasicSeq2seqModel import BasicSeq2SeqModel
#使用beam search必须添加这个导入
from tensorflow.contrib.seq2seq.python.ops import beam_search_ops
from utils.dataPreprocessing import source_seq_list_2_ids
from utils.dataPreprocessing import load_data
import pickle
import numpy as np
'''
神经机器翻译
'''
#模型训练
def train():
    source_data_path = "D:\\nlp语料\\机器翻译语料\\chinese.raw.sample.seg.ch_max_len=50.lines=200000.txt"
    target_data_path = "D:\\nlp语料\\机器翻译语料\\english.raw.sample.seg.ch_max_len=50.lines=200000.txt"
    src_encoding = "utf-8"
    tgt_encoding = "gbk"
    source_split_char = " "
    target_split_char = " "
    smwf = 2 #source 最小词频
    tmwf = 2 #target最小词频
    batch_size = 50
    epochs = 40
    dataInfoObj, gen = load_data(source_data_path,target_data_path,source_split_char,target_split_char,source_minimum_word_frequency=smwf,target_minimum_word_frequency=tmwf,batch_size=batch_size,epochs=epochs,source_encoding=src_encoding,target_encoding=tgt_encoding)
    #保存数据集的一些信息
    f = open("../modelFile/nmt/model2/model.dataInfoObj","wb")
    pickle.dump(dataInfoObj,f)
    f.close()
    #超参数开始
    src_embedding_size = 256
    tgt_embedding_size = 256
    '''
      encoder是否双向
                注意:使用bidirectional，encoder rnn的num_units变为decoder的一半，这是为了能够保证encoder_states和decoder的输入shape能对应上
    '''
    is_encoder_bidirectional = True
    rnn_layer_size = 6
    rnn_num_units = 512
    cell_type = "LSTM"
    lr = 0.5
    decoding_method = "beamSearch"
    #训练
    model = BasicSeq2SeqModel(src_vocab_size=dataInfoObj.source_vocab_size,tgt_time_step=dataInfoObj.target_max_len,tgt_vocab_size=dataInfoObj.target_vocab_size,start_token_id=dataInfoObj.target_token_2_id['<s>'],end_toekn_id=dataInfoObj.target_token_2_id['</s>'])
    model.train("../modelFile/nmt/model2/model.ckpt", gen, src_embedding_size, tgt_embedding_size, is_encoder_bidirectional,rnn_layer_size, rnn_num_units, cell_type, lr,decoding_method=decoding_method,beam_width=10)
    
#模型测试
def test():
    dataInfoObj = pickle.load(open("../modelFile/nmt/model.dataInfoObj","rb"))
    model_path = "../modelFile/nmt/model.ckpt"
    model = BasicSeq2SeqModel(model_path=model_path)
    #预测
    input = load_test_data()
    source_batch,seq_len = source_seq_list_2_ids(dataInfoObj,input)
    answer_logits = model.predict(source_batch, seq_len)
    print("answer_logits:",answer_logits.shape)
    end_token_id = dataInfoObj.target_token_2_id['</s>']
    answer = [[dataInfoObj.target_token_list[index] for index in seq if index != end_token_id] for seq in answer_logits]
    for i in range(len(input)):
        print("".join(input[i]),"  "," ".join(answer[i]))

def load_test_data():
    f = open("./test.txt","r",encoding="utf-8")
    lines = f.readlines()
    sens = [line.strip().split(" ") for line in lines]
    f.close()
    return sens
if __name__=="__main__":
    #train()
    test()
