#encoding=UTF-8
import tensorflow as tf
from seq2seqV2.AttentionSeq2SeqModel import AttentionSeq2SeqModel
#使用beam search必须添加这个导入
from tensorflow.contrib.seq2seq.python.ops import beam_search_ops
from utils.dataPreprocessing import source_seq_list_2_ids
from utils.dataPreprocessing import load_data
import pickle

'''
attention with beam search 

'''

#模型训练
def train():
    source_data_path = "../data/letters_source2.txt"
    target_data_path = "../data/letters_target2.txt"
    #model_path = "../modelFile/testAttentionSeq2Seq/model_beam_search.ckpt"
    model_path = "../modelFile/testAttentionSeq2Seq/model_greedy.ckpt"
    batch_size = 128
    epochs = 20
    dataInfoObj, gen = load_data(source_data_path,target_data_path,None,None,source_minimum_word_frequency=1,target_minimum_word_frequency=1,batch_size=batch_size,epochs=epochs)
    #保存数据集的一些信息
    f = open("../modelFile/testAttentionSeq2Seq/model.dataInfoObj","wb")
    pickle.dump(dataInfoObj,f)
    f.close()
    #超参数开始
    src_embedding_size = 15
    tgt_embedding_size = 15
    '''
      encoder是否双向
                注意:使用bidirectional，encoder rnn的num_units变为decoder的一半，这是为了能够保证encoder_states和decoder的输入shape能对应上
    '''
    is_encoder_bidirectional = True
    rnn_layer_size = 2
    rnn_num_units = 128
    cell_type = "LSTM"
    lr = 0.001
    decoding_method = "greedy"
    attention_mechanism = "scaled_luong"
    #训练
    model = AttentionSeq2SeqModel(src_vocab_size=dataInfoObj.source_vocab_size,tgt_time_step=dataInfoObj.target_max_len,tgt_vocab_size=dataInfoObj.target_vocab_size,start_token_id=dataInfoObj.target_token_2_id['<s>'],end_toekn_id=dataInfoObj.target_token_2_id['</s>'],attention_mechanism=attention_mechanism,batch_size=batch_size)
    model.train(model_path, gen, src_embedding_size, tgt_embedding_size, is_encoder_bidirectional,rnn_layer_size, rnn_num_units, cell_type, lr,decoding_method=decoding_method,beam_width=10)
    
#模型测试
def test():
    dataInfoObj = pickle.load(open("../modelFile/testAttentionSeq2Seq/model.dataInfoObj","rb"))
    #model_path = "../modelFile/testAttentionSeq2Seq/model_greedy.ckpt"
    model_path = "../modelFile/testAttentionSeq2Seq/model_beam_search.ckpt"
    model = AttentionSeq2SeqModel(model_path=model_path)
    #预测
    input = ["abcd","hello","word","kzznhel","trswatm"]
    source_batch,seq_len = source_seq_list_2_ids(dataInfoObj,input)
    prediction_counts = 100
    right_sum = 0
    for c in range(prediction_counts):
        answer_logits = model.predict(source_batch, seq_len)
        print("answer_logits:",answer_logits.shape)
        answer = [[dataInfoObj.target_token_list[index] for index in seq if dataInfoObj.target_token_list[index] != '</s>'] for seq in answer_logits]
        for i in range(len(input)):
            print(input[i],"  ","".join(answer[i]))
            if input[i][::-1] == "".join(answer[i]):
                right_sum = right_sum + 1
    print("acc:",right_sum * 1.0 / (len(input)*prediction_counts))

if __name__=="__main__":
    #如果训练参数除了解码方法其他都一样，那么不同的推理图可以使用同一个训练参数
    #加载模型时可以指定训练图所在位置以及推理图所在位置
    #train()
    test()#tensorboard --logdir logs指令
