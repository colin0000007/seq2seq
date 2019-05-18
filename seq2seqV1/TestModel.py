#encoding=UTF-8
import tensorflow as tf
import numpy as np
import pickle
from utils.dataPreprocessing import source_seq_list_2_ids
dataInfoObj = pickle.load(open("./model.dataInfoObj","rb"))
pad = "<pad>"
input = ["abcd","hello","word"]
batch_real = len(input)
source_batch,seq_len = source_seq_list_2_ids(dataInfoObj,input)
loaded_graph = tf.Graph()
checkpoint = "./seq2seq_model.ckpt"
with tf.Session(graph=loaded_graph) as sess:
    # 加载模型
    loader = tf.train.import_meta_graph(checkpoint + '.meta')
    loader.restore(sess, checkpoint)
    source_batch_input = loaded_graph.get_tensor_by_name('source_batch:0')
    logits = loaded_graph.get_tensor_by_name('predictions2:0')
    input_batch = loaded_graph.get_tensor_by_name("input_batch:0")
    print("input_batch.shape:",input_batch.shape)
    src_seq_len = loaded_graph.get_tensor_by_name('source_batch_seq_len:0')
    tgt_seq_len = loaded_graph.get_tensor_by_name("target_batch_seq_len:0")
    #target_sequence_length = loaded_graph.get_tensor_by_name('target_sequence_length:0')
    answer_logits = sess.run(logits, {source_batch_input: source_batch, 
                                      src_seq_len: seq_len,
                                      input_batch:[batch_real]
                                      })
    print("answer_logits.shape:",answer_logits.shape)
    answer = [[dataInfoObj.target_token_list[index] for index in seq] for seq in answer_logits]
    for i in range(batch_real):
        print(input[i],"  ","".join(answer[i]))