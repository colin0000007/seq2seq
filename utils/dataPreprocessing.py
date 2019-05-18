#encoding=UTF-8
import random
import numpy as np
from collections import Counter
pad = "<pad>"
start_token = "<s>"
end_toekn = "</s>"
unknown_token = "<unknown>"
'''
保存一个数据集的一些属性，包括：
source_max_len
source_token_list
source_token_2_id

target_max_len
target_token_list
target_token_2_id

'''
class DataInfo():
    def __init__(self,source_max_len,source_token_list,source_token_2_id,target_max_len,target_token_list,target_token_2_id):
        self.source_max_len = source_max_len
        self.source_token_list = source_token_list
        self.source_vocab_size = len(source_token_list)
        self.source_token_2_id = source_token_2_id
        self.target_max_len = target_max_len
        self.target_token_list = target_token_list
        self.target_vocab_size = len(target_token_list)
        self.target_token_2_id = target_token_2_id
    
    def set_num_samples(self,num_samples):
        self.num_samples = num_samples
        

'''
数据预处理，生成序列的最大长度，序列词典等操作
'''
def preprocessing(source_seq_list,target_seq_list,source_minimum_word_frequency=1,target_minimum_word_frequency=1):
    source_max_len = max([len(seq) for seq in source_seq_list])
    target_max_len = max([len(seq) for seq in target_seq_list])
    print("source_samples:",len(source_seq_list))
    print("target_samples:",len(target_seq_list))
    for seq in source_seq_list:
        if len(seq) == 3636:
            print(seq)
            break
    source_tokens = [token for seq in source_seq_list for token in seq]
    target_tokens = [token for seq in target_seq_list for token in seq]
    source_counter = Counter(source_tokens)
    target_counter = Counter(target_tokens)
    source_token_list = [k for k,w in source_counter.items() if w >= source_minimum_word_frequency]
    target_token_list = [k for k,w in target_counter.items() if w >= target_minimum_word_frequency]
    source_token_list.insert(0, pad)
    source_token_list.insert(1,unknown_token)
    target_token_list.insert(0, pad)
    target_token_list.insert(1,start_token)
    target_token_list.insert(2,end_toekn)
    target_token_list.insert(3,unknown_token)
    source_token_2_id = dict(zip(source_token_list,[i for i in range(len(source_token_list))]))
    target_token_2_id = dict(zip(target_token_list,[i for i in range(len(target_token_list))]))
    #加上1是因为target_batch_x首需要<s>
    #target_batch_y末需要</s>
    target_max_len = target_max_len + 1
    print("data preprocessing:")
    print("source_seq_max_len:",source_max_len,"\nsource_vocab_size:",len(source_token_list),"\nsource_dict:",source_token_list[0:int(0.01*len(source_token_list))])
    print()
    print("target_seq_max_len:",target_max_len,"\ntarget_vocab_size:",len(target_token_list),"\ntarget_dict:",target_token_list[0:int(0.01*len(target_token_list))])
    dataInfoObj = DataInfo(source_max_len, source_token_list, source_token_2_id, target_max_len, target_token_list, target_token_2_id)
    dataInfoObj.set_num_samples(len(source_seq_list))
    return dataInfoObj

#将seq列表转换位int id
def source_seq_list_2_ids(dataInfoObj,source_seq_list):
    #转换为id
    source_seq_int = []
    for seq in source_seq_list:
        seq_new = []
        for token in seq:
            if token in dataInfoObj.source_token_list:
                seq_new.append(dataInfoObj.source_token_2_id[token])
            else:
                seq_new.append(dataInfoObj.source_token_2_id[unknown_token])
        source_seq_int.append(seq_new)
    #保存source序列的真实长度
    source_seq_len_real = []
    #不足max的补充<pad>
    for seq in source_seq_int:
        source_seq_len_real.append(len(seq))
        for i in range(dataInfoObj.source_max_len - len(seq)):
            seq.append(dataInfoObj.source_token_2_id[pad])
    return source_seq_int,source_seq_len_real

def target_ids_2_seq():
    return

def target_seq_list_2_ids(dataInfoObj,target_seq_list):
    #转换为id
    target_seq_int = []
    for seq in target_seq_list:
        seq_new = []
        for token in seq:
            if token in dataInfoObj.target_token_list:
                seq_new.append(dataInfoObj.target_token_2_id[token])
            else:
                seq_new.append(dataInfoObj.target_token_2_id[unknown_token])
        target_seq_int.append(seq_new)
    #真实的序列长度
    target_seq_len_real = []
    for seq in target_seq_int:
        #首加上<s>剩余位置补充<pad>
        seq.insert(0, dataInfoObj.target_token_2_id[start_token])
        target_seq_len_real.append(len(seq))
        for i in range(dataInfoObj.target_max_len - len(seq)):
            seq.append(dataInfoObj.target_token_2_id[pad]) 
    return target_seq_int,target_seq_len_real
#source_max_len,source_dic_list,source_token_2_id,target_max_len,target_dic_list,target_token_2_id,source_seq_list, target_seq_list,
def batch_generator(source_seq_list,target_seq_list,dataInfoObj,batch_size=128,epochs=20):
    #将seq中的字符变成int，并补充<pad> <s> </s>
    source_seq_int,source_seq_len_real = source_seq_list_2_ids(dataInfoObj,source_seq_list)
    target_seq_int,target_seq_len_real = target_seq_list_2_ids(dataInfoObj,target_seq_list)
    num_sample = len(source_seq_int)
    num_batch = num_sample // batch_size
    print("num_samples:",num_sample)
    print("batch_size:",batch_size)
    print("num_batch:",num_batch)
    print("epochs:",epochs)
    indices = [i for i in range(num_sample)]
    source_seq_int = np.array(source_seq_int)
    target_seq_int = np.array(target_seq_int)
    for i in range(epochs):
        #对样本索引打乱
        random.shuffle(indices)
        for j in range(num_batch):
            source_batch = [source_seq_int[index] for index in indices[j*batch_size:(j+1)*batch_size]]
            source_batch_seq_len = [source_seq_len_real[index] for index in indices[j*batch_size:(j+1)*batch_size]]
            source_batch_max_len = np.max(source_batch_seq_len)
            #丢掉source_batch不必要的部分
            source_batch = np.array(source_batch)[:,0:source_batch_max_len]
            target_batch_seq_len = [target_seq_len_real[index] for index in indices[j*batch_size:(j+1)*batch_size]]
            '''
                    x:<s> A B C D  <pad> <pad>
                    y:A  B C D </s> <pad> <pad>
            '''
            target_batch_x = [target_seq_int[index] for index in indices[j*batch_size:(j+1)*batch_size]]
            #将target_batch_x向前移动一格，末尾补充pad，也就是用上一个字预测下一个字
            target_batch_y = [[target_seq_int[index,k] for k in range(1,target_seq_len_real[index])] for index in indices[j*batch_size:(j+1)*batch_size]]
            for seq in target_batch_y:
                #每个seq后面添加</s>
                seq.append(dataInfoObj.target_token_2_id[end_toekn])
                #剩余位置添加<pad>
                ty_len = len(seq)
                for _ in range(dataInfoObj.target_max_len-ty_len):
                    seq.append(dataInfoObj.target_token_2_id[pad])
            target_batch_max_len = np.max(target_batch_seq_len)
            #同样丢掉target中不必要的部分
            target_batch_x = np.array(target_batch_x)[:,0:target_batch_max_len]
            target_batch_y = np.array(target_batch_y)[:,0:target_batch_max_len]
            yield source_batch,target_batch_x,target_batch_y,target_batch_max_len,source_batch_seq_len,target_batch_seq_len,str(j+1)+'/'+str(num_batch),str(i+1)+'/'+str(epochs)

#将source文件和target文件加载进来，存储到2个list中
#只需要重写make_source_target_list这个方法就可以加载其他数据
def make_source_target_list(source_path,target_path,source_split_char=None,target_split_char=None,source_encoding="gbk",target_encoding="gbk"):
    source_lines = None
    target_lines = None
    with open(source_path,"r",encoding=source_encoding) as f:
        source_lines = f.readlines()
    with open(target_path,"r",encoding=target_encoding) as f:
        target_lines = f.readlines()
    return make_list(source_lines, source_split_char),make_list(target_lines, target_split_char)

def make_list(lines,split_char=None):
    lines = [line.strip().lower() for line in lines]
    seq_list = []
    if split_char == None:
        seq_list = [[token for token in seq] for seq in lines]
    else:
        seq_list = [[token for token in seq.split(split_char)] for seq in lines]
    return seq_list

def load_data(source_data_path,target_data_path,source_split_char,target_split_char,source_minimum_word_frequency=1,target_minimum_word_frequency=1,batch_size=128,epochs=20,source_encoding="gbk",target_encoding="gbk"):
    source_seq_list, target_seq_list = make_source_target_list(source_data_path,target_data_path,source_split_char,target_split_char,source_encoding,target_encoding)
    dataInfoObj = preprocessing(source_seq_list, target_seq_list,source_minimum_word_frequency,target_minimum_word_frequency)
    gen = batch_generator(source_seq_list,target_seq_list,dataInfoObj,batch_size,epochs)
    return dataInfoObj,gen

'''
source_data_path = "../data/letters_source2.txt"
target_data_path = "../data/letters_target2.txt"
dataInfoObj,gen = load_data(source_data_path, target_data_path, None, None, source_minimum_word_frequency=1, target_minimum_word_frequency=1)
for source_batch,target_batch_x,target_batch_y,source_batch_seq_len,target_batch_seq_len,j,i in gen:
    print("source_batch:",source_batch)
    print("target_batch_x:",target_batch_x)
    print("target_batch_y:",target_batch_y)
    print("source_batch_seq_len:",source_batch_seq_len)
    print("target_batch_seq_len:",target_batch_seq_len)
    break
'''
