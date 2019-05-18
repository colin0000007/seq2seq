#encoding:UTF-8
import numpy as np
'''
@author: outsider
生成一个用于seq2seq模型的数据集
source:一串字母
target:一串字母的倒序
'''

#序列最大长度
max_len = 10
min_len = 2
num_samples = 10000
letters = np.array([chr(letter) for letter in range(ord('a'),ord('z')+1)])
indices = [i for i in range(26)]

data_x = []
data_y = []
for i in range(num_samples):
    s_len = np.random.randint(min_len,max_len)
    #随机选取s_len个index
    indices_s = np.random.choice(indices,size=s_len,replace=False)
    sample = "".join(letters[indices_s])
    data_x.append(sample)
    data_y.append(sample[::-1])#反转字符串
with open("./data/letters_source2.txt","w",encoding='utf-8') as f:
    f.write("\n".join(data_x))
with open("./data/letters_target2.txt","w",encoding='utf-8') as f:
    f.write("\n".join(data_y))
