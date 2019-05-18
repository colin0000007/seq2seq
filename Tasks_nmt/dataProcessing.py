#encoding=UTF-8
import pkuseg
seg = pkuseg.pkuseg()

#按照句子长度删除掉一些长句
def filter_by_sen_len():
    source_data_path = "D:\\nlp语料\\机器翻译语料\\chinese.raw.sample.seg.txt"
    target_data_path = "D:\\nlp语料\\机器翻译语料\\english.raw.sample.seg.txt"
    src_encoding = "utf-8"
    tgt_encoding = "gbk"
    src_lines_len = None
    tgt_lines_len = None
    src_lines = None
    tgt_lines = None
    with open(source_data_path,"r",encoding=src_encoding) as f:
        src_lines = f.readlines()
        src_lines_len = [len(seq.split(" ")) for seq in src_lines]
    with open(target_data_path,"r",encoding=tgt_encoding) as f:
        tgt_lines = f.readlines()
        tgt_lines_len = [len(seq.split(" ")) for seq in tgt_lines]
    
    #删除词数量大于100的句子，实际占比只有0.0035
    #但是最长的确到了几千，极大的影响了效率
    max_len = 50
    indices = []
    for i in range(len(src_lines_len)):
        if src_lines_len[i] > max_len:
            indices.append(i)
    print(len(indices))
    print("删除百分比:",len(indices)*1.0 / len(src_lines_len))
    src_lines_new = []
    tgt_lines_new = []
    indices_needed = []
    samples = len(src_lines)
    for i in range(samples):
        if i %5000 == 0:
            print(i,"/",samples)
        if i not in indices:
            src_lines_new.append(src_lines[i])
            tgt_lines_new.append(tgt_lines[i])
    print("len(src_lines_new):",len(src_lines_new))
    print("len(tgt_lines_new):",len(tgt_lines_new))
    f1 = open("D:\\nlp语料\\机器翻译语料\\chinese.raw.sample.seg.ch_max_len="+str(max_len)+".txt","w",encoding=src_encoding)
    f2 = open("D:\\nlp语料\\机器翻译语料\\english.raw.sample.seg.ch_max_len="+str(max_len)+".txt","w",encoding=tgt_encoding)
    for line in src_lines_new:
        f1.write(line)
    for line in tgt_lines_new:
        f2.write(line)
    f1.close()
    f2.close()


#产生指定数量样本的数据
def filter_by_sens_num():
    source_data_path = "D:\\nlp语料\\机器翻译语料\\chinese.raw.sample.seg.ch_max_len=50.txt"
    target_data_path = "D:\\nlp语料\\机器翻译语料\\english.raw.sample.seg.ch_max_len=50.txt"
    src_encoding = "utf-8"
    tgt_encoding = "gbk"
    src_lines = None
    tgt_lines = None
    with open(source_data_path,"r",encoding=src_encoding) as f:
        src_lines = f.readlines()
    with open(target_data_path,"r",encoding=tgt_encoding) as f:
        tgt_lines = f.readlines()
    
    using_lines = 200000
    src_lines = src_lines[0:using_lines]
    tgt_lines = tgt_lines[0:using_lines]
    f1 = open("D:\\nlp语料\\机器翻译语料\\chinese.raw.sample.seg.ch_max_len=50."+"lines="+str(using_lines)+".txt","w",encoding=src_encoding)
    f2 = open("D:\\nlp语料\\机器翻译语料\\english.raw.sample.seg.ch_max_len=50."+"lines="+str(using_lines)+".txt","w",encoding=tgt_encoding)
    for line in src_lines:
        f1.write(line)
    for line in tgt_lines:
        f2.write(line)
    f1.close()
    f2.close()

