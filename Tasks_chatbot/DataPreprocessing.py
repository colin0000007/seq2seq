#encoding=UTF-8
import pkuseg
'''
seg = pkuseg.pkuseg()           # 以默认配置加载模型
text = seg.cut("what's wrong with you?")  # 进行分词
print(text)
'''
'''
处理青云对话数据，分开为source和target，分词使用pkuseg
'''
path  = "D:\\nlp语料\\各类中文对话语料\\qingyun-11w\\12万对话语料青云库.csv"
'''
格式说明
每一行为问题和回答
问题和回答之间使用' | '分割
'''
srcLines = []
tgtLines = []
split_char = ' '#词语间隔符号
seg = pkuseg.pkuseg()
src_tgt_dic = set()
src_dic = set()
tgt_dic = set()
save_path = "D:\\nlp语料\\各类中文对话语料\\qingyun-11w\\"
with open(path,"r",encoding='utf-8') as f:
    lines = f.readlines()
    #分词会自动去掉换行等符号
    for line in lines:
        src_tgt = line.split(' | ')
        src_segd = seg.cut(src_tgt[0])
        tgt_segd = seg.cut(src_tgt[1])
        for w in src_segd:
            src_tgt_dic.add(w)
            src_dic.add(w)
        for w in tgt_segd:
            src_tgt_dic.add(w)
            tgt_dic.add(w)
        srcLines.append(split_char.join(src_segd))
        tgtLines.append(split_char.join(tgt_segd))
    '''
    with open(save_path+"sources.txt","w",encoding="utf-8") as f2:
        for line in srcLines:
            f2.write(line+"\n")
    with open(save_path+"targets.txt","w",encoding="utf-8") as f3:
        for line in tgtLines:
            f3.write(line+"\n")
    '''
    print("样本数量:",len(lines))
    print("source和target共词库数量:",len(src_tgt_dic))
    print("source词库数量:",len(src_dic))
    print("target词库数量:",len(tgt_dic))
    print("source和target词库重复数量:",len(src_dic)+len(tgt_dic)-len(src_tgt_dic))
    print("重复率:",len(src_tgt_dic)*1.0/(len(src_dic)+len(tgt_dic)))
    print("词库词语示例:")
    size = 100
    i = 0
    for w in src_tgt_dic:
        print(w,end="/")
        i = i + 1
        if i == size:
            break