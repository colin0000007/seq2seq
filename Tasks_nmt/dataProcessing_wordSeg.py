#encoding=UTF-8
import nltk
import re

#英文分词
def eng_seg():
    #基于标点符号的分词
    from nltk.tokenize import WordPunctTokenizer 
    tokenizer = WordPunctTokenizer()
    print(tokenizer.tokenize("don't do that!"))
    path = "D:\\nlp语料\\机器翻译语料\\english.raw.sample.txt"
    f = open(path,"r")
    '''
    text = f.read()
    splChars = set()
    for ch in text:
        if (ch >= 'a' and ch <= 'z') or  (ch >= 'A' and ch <= 'Z'):
            pass
        else:
            splChars.add(ch)
    
    print(splChars)
    '''
    
    lines = f.readlines()
    print(len(lines))
    line_tokenized = []
    split_char = " "
    for line in lines:
        line_tokenized.append(split_char.join(tokenizer.tokenize(line)))
    f2 = open("D:\\nlp语料\\机器翻译语料\\english.raw.sample.seg.txt","w")
    for line in line_tokenized:
        f2.write(line+"\n")
    f.close()
    f2.close()

#中文分词
def chinese_seg():
    path = 'D:\\nlp语料\\机器翻译语料\\chinese.raw.sample.seg'
    pass