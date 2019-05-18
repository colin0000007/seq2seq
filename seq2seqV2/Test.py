#encoding=UTF-8
import tensorflow as tf


def fun(par):
    if par != 'greedy':
        raise Exception('the value of par is greedy only!')
    print("正常")

fun('aa')