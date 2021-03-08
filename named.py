import numpy as np
import pandas as pd
import jieba
from collections import Counter
import re

DATA_PATH = './data/clean_fund_stock.xlsx'

'''
pattern = [['医疗','医药','健康', '中药', '保健'],
           ['制造',],
           ['智能', '科技', '智能', '互联','互联网','信息产业', '高新技术', '计算机行业','电路'],
           ['动力','能源', '环保', '新动力', ],
           ['产业',],
           ['消费',]]
'''

'''
pattern = [
           ['医疗','医药','健康', '中药', '保健'], # 医疗
           ['智能', '科技', '智能', '互联','互联网','信息产业', '高新技术', '计算机行业'], # 计算机
           ['动力','能源', '环保', '新动力', '低碳', '环境'], # 能源环保
            ['制造','电路', '材料', '工业'],
            ['产业', '消费', '主题', '行业'],
            ['成长'],
            ['价值'],
            ['量化']]
'''

pattern = [['成长', '价值'], ['医疗','医药','健康', '中药', '保健','智能', '科技', '智能','互联','互联网','信息产业', '高新技术', '计算机行业',
            '动力','能源', '环保', '新动力', '低碳', '环境', '制造','电路', '材料', '工业', '产业', '消费', '主题', '行业'],
          ['量化']]

def make_pat_dic(pattern_list):
    pat_dic = {}
    for i in range(len(pattern_list)):
        for item in pattern_list[i]:
            pat_dic[item] = i
    return pat_dic

def get_pat_label(string, dic):
    for key, value in dic.items():
        if key in string:
            return value
    return -1

'''
segs = []
for fund in fund_names:
    seg_list = jieba.cut(fund, cut_all=True)
    segs.extend(seg_list) # 全模式   
counter = Counter(segs)
'''

pat_dic = make_pat_dic(pattern)
data = pd.read_excel(DATA_PATH, header=0)
fund_names = [i[0] for i in (data.loc[:, ['基金名称']]).values]
funds = data.loc[:, ['基金名称']]
li = [get_pat_label(fund, pat_dic) for fund in funds['基金名称']]
funds['label'] = li
print(funds['label'].value_counts())
funds.to_csv('./data/funds_labels.csv')


