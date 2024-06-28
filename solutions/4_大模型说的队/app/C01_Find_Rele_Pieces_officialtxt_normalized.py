import json
import csv
import pandas as pd
import copy 
n = 30
import re
from collections import Counter
import math

# 根据问题的token和对于公司介绍的CSV文件中每一行的token，计算相似度，找到与问题最相似的片段

pattern1 = r'截至'
pattern2 = r'\d{1,4}年\d{1,2}月\d{1,2}日'


q_file_dir = '/app/intermediate/A02_question_classify_entity.csv' # '问题id', '问题', '分类', '对应实体', 'csv文件名'
q_file =  pd.read_csv(q_file_dir,delimiter = ",",header = 0)

normalized_dir = '/app/data/AD_normalized_ot.csv'

normalized_file = pd.read_csv(normalized_dir,delimiter = ",",header = 0)
n_list_1 = list(normalized_file['文件名'])
n_list_2 = list(normalized_file['normalized'])

pdf_csv_file_dir = '/app/data/txt2csv_normalized'

from modelscope import AutoModelForCausalLM, AutoTokenizer, snapshot_download
from modelscope import GenerationConfig

model_dir = '/tcdata/models/Tongyi-Finance-14B-Chat'

tokenizer = AutoTokenizer.from_pretrained(model_dir, trust_remote_code=True)

# 计算两个计数向量的加权余弦相似度。计算问题token和公司介绍token中每一行的余弦相似度
# c1中出现的次数乘以c2中出现的次数，除以normalized_dict中出现的次数，在求和
# 计算c1和c2的加权模
def counter_cosine_similarity(c1, c2, normalized_dict): #使用截断的ccs
    terms = set(c1).union(c2) # 取并集
    dotprod = sum(c1.get(k, 0) * c2.get(k, 0)/normalized_dict.get(k,1) for k in terms)
    magA = math.sqrt(sum(c1.get(k, 0)**2/(normalized_dict.get(k,1)**2) for k in terms))
    magB = math.sqrt(sum(c2.get(k, 0)**2/(normalized_dict.get(k,1)**2) for k in terms))
    
    if magA * magB != 0:
        return dotprod / (magA * magB)
    else:
        return 0
    
g = open('/app/intermediate/AB01_question_with_related_text_ot_normalized.csv', 'w', newline='', encoding = 'utf-8-sig') 
csvwriter = csv.writer(g)
csvwriter.writerow(['问题id','问题','对应实体','csv文件名','top_n_pages_index','top_n_pages_similarity','top_n_pages']) # top_n_pages对应的相似片段

stopword_list = ['根据','招股意见书','招股意向书','报告期内','截至','千元','万元','哪里','哪些','哪个','分别','知道',"什么",'是否','分别','多少','为','?','是','和',
'的','我','想','元','。','？','，','怎样','谁','以及','了','在','哪','对']
bd_list = [',','.','?','。','，','[',']']
print('C01_Started')
for cyc in range(1000):
    temp_q = q_file[cyc:cyc+1]['问题'][cyc]
    
    temp_e = q_file[cyc:cyc+1]['对应实体'][cyc] # 相似度最大的公司名
    if temp_e == 'N_A': # 为sql任务的会为空
        csvwriter.writerow([q_file[cyc:cyc+1]['问题id'][cyc],
                            q_file[cyc:cyc+1]['问题'][cyc],
                            'N_A','N_A','N_A','N_A'])
        continue
    else:
        temp_csv_dir = pdf_csv_file_dir +'/' + q_file[cyc:cyc+1]['csv文件名'][cyc]
        company_csv = pd.read_csv(temp_csv_dir,delimiter = ",",header = 0) # 读取公司的CSV文件
        temp_hash = q_file[cyc:cyc+1]['csv文件名'][cyc][0:-8]+'.txt' # 得到公司SCV文件名的哈希值，后8位截取的是后缀.PDF.SCV
        
        normalized_id = n_list_1.index(temp_hash)
        normalized_dict = eval(n_list_2[normalized_id]) # 得到normalized字典
        company_csv = pd.read_csv(temp_csv_dir,delimiter = ",",header = 0)
        temp_q = temp_q.replace(' ','')

            
            
        #停用词？
        temp_q = temp_q.replace(temp_e,' ') # 把问题中的公司名替换掉
        for word in stopword_list:
            temp_q = temp_q.replace(word,' ') # 问题中的停顿词替换掉
        temp_q_list = temp_q.split()
        temp_q_tokens = list()
        for word in temp_q_list:
            temp_q_tokens_add = tokenizer(word)
            temp_q_tokens_add = temp_q_tokens_add['input_ids']
            for word_add in temp_q_tokens_add:
                temp_q_tokens.append(word_add)

        C_temp_q_tokens = Counter(temp_q_tokens)
        list_sim = list()
        for cyc2 in range(len(company_csv)): # 遍历公司介绍SCV文件中的每一行
            temp_sim = 0
            temp_file_piece = ''
            if company_csv[cyc2:cyc2+1]['纯文本'][cyc2] == company_csv[cyc2:cyc2+1]['纯文本'][cyc2]:
                temp_file_piece = company_csv[cyc2:cyc2+1]['纯文本'][cyc2]
            
            for bd in bd_list:
                temp_file_piece = temp_file_piece.replace(bd,' ') # 替换标点符号
                
            temp_s_tokens = tokenizer(temp_file_piece)
            temp_s_tokens = temp_s_tokens['input_ids']
            
            C_temp_s_tokens = Counter(temp_s_tokens)
            C_temp_s_tokens['220'] = 0 # 特殊字符设置为0
            
            
            if temp_q_tokens == '':
                temp_sim = 0
            else: # 计算问题token的计数器和公司介绍token中每一行的计数器的余弦相似度
                temp_sim = counter_cosine_similarity(C_temp_q_tokens,C_temp_s_tokens,normalized_dict)
            list_sim.append(temp_sim)
            
        #找到相似度最大的
        t = copy.deepcopy(list_sim) 
        max_number = []
        max_index = []
        
        for _ in range(n): # 取相似度最大的n个，n=30
            number = max(t)
            index = t.index(number)
            t[index] = 0
            max_number.append(number)
            max_index.append(index)
        t = []

        
        #将对应的index片段放入文件
        temp_file_pieces_list = list()
        for index in max_index:
            temp_dict = {}
            if company_csv[index:index+1]['纯文本'][index] == company_csv[index:index+1]['纯文本'][index]:
                temp_dict['text'] = company_csv[index:index+1]['纯文本'][index]
            
            temp_file_pieces_list.append(temp_dict)


        csvwriter.writerow([q_file[cyc:cyc+1]['问题id'][cyc],
                    q_file[cyc:cyc+1]['问题'][cyc],
                    temp_e,q_file[cyc:cyc+1]['csv文件名'][cyc],max_index,max_number,temp_file_pieces_list])
g.close()  
                      