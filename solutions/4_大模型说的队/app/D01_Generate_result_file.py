import csv
import json
import pandas as pd

# 写入答案，如果有相似公司名，使用检索文档结果输入LLM生成的答案；如果没有公司名，使用sql查询结果输入LLM生成的答案

csv_file_1_dir = "/app/intermediate/FA_V5_SQL.csv" # '问题id','问题','FA','SQL结果'
csv_file_1 = pd.read_csv(csv_file_1_dir,delimiter = ",",header = 0)
csv_file_2_dir = "/app/intermediate/FA_V5_Text_cap4_4_nt.csv" # '问题id','问题','实体答案','final_ans1','ans_list'
csv_file_2 = pd.read_csv(csv_file_2_dir,delimiter = ",",header = 0)

print('D01_started')
list_of_items = list()
for cyc in range(1000):
    temp_dict = {}
    temp_dict['id'] = str(csv_file_1['问题id'][cyc])
    temp_dict["question"]= csv_file_1['问题'][cyc]
    temp_answer = ""
    if csv_file_2[cyc:cyc+1]['实体答案'][cyc] != 'N_A': # 有公司名
        temp_answer = str(csv_file_2[cyc:cyc+1]['final_ans1'][cyc]) # 使用检索文档结果，LLM生成的答案
    else:
        temp_answer = str(csv_file_1[cyc:cyc+1]['FA'][cyc]) # 根据sql查询结果LLM生成的答案
    
    for cyc in range(10):
        temp_answer = temp_answer.replace("根据资料%s" % str(cyc),'') # 删除“根据资料0”，“根据资料1”类似文字
    
    temp_dict["answer"] = temp_answer
    list_of_items.append(temp_dict)
    
with open('/app/submit_result.jsonl',mode = 'w', encoding = 'utf-8') as f:
    for cyc in range(1000):
        temp_dict = list_of_items[cyc]
        temp_str = json.dumps(temp_dict,ensure_ascii = False).replace('{"id": "','{"id": ').replace('", "question":',', "question":')
        f.write(temp_str+'\n')

f.close()
exit()