import csv
import pandas as pd
import numpy as np
import re
import copy
from modelscope import AutoModelForCausalLM, AutoTokenizer, snapshot_download
from modelscope import GenerationConfig

# A02：当任务为text 文本检索时，每个问题添加实体（对于的公司名称）和对于的csv文件

model_dir = '/tcdata/models/Tongyi-Finance-14B-Chat'

# Note: The default behavior now has injection attack prevention off.
tokenizer = AutoTokenizer.from_pretrained(model_dir, trust_remote_code=True)

new_question_file_dir = '/app/intermediate/A01_question_classify.csv'
new_question_file = pd.read_csv(new_question_file_dir, delimiter=",", header=0)
company_file_dir = '/app/data/files/AF0_pdf_to_company.csv'
company_file = pd.read_csv(company_file_dir, delimiter=",", header=0)
company_data_csv_list = list()
company_index_list = list()  # 公司名称的token_id的list
company_name_list = list()
for cyc in range(len(company_file)):
    company_name_list.append(company_file[cyc:cyc + 1]['公司名称'][cyc])
    company_data_csv_list.append(company_file[cyc:cyc + 1]['csv文件名'][cyc])
    temp_index_cp = tokenizer(company_file[cyc:cyc + 1]['公司名称'][cyc])
    temp_index_cp = temp_index_cp['input_ids']  # 把公司名称转换为token_id
    company_index_list.append(temp_index_cp)

g = open('/app/intermediate/A02_question_classify_entity.csv', 'w', newline='', encoding='utf-8-sig')
csvwriter = csv.writer(g)
csvwriter.writerow(['问题id', '问题', '分类', '对应实体', 'csv文件名'])  # 对应实体 = 公司名称

for cyc in range(len(new_question_file)):

    tempw_id = new_question_file[cyc:cyc + 1]['问题id'][cyc]
    tempw_q = new_question_file[cyc:cyc + 1]['问题'][cyc]
    tempw_q_class = new_question_file[cyc:cyc + 1]['分类'][cyc]
    tempw_entity = 'N_A'
    tempw_csv_name = 'N_A'

    if new_question_file[cyc:cyc + 1]['分类'][cyc] == 'Text':  # 如果任务分类是文本检索
        temp_index_q = tokenizer(new_question_file[cyc:cyc + 1]['问题'][cyc])
        temp_index_q = temp_index_q['input_ids']  # 把这个问题转换为token_id
        q_cp_similarity_list = list()
        for cyc2 in range(len(company_file)):
            temp_index_cp = company_index_list[cyc2]
            # 计算公司名token和问题token的 Jaccard 相似度
            temp_simi = len(set(temp_index_cp) & set(temp_index_q)) / (len(set(temp_index_cp)) + len(set(temp_index_q)))
            q_cp_similarity_list.append(temp_simi)  # 计算这个问题token和每个公司名token的相似度

        t = copy.deepcopy(q_cp_similarity_list)
        max_number = []
        max_index = []

        for _ in range(1):  # 找出相似度最大的n个公司名，这里取的n=1，
            number = max(t)
            index = t.index(number)
            t[index] = 0
            max_number.append(number)  # 找出相似度最大的公司名
            max_index.append(index)  # 找出相似度最大的公司名在列表中的位置
        t = []
        tempw_entity = company_name_list[max_index[0]]
        tempw_csv_name = company_data_csv_list[max_index[0]]

        csvwriter.writerow([str(tempw_id), str(tempw_q), tempw_q_class, tempw_entity, tempw_csv_name])
    elif new_question_file[cyc:cyc + 1]['分类'][cyc] == 'SQL':  # 任务分类为SQL的不作处理，实体位置写入N_A
        csvwriter.writerow([str(tempw_id), str(tempw_q), tempw_q_class, tempw_entity, tempw_csv_name])
    else:
        find_its_name_flag = 0
        for cyc_name in range(len(company_name_list)):
            if company_name_list[cyc_name] in tempw_q:  # 如果公司名称出现在问题里，把公司的名字和csv文件存入
                tempw_entity = company_name_list[cyc_name]
                tempw_csv_name = company_data_csv_list[cyc_name]
                csvwriter.writerow([str(tempw_id), str(tempw_q), tempw_q_class, tempw_entity, tempw_csv_name])
                find_its_name_flag = 1
                break
        if find_its_name_flag == 0:
            csvwriter.writerow([str(tempw_id), str(tempw_q), tempw_q_class, tempw_entity, tempw_csv_name])

g.close()
print('A02_finished')
exit()
