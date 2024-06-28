import os
from langchain.document_loaders import TextLoader
from langchain.text_splitter import CharacterTextSplitter
from tqdm import tqdm
import jsonlines
import math
import re

from modelscope import AutoModelForCausalLM, AutoTokenizer, snapshot_download
from modelscope import GenerationConfig
import thulac
from setting import *

def lcs_seq_num(indices): # 计算最长公共子序列（LCS）中连续子序列的数量
    c = 0
    for i in range(1, len(indices)):
        if indices[i] - indices[i - 1] > 1: # 如果当前下标与前一个下标的差值大于 1，则表示这两个下标在原字符串中不连续，计数器 c 增加 1。
            c += 1
    return len(indices) - c # 表示 LCS 序列中连续子序列的数量。


def construct_lcs_with_indices(X, Y, L):
    '''
    计算两个字符串X和Y的LCS (最长公共子序列)。
    :param L: LCS
    :return: 返回两个字符串的最长公共子序列，LCS 中每个字符在字符串 Y 中的下标列表。
    '''
    lcs = []
    indices = []
    i, j = len(X), len(Y)

    while i > 0 and j > 0:
        if X[i - 1] == Y[j - 1]:
            lcs.append(X[i - 1])
            indices.append(j - 1)  # 添加当前字符在Y中的下标
            i -= 1
            j -= 1
        elif L[i - 1][j] > L[i][j - 1]:
            i -= 1
        else:
            j -= 1

    # 反转lcs和indices列表以获得正确的顺序
    return ''.join(reversed(lcs)), list(reversed(indices))


def lcs_with_character_indices(X, Y):
    '''
    计算两个字符串X和Y的LCS (最长公共子序列)。
    :return: X 和 Y 的最长公共子序列（LCS）的长度、LCS 序列、 LCS 序列在字符串 Y 中的下标位置。
    '''
    m, n = len(X), len(Y) # 获取字符串 X 和 Y 的长度
    L = [[0] * (n + 1) for _ in range(m + 1)] #  # 初始化二维数组 L，尺寸为 (m+1) x (n+1)

    for i in range(1, m + 1): # 遍历 X 的所有字符
        for j in range(1, n + 1):  # 遍历 Y 的所有字符
            if X[i - 1] == Y[j - 1]: # 如果当前字符相等
                L[i][j] = L[i - 1][j - 1] + 1 # 将当前字符加入LCS，长度加 1
            else: # 如果当前字符不相等
                L[i][j] = max(L[i - 1][j], L[i][j - 1])  # 取决于前一个状态的最大值

    lcs_seq, indices = construct_lcs_with_indices(X, Y, L)

    return len(lcs_seq), lcs_seq, indices


def close_match(q, indic, n):
    '''

    :param q: 查询的名称
    :param indic: 名称列表
    :param n: TOP_K
    :return: 返回前TOP_K个相似得分的数据
    '''
    indic_re = []
    for key in indic:
        r_len, lcs_seq, indices = lcs_with_character_indices(q, key) # 计算 q 和 key 的最长公共子序列（LCS）的长度、LCS 序列、LCS 序列在字符串 key 中的下标位置。
        seq_num = lcs_seq_num(indices) # 返回最长公共子序列（LCS）中连续子序列的数量
        indic_re.append((key, r_len + seq_num)) # 计算相似度：LCS的长度加上连续子序列的数量
    indic_re.sort(key=lambda x: x[1], reverse=True) # 倒序
    return indic_re[:n]


def extract_inc_name(question, docs):
    name = re.findall(r'(?:.*，)?(.*?)(?:股份)?有限公司', question) # 提取公司名称：选地匹配任意字符后跟一个中文逗号 ，。捕获并提取从上述匹配部分之后直到 "有限公司" 之前的任意字符，这些字符不包括 "股份"（如果存在）
    if len(name) == 0:
        name = re.findall(r'(?:.*，)?(.*?)股份', question) # 可选地匹配任意字符后跟一个中文逗号 ，。捕获并提取从上述匹配部分之后直到 "股份" 之前的任意字符，这些字符不包括 "股份"
    if len(name) == 0:
        name = question # 如果没匹配到，name就是问题
    else:
        name = name[0]
    inc = close_match(name, docs.keys(), n=1) # 返回问题和所有介绍文件中出现公司名，相似度最高的公司名称及其相似度（LCS的长度加上连续子序列的数量）
    return inc[0][0]


def extrac_doc_page(thu, stop_words, doc, question):
    seg_list = thu.cut(question.replace('.', '点')) # 用分词器对问题进行分词
    seg_list = [x[0] for x in seg_list]
    key_words = list(set(seg_list) - stop_words) # 删除停用词
    key_word_tf = {key: 0 for key in key_words}
    key_word_idf = {key: [] for key in key_words}
    for idx, p in enumerate(doc):
        finds = re.findall('|'.join(key_words), p.page_content) # 匹配文档片段中出现的问题中的词
        for r in finds:
            key_word_tf[r] += 1 # 记录该词出现的次数
            key_word_idf[r].append(idx) # 记录该词出现在的文档片段索引
    key_tfidf = []
    for key in key_words:
        # tf = key_word_tf[key]
        tf = 1
        idf = math.log(len(doc) / (len(set(key_word_idf[key])) + 1)) # 计算IDF
        key_tfidf.append((key, tf * idf)) # TFIDF得分，作者把TF设置为1，结果其实就是IDF得分
    key_tfidf.sort(key=lambda x: x[1], reverse=True)
    doc_p = [[i, 0] for i in range(len(doc))] # 这些文档片段的全部IDF得分，key是文档索引，value是IDF得分
    for key in key_tfidf:
        p_id = list(set(key_word_idf[key[0]])) # 得到该词出现在哪些文档中
        for p in p_id:
            doc_p[p][1] += key[1] # 把该词的IDF得分加上
    doc_p.sort(key=lambda x: x[1], reverse=True)
    after_rank_doc = []
    for i in doc_p[0:20]: # IDF得分前20的文档片段
        after_rank_doc.append(doc[i[0]].page_content) # 保存前20的文档片段的内容
    return '\n\n'.join(after_rank_doc), key_tfidf, doc_p # 返回相似度前20的文档片段的内容，问题中出现过的词的IDF得分，文档片段全部IDF得分


def read_jsonl(path):
    content = []
    with jsonlines.open(path, "r") as json_file:
        for obj in json_file.iter(type=dict, skip_invalid=True):
            content.append(obj)
    return content


def write_jsonl(path, content):
    with jsonlines.open(path, "w") as json_file:
        json_file.write_all(content)

### 文档分片 ####
def doc_faiss_create(fpath):
    files = os.listdir(fpath)
    docs = {}
    for fn in tqdm(files): # 遍历每一个scv.txt文件
        with open(os.path.join(fpath, fn)) as f:
            txt = f.read()
            # 匹配文档中的所有公司前缀（股份有限公司前的名字）
            r = re.findall('(?:^|：)(.*?)股份有限公司', txt) # 提取到的内容是开头或冒号 ： 和 "股份有限公司" 之间的字符。
            if len(r) == 0:
                r = re.findall('(?:^|：|\n)(.*?)股份有限公司', txt)[:3] #提取到的内容是开头或冒号 ：或换行符 \n 和 "股份有限公司" 之间的字符。
            assert len(r) != 0
            new = []
            for n in r:
                if '证券' not in n: # 如果名称中没有“证券”2字
                    new.append(n)
            r = new
        raw_documents = TextLoader(os.path.join(fpath, fn)).load()
        txt = []
        for line in raw_documents[0].page_content.split('\n'): # 文档内容用换行符分割
            if '......' in line: # 如果出现......，跳过，一般是目录
                continue
            else:
                txt.append(line)

        raw_documents[0].page_content = ''.join(txt)# 过滤了文件目录

        text_splitter = CharacterTextSplitter('，', chunk_size=200, chunk_overlap=0)
        documents = text_splitter.split_documents(raw_documents)
        docs[','.join(r)] = documents # 将出现的公司名称和文档内容对应起来
    return docs

def text_solution(fpath):
    model_dir = os.path.join(models_dir,'Tongyi-Finance-14B-Chat')
    tokenizer = AutoTokenizer.from_pretrained(model_dir, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(model_dir, device_map="cuda:0", trust_remote_code=True).eval()
    model.generation_config = GenerationConfig.from_pretrained(model_dir, trust_remote_code=True)
    # 遍历所有公司介绍scv文件，得到单个文档中出现的公司名称和对应documents字典
    docs = doc_faiss_create(os.path.join(data_dir,'bs_challenge_financial_14b_dataset/pdf_txt_file'))

    prompt = '''你是一个能精准提取文本信息并回答问题的AI。请你用下面提供的材料回答问题，材料是：```%s```。
    请根据以上材料回答问题："%s"。如果能根据给定材料回答，则提取出最合理的答案来回答问题,并回答出完整内容，如果不能找到答案，则回答“无法回答”，需要将问题复述一遍，现在请输出答案：'''
    answers = []
    stop_words = set(open('/app/baidu_stopwords.txt').read().split('\n') +
                     open('/app/hit_stopwords.txt').read().split('\n')
                     + ['，', '？', '。']) # 全部停用词
    thu = thulac.thulac(seg_only=True) # 用thulac工具进行分词。只进行分词，不进行词性标注
    ques = read_jsonl(os.path.join(data_dir,qfpath)) # 读取问题json文件

    for q in ques:
        if re.search(r'(?:股票|基金|A股|港股)', q['question']): # 如果包含这几个字，则跳过
            continue
        print(q['id'])
        name = extract_inc_name(q['question'], docs)# 返回问题和所有介绍文件中出现公司名，相似度最高的公司名
        doc = docs[name]

        prompt_inc = '''你是一个能精准提取公司名称的AI，公司名称类似`武汉兴图新科电子股份有限公司`，如果无法提取请输出`股份有限公司`。从下面文本中抽取公司名称:`{}`，
        注意文本只能来源于上述文本,现在请抽取：'''
        response, history = model.chat(tokenizer, prompt_inc.format(q['question']), history=None, temperature=0.0,
                                       do_sample=False) # 用LLM从问题中提取公司名
        qu = q['question'].replace(response, '')
        if qu == '':
            qu = q['question'].replace('股份', '').replace('有限公司', '')
        refer_txt, key_tfidf, doc_p = extrac_doc_page(thu, stop_words, doc, qu)# 返回相似度前20的文档的内容，问题中出现过的词的IDF得分，文档全部IDF得分

        response, history = model.chat(tokenizer, prompt % (refer_txt[:3000], q['question']), history=None,
                                       temperature=0.0, do_sample=False) # 让LLM根据参考内容给出问题的答案，如果无法回答输出“无法回答”
        if '无法' in response:
            retry = 0
            while retry < 3:
                response, history = model.chat(tokenizer, prompt % (
                refer_txt[retry * 2000: (retry + 1) * 2000], q['question']), history=None, # 根据参考内容后面的内容继续回答
                                               temperature=0.0, do_sample=False)
                if '无法' not in response:
                    break
                retry += 1

        q['answer'] = response
        answers.append(q)
        print(q['question'], '   回答：', response)


    write_jsonl(fpath, ques)
