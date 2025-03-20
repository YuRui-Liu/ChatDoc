# -*-coding:utf-8 -*-

import warnings

warnings.filterwarnings('ignore')
import re
import traceback
import json
import random
import numpy as np
from langchain_chroma import Chroma
import chromadb
from langchain import PromptTemplate
from openai import OpenAI
from py2neo import Graph
from model import RagEmbedding, RagLLM, QwenLLM
from prompt_cfg import rule_template, keyword_prompt,rule_sys_template

llm = RagLLM(sys_prompt=rule_sys_template)
embedding_model = RagEmbedding()
# chroma_client = chromadb.HttpClient(host="localhost", port=8000)
chroma_client = chromadb.PersistentClient(path="./data/data_base")
zhidu_db = Chroma("zhidu_db",
                  embedding_model.get_embedding_fun(),
                  client=chroma_client)


# graph = Graph("bolt://180.xxx.26.xx:7687", user='neo4j', password='neo4j@123',name='neo4j')

def run_chat(prompt, history=[]):
    client = OpenAI(
        base_url='http://180.xxx.xxx.247:11434/v1/',
        api_key='qwen2:72b',
    )

    history_msg = []
    for idx, msg in enumerate(history):
        if idx == 0:
            continue
        history_msg.append({"role": "user", "content": msg[0]})
        history_msg.append({"role": "assistant", "content": msg[1]})
    # print(history_msg)

    chat_completion = client.chat.completions.create(
        messages=history_msg + [
            {
                'role': 'user',
                'content': prompt,
            }
        ],
        max_tokens=4096,  # 最大生成的token数量。
        stream=True,  # 开启流式输出
        model='qwen2:72b',
        temperature=0.1,  # 控制生成文本的随机性。越低越确定，越高越随机。
        top_p=0.9,
    )
    return chat_completion


def run_rag_pipline(query, context_query, k=3, context_query_type="query",
                    stream=True, prompt_template=rule_template,
                    temperature=0.1):  # 定义
    '''
     RAG Pipline，处理查询和上下文
    :param query:
    :param context_query:  上下文查询类型
    :param k:
    :param context_query_type:
    :param stream:
    :param prompt_template:
    :param temperature:
    :return:
    '''
    llm.update_message(None, "")
    if context_query_type == "vector":  # 如果上下文查询类型是向量
        related_docs = zhidu_db.similarity_search_by_vector(context_query, k=k)  # 使用向量搜索相似文档
    elif context_query_type == "query":  # 如果上下文查询类型是查询字符串
        related_docs = zhidu_db.similarity_search(context_query, k=k)  # 使用查询字符串搜索相似文档
    elif context_query_type == "doc":  # 如果上下文查询类型是文档
        related_docs = context_query  # 直接使用传入的文档作为上下文
    else:  # 默认情况
        related_docs = zhidu_db.similarity_search(context_query, k=k)  # 使用查询字符串搜索相似文档

    # print("related_docs:", related_docs)
    """
    [Document(id='708d5617-da16-4f43-9d52-d978588e7d8b', metadata={'is_table': 1, 'type': 'ori'}, page_content='<table><caption>病假发放标准：</caption>\n<tr><td  >病假时间</td><td  >连续工龄</td><td  >发放标准</td></tr>\n<tr><td></td><td  >不满二年</td><td  >60% </td></tr>\n<tr><td></td><td  >已满二年不满四年</td><td  >70% </td></tr>\n<tr><td  >6 个月以内病假</td><td  >已满四年不满六年</td><td  >80% </td></tr>\n<tr><td></td><td  >已满六年不满八年</td><td  >90% </td></tr>\n<tr><td></td><td  >八年以上</td><td  >100% </td></tr>\n<tr><td></td><td  >不满一年</td><td  >40% </td></tr>\n<tr><td  >6 个月以上病假</td><td  >已满一年不满三年</td><td  >50% </td></tr>\n<tr><td></td><td  >连续工龄三年以上</td><td  >60% </td></tr>\n</table>'), Document(id='99918fe4-e709-48f7-962c-e498e6c2b846', metadata={'is_table': 1, 'type': 'ori'}, page_content='<table><caption>病假发放标准：</caption>\n<tr><td  >病假时间</td><td  >连续工龄</td><td  >发放标准</td></tr>\n<tr><td></td><td  >不满二年</td><td  >60% </td></tr>\n<tr><td></td><td  >已满二年不满四年</td><td  >70% </td></tr>\n<tr><td  >6 个月以内病假</td><td  >已满四年不满六年</td><td  >80% </td></tr>\n<tr><td></td><td  >已满六年不满八年</td><td  >90% </td></tr>\n<tr><td></td><td  >八年以上</td><td  >100% </td></tr>\n<tr><td></td><td  >不满一年</td><td  >40% </td></tr>\n<tr><td  >6 个月以上病假</td><td  >已满一年不满三年</td><td  >50% </td></tr>\n<tr><td></td><td  >连续工龄三年以上</td><td  >60% </td></tr>\n</table>'), Document(id='c8662f2d-a699-4f9f-926e-f3c0e09db754', metadata={'is_table': 1, 'type': 'ori'}, page_content='<table><caption>病假发放标准：</caption>\n<tr><td  >病假时间</td><td  >连续工龄</td><td  >发放标准</td></tr>\n<tr><td></td><td  >不满二年</td><td  >60% </td></tr>\n<tr><td></td><td  >已满二年不满四年</td><td  >70% </td></tr>\n<tr><td  >6 个月以内病假</td><td  >已满四年不满六年</td><td  >80% </td></tr>\n<tr><td></td><td  >已满六年不满八年</td><td  >90% </td></tr>\n<tr><td></td><td  >八年以上</td><td  >100% </td></tr>\n<tr><td></td><td  >不满一年</td><td  >40% </td></tr>\n<tr><td  >6 个月以上病假</td><td  >已满一年不满三年</td><td  >50% </td></tr>\n<tr><td></td><td  >连续工龄三年以上</td><td  >60% </td></tr>\n</table>')]

    """
    context = "\n".join([f"上下文{i + 1}: {doc.page_content} \n" \
                         for i, doc in enumerate(related_docs)])  # 将相关文档内容合并为上下文字符串
    # print("context：", context)
    prompt = PromptTemplate(  # 创建提示模板
        input_variables=["question", "context"],
        template=prompt_template, )  # 使用传入的模板（rule_template）
    llm_prompt = prompt.format(question=query, context=context)  # 格式化提示，填入查询和上下文
    # print("llm_prompt:", llm_prompt)
    if stream:  # 如果启用流式输出
        response = llm(llm_prompt, stream=True)  # 调用 LLM，传入提示并启用流式输出

        # print("utils", response)

        # for chunk in response:
        #     ab = chunk.choices[0].delta.content
        #     print(ab)
        return response, context  # 返回流式响应和上下文
    else:  # 如果不启用流式输出
        response = llm(llm_prompt, stream=False, temperature=temperature)  # 调用 LLM，传入提示和温度参数
        return response, context  # 返回完整响应和上下文


def parse_query(query, max_keywords=3):
    '''
    查询解析函数，大模型提取关键词
    :param query:
    :param max_keywords: 指定最大关键词数
    :return:
    '''
    prompt_template = PromptTemplate(  # 创建提示模板
        input_variables=["query_str", "max_keywords"],  # 定义模板的输入变量
        template=keyword_prompt,  # 使用外部定义的关键词提取模板（keyword_prompt）
    )

    final_prompt = prompt_template.format(max_keywords='3',  # 格式化模板，
                                          query_str=query)  # 填入查询字符串
    llm.update_message(None, "")
    response = llm(final_prompt)  # 调用 LLM，传入格式化后的提示，获取响应
    keywords = response.split('\n')[0].split('^')  # 从响应中提取第一行，并按 '^' 分割为关键词列表
    return keywords  # 返回关键词列表


def extract_tables_and_remainder(text):
    '''

    :param text: 文本字符串作为输入
    :return: 返回一个元组，包含提取的表格列表和剩余文本
    '''
    pattern = r'<table.*?>.*?</table>'
    tables = re.findall(pattern, text, re.DOTALL)
    remainder = re.sub(pattern, '', text, flags=re.DOTALL).strip()
    return tables, remainder
