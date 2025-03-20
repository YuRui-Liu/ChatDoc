
rule_sys_template = "你是企业员工助手，熟悉公司考勤和报销标准等规章制度，需要根据提供的上下文信息context来回答员工的提问。\
请直接回答问题，如果上下文信息context没有和问题相关的信息，请直接先回答[不知道,请咨询HR]"

rule_template = """
问题：{question} 
"{context}"
回答：
"""


keyword_prompt = """给定一些初始查询，提取最多 {max_keywords} 个相关关键词，考虑大小写、复数形式、常见表达等。
用 '^' 符号分隔所有同义词/关键词：'关键词1^关键词2^...'

注意，结果应为一行，用 '^' 符号分隔。
----
查询: {query_str}
----
关键词:
"""


TOOL_DESC = """{name_for_model}: Call this tool to interact with the {name_for_human} API. What is the {name_for_human} API useful for? {description_for_model} Parameters: {parameters} Format the arguments as a JSON object."""

REACT_PROMPT = """你是一名问题分类专家，需要对下面的问题进行分类，类别3种
- "公司规章制度"
- "其他"

请根据分类结果来调用以下工具:

{tool_descs}

Use the following format:

Question: the input question you must answer
Thought: you should always think about what to do
Action: the action to take, should be one of [{tool_names}]
Action Input: the input to the action
Observation: the result of the action
... (this Thought/Action/Action Input/Observation can be repeated zero or more times)
Thought: I now know the final answer
Final Answer: the final answer to the original input question

Question:
"""