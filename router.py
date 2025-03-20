import json
from utils import run_rag_pipline
from model import RagLLM
from prompt_cfg import TOOL_DESC, REACT_PROMPT


class Tools:
    def __init__(self) -> None:
        self.toolConfig = self._tools()

    def _tools(self):
        tools = [
            {
                'name_for_human': '查询公司规章制度的工具',
                'name_for_model': 'get_guizha',
                'description_for_model': '获取公司的相关规章制度，包括考勤、工作时间、请假、出差费用规定',
                'parameters': []
            },
            {
                'name_for_human': '查询其他问题的工具',
                'name_for_model': 'other',
                'description_for_model': '获取其他问题的信息等',
                'parameters': []
            }
        ]
        return tools


    def get_guizha(self, query):
        """
        插件：规章制度  被call_plugin调用
        :param query:
        :return:
        """
        return run_rag_pipline(query, query, stream=True)

    def other(self, query):
        return "对不起，我不能回答这个问题"


class Agent:

    def __init__(self) -> None:
        self.tool = Tools()
        self.model = RagLLM()  # 实例化 RagLLM 类，作为语言模型
        self.system_prompt = self.build_system_input()

    def build_system_input(self):
        """
        构建系统提示（包含工具描述）
        """
        tool_descs, tool_names = [], []  # 初始化工具描述列表和工具名称列表
        for tool in self.tool.toolConfig:  # 遍历工具配置
            tool_descs.append(TOOL_DESC.format(**tool))  # 使用 TOOL_DESC 模板格式化工具描述
            tool_names.append(tool['name_for_model'])  # 添加工具名称到列表
        tool_descs = '\n\n'.join(tool_descs)  # 将工具描述用双换行符连接成字符串
        tool_names = ','.join(tool_names)  # 将工具名称用逗号连接成字符串
        sys_prompt = REACT_PROMPT.format(tool_descs=tool_descs, tool_names=tool_names)  # 使用 REACT_PROMPT 模板格式化系统提示
        return sys_prompt  # 返回构建好的系统提示

    def parse_latest_plugin_call(self, text):
        """
        解析文本中最近的插件调用
        :param text:
        :return:
        """
        plugin_name, plugin_args = '', ''  # 初始化插件名称和参数为空字符串
        i = text.rfind('\nAction:')  # 查找最后一个 '\nAction:' 的位置
        j = text.rfind('\nAction Input:')  # 查找最后一个 '\nAction Input:' 的位置
        k = text.rfind('\nObservation:')  # 查找最后一个 '\nObservation:' 的位置
        if 0 <= i < j:  # 如果存在 Action 且在 Action Input 之前
            if k < j:  # 如果 Observation 在 Action Input 之前（未完成调用）
                text = text.rstrip() + '\nObservation:'  # 添加 Observation 标记
            k = text.rfind('\nObservation:')  # 重新查找 Observation 位置
            plugin_name = text[i + len('\nAction:'): j].strip()  # 提取插件名称
            plugin_args = text[j + len('\nAction Input:'): k].strip()  # 提取插件参数
            text = text[:k]  # 截取文本到 Observation 之前
        return plugin_name, plugin_args, text  # 返回插件名称、参数和截取后的文本

    def call_plugin(self, plugin_name, plugin_args, ori_text):
        """
        调用指定的插件   被 text_completion 调用
        :param plugin_name:
        :param plugin_args:
        :param ori_text:
        :return:
        """
        try:
            plugin_args = json.loads(plugin_args)  # 尝试将插件参数解析为 JSON 格式
        except:
            pass  # 如果解析失败，保持原始字符串形式
        if plugin_name == 'get_guizha':
            return self.tool.get_guizha(ori_text)
        if plugin_name == 'other':
            return self.tool.other(ori_text)

    def text_completion(self, text, history=[]):
        """
        定义文本补全方法，处理用户输入
        :param text: 原始输入文本
        :param history: 历史输入文本
        :return:
        """
        ori_text = text  # 保存原始文本
        text = "\nQuestion:" + text  # 在输入文本前添加 Question 标记
        response = self.model(f"{self.system_prompt} \n {text}")  # 调用语言模型，传入系统提示和用户输入
        """
        response格式
            Thought: 这个问题不属于"公司规章制度"或"企业、金融和商业"类别，应该归类为"其他"。
            Action: other
            Action Input: {}
            Observation: 
            Thought: 我现在知道最终答案了
            Final Answer: 你好，这个问题似乎没有涉及到具体的内容。如果您有其他问题，欢迎随时提问。
        """
        plugin_name, plugin_args, response = self.parse_latest_plugin_call(response)  # 解析模型响应中的插件调用

        """
        plugin_name = other
        plugin_args = {}
        response格式
        Thought: 这个问题不属于"公司规章制度"或"企业、金融和商业"类别，应该归类为"其他"。
        Action: other
        Action Input: {}
        """

        # 匹配插件
        if plugin_name:  # 如果解析到插件名称
            return self.call_plugin(plugin_name, plugin_args, ori_text)  # 调用对应的插件并返回结果
        return "对不起，我不能回答这个问题"  # 如果没有插件调用，返回默认回复
