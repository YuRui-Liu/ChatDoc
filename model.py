import warnings

warnings.filterwarnings('ignore')
from langchain.llms.base import LLM
from typing import Any, List, Optional
from openai import OpenAI
from langchain.callbacks.manager import CallbackManagerForLLMRun


# Rag模型
class RagLLM(object):
    client: Optional[Any] = None
    message: Optional[List] = None

    def __init__(self, api_key="sk-c5cfa1d6bd5a4c6f82d4041f24a1c03f",
                 base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
                 model="qwen2.5-72b-instruct",
                 sys_prompt=""):
        super().__init__()
        self.api_key = api_key
        self.base_url = base_url
        self.model = model
        self.client = OpenAI(api_key=self.api_key, base_url=self.base_url, )
        self.message = [{"role": "system", "content": sys_prompt},
                        {"role": "user", "content": None}, ]

    def update_message(self, query, prompt):
        if prompt is not None:
            self.message[0]["content"] = prompt
        if query is not None:
            self.message[1]["content"] = query

    def __call__(self, prompt: str, **kwargs: Any):
        self.update_message(query=prompt, prompt=kwargs.get('prompt_sys', None))  # 更新message
        completion = self.client.chat.completions.create(model=self.model,
                                                         messages=self.message,
                                                         temperature=kwargs.get('temperature', 0.1),
                                                         top_p=kwargs.get('top_p', 0.9),
                                                         max_tokens=kwargs.get('max_tokens', 4096),
                                                         stream=kwargs.get('stream', False))
        if kwargs.get("stream", False):
            return completion

        return completion.choices[0].message.content

    def reset_client(self):
        self.client = OpenAI(api_key=self.api_key, base_url=self.base_url)

# Qwen 模型
class QwenLLM(LLM):
    client: Optional[Any] = None
    message: Optional[List] = None
    model_config = {"extra": "allow"}  # 允许额外字段

    def __init__(self, api_key="sk-c5cfa1d6bd5a4c6f82d4041f24a1c03f",
                 base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
                 model="qwen2.5-72b-instruct",
                 sys_prompt=""):
        super().__init__()
        self.api_key = api_key
        self.base_url = base_url
        self.model = model
        self.client = OpenAI(api_key=self.api_key, base_url=self.base_url, )
        self.message = [{"role": "system", "content": sys_prompt},
                        {"role": "user", "content": None}, ]

    def update_message(self, query, prompt):
        if prompt is not None:
            self.message[0]["content"] = prompt
        if query is not None:
            self.message[1]["content"] = query

    def _call(self,
              prompt: str,
              stop: Optional[List[str]] = None,
              run_manager: Optional[CallbackManagerForLLMRun] = None,
              model="qwen2.5-72b-instruct",
              **kwargs: Any):
        self.update_message(prompt, kwargs.get('prompt_sys', None))  # 更新message
        completion = self.client.chat.completions.create(model=self.model,
                                                         messages=self.message,
                                                         temperature=kwargs.get('temperature', 0.1),
                                                         top_p=kwargs.get('top_p', 0.9),
                                                         max_tokens=kwargs.get('max_tokens', 4096),
                                                         stream=kwargs.get('stream', False))
        return completion.choices[0].message.content

    @property
    def _llm_type(self) -> str:
        return self.model


from langchain.embeddings.huggingface import HuggingFaceEmbeddings


class RagEmbedding(object):
    def __init__(self, model_path="./data/llm_app/embedding_models/bge-m3//",
                 device="cpu"):
        self.embedding = HuggingFaceEmbeddings(model_name=model_path,
                                               model_kwargs={"device": "cpu"})

    def get_embedding_fun(self):
        return self.embedding
