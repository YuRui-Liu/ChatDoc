import os
import sys
import time
import gradio as gr
from dataclasses import dataclass
from typing import ClassVar
from ui.theme import JS_LIGHT_THEME, CSS
from logger import Logger
import random
from router import Agent
from utils import extract_tables_and_remainder, llm


@dataclass
class DefaultElement:
    DEFAULT_MESSAGE: ClassVar[dict] = {"text": ""}
    DEFAULT_MODEL: str = ""
    DEFAULT_HISTORY: ClassVar[list] = []
    DEFAULT_DOCUMENT: ClassVar[list] = []

    HELLO_MESSAGE: str = "Hi, 我是你的企业员工助手，请问有什么需要帮助的吗？"
    DEFAULT_STATUS: str = "准备就绪!"
    ANSWERING_STATUS: str = "回答中, 请稍后"
    COMPLETED_STATUS: str = "全部完成"


class LLMResponse:
    def __init__(self) -> None:
        self.router = Agent()

    def _yield_string(self, message: str):
        for i in range(len(message)):
            time.sleep(0.01)
            yield (
                DefaultElement.DEFAULT_MESSAGE,
                [[None, message[: i + 1]]],
                DefaultElement.DEFAULT_STATUS,
            )

    def welcome(self):
        """
        欢迎消息方法
        :return:
        """
        yield from self._yield_string(DefaultElement.HELLO_MESSAGE)  # 调用 _yield_string 生成欢迎消息的流式输出

    def empty_message(self):
        """

        :return:
        """
        yield from self._yield_string(DefaultElement.EMPTY_MESSAGE)

    def stream_response(
            self,
            message: str,
            history
    ):
        answer = []
        rag_response = self.router.text_completion(message)
        if isinstance(rag_response, str):  # 没有用到rag
            yield (
                DefaultElement.DEFAULT_MESSAGE,
                history + [[message, rag_response]],
                DefaultElement.ANSWERING_STATUS,
                "这个问题不属于本系统回答的范围",
                "",)
        else:
            # print("ui/rag_response:", rag_response)
            for chunk in rag_response[0]:
                # print("ui/rag_response.chunk:",chunk)
                text = chunk.choices[0].delta.content
                answer.append(text)
                gen_text = "".join(answer)

                yield (
                    DefaultElement.DEFAULT_MESSAGE,
                    history + [[message, gen_text]],
                    DefaultElement.ANSWERING_STATUS,
                    "参考信息生成中...",
                    "",)

            tables, context = extract_tables_and_remainder(rag_response[1])
            yield (
                DefaultElement.DEFAULT_MESSAGE,
                history + [[message, gen_text]],
                DefaultElement.COMPLETED_STATUS,
                context,
                "<br>".join(tables),)


class LocalChatbotUI:
    def __init__(
            self,
            logger: Logger,
            data_dir: str = "data/data",
            avatar_images=["./assets/user.png", "./assets/bot.png"],
    ):
        self._logger = logger
        self._data_dir = os.path.join(os.getcwd(), data_dir)
        self._avatar_images = [
            os.path.join(os.getcwd(), image) for image in avatar_images
        ]
        self._variant = "panel"
        self._llm_response = LLMResponse()

        self.model_config = {"base_url": llm.base_url, "api_key": llm.api_key, "model": llm.model}

    def _get_respone(
            self,
            chat_mode: str,
            message,
            chatbot,
            progress=gr.Progress(track_tqdm=True), ):

        # print("message:", message)
        if message["text"] in [None, ""]:
            for m in self._llm_response.welcome():
                yield m
        else:
            sys.stdout = self._logger
            history = chatbot
            # if chat_mode == "QA":
            #     history = []
            for m in self._llm_response.stream_response(message["text"], history):
                yield m

    def _undo_chat(self, history):
        """
        撤销按钮事件
        :param history:
        :return:
        """
        if len(history) > 0:
            history.pop(-1)
            return history
        return DefaultElement.DEFAULT_HISTORY

    def _reset_chat(self):
        """
        重置按钮
        :return:
        """
        gr.Info("Reset chat!")
        return (
            DefaultElement.DEFAULT_MESSAGE,
            DefaultElement.DEFAULT_HISTORY,
            DefaultElement.DEFAULT_STATUS,
            "",
            "",
        )

    def _clear_chat(self):
        """
        清空事件
        :return:
        """
        gr.Info("Clear chat!", duration=2)
        return (
            DefaultElement.DEFAULT_MESSAGE,
            DefaultElement.DEFAULT_HISTORY,
            DefaultElement.DEFAULT_STATUS,
            "",
            "",
        )

    def _show_hide_setting(self, state, type_name):
        """
        显示或隐藏侧边框
        :param state:
        :return:
        """
        state = not state
        if type_name == 'result':
            label = "Hide Result" if state else "Show Result"
        elif type_name == "setting":
            label = "Hide Setting" if state else "Show Setting"
        else:
            print("[Error]未定义的type!")
        return (label, gr.update(visible=state), state)

    def _welcome(self):
        """
        欢迎事件
        :return:
        """
        for m in self._llm_response.welcome():
            yield m

    # 处理设置保存的函数
    def _save_settings(self, base_url, api_key, model_name):
        self.model_config["base_url"] = base_url
        self.model_config["api_key"] = base_url
        self.model_config["model"] = model_name
        llm.api_key = api_key
        llm.model = model_name
        llm.base_url = base_url
        llm.reset_client()
        gr.Info("Settings saved!")

    def build(self):
        with gr.Blocks(
                theme=gr.themes.Soft(primary_hue="slate"),
                js=JS_LIGHT_THEME,
                css=CSS,
                title="ChatDoc：企业员工助手"
        ) as app:
            with gr.Row():
                gr.HTML("""
                        <div>
                            <h1 style="text-align: center; display:block;">ChatDoc: 企业员工助手</h1>
                        </div>
                        """)

            with gr.Tab("ChatDoc Info"):
                sidebar_state = gr.State(True)
                sidebar2_state = gr.State(False)
                with gr.Row(variant=self._variant, equal_height=True):
                    # 侧边栏
                    with gr.Column(variant=self._variant, scale=10, visible=sidebar_state.value) as setting:
                        with gr.Row(variant=self._variant, equal_height=False):
                            with gr.Column(variant=self._variant):
                                status = gr.Textbox(
                                    label="运行情况",
                                    value="准备就绪!",
                                    interactive=False)

                        with gr.Row(variant=self._variant, equal_height=False):
                            with gr.Column(variant="compact"):
                                contexts = gr.Textbox(
                                    label="参考信息",
                                    value="",
                                    interactive=False)
                                table_contexts = gr.HTML("<div>参考表格信息<div>")
                    # 侧边栏2
                    with gr.Column(variant=self._variant, scale=10, visible=sidebar2_state.value) as setting1:
                        with gr.Row(variant=self._variant, equal_height=False):
                            with gr.Column(variant=self._variant):
                                url_base_txt = gr.Textbox(
                                    label="url_base",
                                    value=self.model_config["base_url"],
                                    lines=3,
                                    interactive=True)
                            with gr.Column(variant=self._variant):
                                api_key_txt = gr.Textbox(
                                    label="api_key",
                                    value=self.model_config["api_key"],
                                    lines=3,
                                    interactive=True)
                            with gr.Column(variant=self._variant):
                                model_name_txt = gr.Textbox(
                                    label="model",
                                    value=self.model_config["model"],
                                    lines=2,
                                    interactive=True)

                        with gr.Row(variant=self._variant, equal_height=False):
                            save_setting_btn = gr.Button(value="SaveSetting", min_width=20)

                    #
                    with gr.Column(scale=30, variant=self._variant):
                        chatbot = gr.Chatbot(
                            layout="bubble",
                            value=[],
                            height=550,
                            scale=2,
                            show_copy_button=True,
                            bubble_full_width=False,
                            avatar_images=self._avatar_images,
                        )

                        with gr.Row(variant=self._variant):
                            chat_mode = gr.Dropdown(
                                choices=["QA"],
                                value="QA",
                                min_width=50,
                                show_label=False,
                                interactive=True,
                                allow_custom_value=False,
                            )
                            message = gr.MultimodalTextbox(
                                value=DefaultElement.DEFAULT_MESSAGE,
                                placeholder="你好，请输入问题",
                                file_types=[".txt", ".pdf", ".csv"],
                                show_label=False,
                                scale=6,
                                lines=1,
                                autofocus=True,
                            )
                        with gr.Row(variant="compact"):
                            ui_btn = gr.Button(
                                value="Hide result"
                                if sidebar_state.value
                                else "Show result",
                                min_width=20,
                            )
                            undo_btn = gr.Button(value="Undo", min_width=20)
                            clear_btn = gr.Button(value="Clear", min_width=20)
                            reset_btn = gr.Button(value="Reset", min_width=20)
                            settings_btn = gr.Button(value="Settings", min_width=20)


            # process event
            clear_btn.click(self._clear_chat, outputs=[message, chatbot, status, contexts, table_contexts])
            undo_btn.click(self._undo_chat, inputs=[chatbot], outputs=[chatbot])
            reset_btn.click(self._reset_chat, outputs=[message, chatbot, status, contexts, table_contexts])
            settings_btn.click(self._show_hide_setting, inputs=[sidebar2_state, gr.Textbox(value="setting",visible=False)],
                               outputs=[settings_btn, setting1, sidebar2_state])
            save_setting_btn.click(self._save_settings,
                                   inputs=[url_base_txt, api_key_txt, model_name_txt])

            message.submit(
                self._get_respone,
                inputs=[chat_mode, message, chatbot],
                outputs=[message, chatbot, status, contexts, table_contexts],
            )

            ui_btn.click(
                self._show_hide_setting,
                inputs=[sidebar_state, gr.Textbox(value="result",visible=False)],
                outputs=[ui_btn, setting, sidebar_state],
            )
            app.load(self._welcome, outputs=[message, chatbot, status])

        return app
