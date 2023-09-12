from langchain.chat_models import ChatOpenAI
from langchain.chains import LLMChain
from langchain.llms import ChatGLM

from langchain.prompts.chat import (
    ChatPromptTemplate,
    SystemMessagePromptTemplate,
    HumanMessagePromptTemplate,
)
from langchain import PromptTemplate

from utils import LOG
from abc import ABCMeta, abstractmethod

class TranslationChain(metaclass=ABCMeta):
    
    def run(self, text: str, source_language: str, target_language: str, style: str) -> (str, bool):
        result = ""
        try:
            result = self.chain.run({
                "text": text,
                "source_language": source_language,
                "target_language": target_language,
                "style": style
            })
        except Exception as e:
            LOG.error(f"An error occurred during translation: {e}")
            return result, False

        return result, True

class OpenAITranlateChain(TranslationChain):
    
    def __init__(self, model_name: str = "gpt-3.5-turbo", verbose: bool = True):            
        template = (
        """Act as a language translator. 
        You will receive text to translate, and your goal is to translate the text from {source_language} to {target_language} in a {style} style.
        
        and don’t short my text. 
        Do not echo my prompt. 
        Do not remind me what I asked you for. 
        And don’t short my text .
        do not apologize. 
        Do not explain what and why,  just give me your best possible Output. 
        output should only be the translated text no additional explanation and text.
        """    
        )        
        system_message_prompt = SystemMessagePromptTemplate.from_template(template)

        # 待翻译文本由 Human 角色输入
        human_template = "{text}"
        human_message_prompt = HumanMessagePromptTemplate.from_template(human_template)

        # 使用 System 和 Human 角色的提示模板构造 ChatPromptTemplate
        chat_prompt_template = ChatPromptTemplate.from_messages(
            [system_message_prompt, human_message_prompt]
        )

        # 为了翻译结果的稳定性，将 temperature 设置为 0
        chat = ChatOpenAI(model_name=model_name, temperature=0, verbose=verbose)

        self.chain = LLMChain(llm=chat, prompt=chat_prompt_template, verbose=verbose)


class ChatGLMTranlateChain(TranslationChain):
    def __init__(self, model_name: str, verbose:bool):
               
        template = (
        """
            你现在是资深的多语言翻译专家， 请采用{style}的语言风格，把下面的内容，从语言{source_language}翻译到语言{target_language}：
            {text}
        """    
        )        
        tranlate_prompt = PromptTemplate(
            input_variables=["source_language","target_language","style","text"],
            template=template
        )
        
        endpoint_url = "http://127.0.0.1:8000"
        glm_llm = ChatGLM(
            endpoint_url=endpoint_url,
            max_token=80000,
            history=[],
            top_p=0.9,
            model_kwargs={"sample_model_args": False},
        )
        self.chain = LLMChain(llm=glm_llm, prompt=tranlate_prompt, verbose=verbose)
  
        
chain_factories = {
   "gpt-3.5-turbo": OpenAITranlateChain,
   "chatglm-6b": ChatGLMTranlateChain,
}

def create_tranlate_chain(model_name, verbose: bool = False) -> TranslationChain:
   if model_name not in chain_factories:
       raise ValueError("Invalid model name!")
   return chain_factories[model_name](model_name,verbose)