import sys
import os

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import pytest
from .translator.translation_chain import create_tranlate_chain

@pytest.fixture
def translation_instance():
    # 创建翻译实例，用于测试
    return create_tranlate_chain(model_name="chatglm-6b", verbose=True)

def test_run_positive(translation_instance):
    source_language = "Chinese"
    target_language = "English"
    style = "creative"
    text = """
        预训练模型在自然语言处理领域并不是 OpenAI 的专利。早在 2013 年，就有一篇叫做 Word2Vec 的经典论文谈到过。它能够通过预训练，根据同一个句子里一个单词前后出现的单词，来得到每个单词的向量。而在 2018 年，Google 关于 BERT 的论文发表之后，整个业界也都会使用 BERT 这样的预训练模型，把一段文本变成向量用来解决自己的自然语言处理任务。在 GPT-3 论文发表之前，大家普遍的结论是，BERT 作为预训练的模型效果也是优于 GPT 的。
    """
    
    result, success = translation_instance.run(
        text, source_language, target_language, style)

    print(result)
    assert success is True
