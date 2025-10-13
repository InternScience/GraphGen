# pylint: disable=C0301
TEMPLATE_SINGLE_EN: str = """The answer to a question is provided. Please generate a question that corresponds to the answer.

################
Answer:
{answer}
################
Question:
"""

TEMPLATE_SINGLE_ZH: str = """下面提供了一个问题的答案，请生成一个与答案对应的问题。

################
答案：
{answer}
################
问题：
"""


# TODO: 修改这里的prompt
TEMPLATE_MULTI_EN = """You are an assistant to help read a article and then rephrase it in a question answering format. The user will provide you with an article with its content. You need to generate a paraphrase of the same article in question and answer format with one tag of "Question: ..." followed by "Answer: ...". Remember to keep the meaning and every content of the article intact.

Here is the format you should follow for your response:
Question: <Question>
Answer: <Answer>

Here is the article you need to rephrase:
{doc}
"""

TEMPLATE_MULTI_ZH = """你是一位助手，帮助阅读一篇文章，然后以问答格式重述它。用户将为您提供一篇带有内容的文章。你需要以一个标签"问题：..."为开头，接着是"答案：..."，生成一篇与原文章相同的问答格式的重述。请确保保持文章的意义和每个内容不变。

以下是你应该遵循的响应格式：
问题： <问题>
答案： <答案>

以下是你需要重述的文章：
{doc}
"""

QUESTION_GENERATION_PROMPT = {
    "English": {
        "SINGLE_TEMPLATE": TEMPLATE_SINGLE_EN,
        "MULTI_TEMPLATE": TEMPLATE_MULTI_EN,
    },
    "Chinese": {
        "SINGLE_TEMPLATE": TEMPLATE_SINGLE_ZH,
        "MULTI_TEMPLATE": TEMPLATE_MULTI_ZH,
    },
}
