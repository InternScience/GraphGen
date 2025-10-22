def detect_main_language(text):
    """
    Detect the main language of the text, 'zh' for Chinese, 'en' for English

    :param text:
    :return:
    """
    assert isinstance(text, str)

    def is_chinese_char(char):
        return "\u4e00" <= char <= "\u9fff"

    def is_english_char(char):
        return char.isascii() and char.isalpha()

    # 去除空格和标点符号
    text = "".join(char for char in text if char.strip())

    chinese_count = sum(1 for char in text if is_chinese_char(char))
    english_count = sum(1 for char in text if is_english_char(char))

    total = chinese_count + english_count
    if total == 0:
        return "en"

    chinese_ratio = chinese_count / total

    if chinese_ratio >= 0.5:
        return "zh"
    return "en"


def detect_if_chinese(text):
    """
    Detect if the text contains any Chinese characters

    :param text:
    :return:
    """

    assert isinstance(text, str)
    return any("\u4e00" <= char <= "\u9fff" for char in text)
