import functools
import inspect
import json
import os
from contextlib import contextmanager

import gradio as gr
import langcodes
import yaml
from gradio.blocks import Block, BlockContext, Context, LocalContext


# Monkey patch to escape I18nString type being stripped in gradio.Markdown
def escape_caller(func):
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        if args and isinstance(args[0], I18nString):
            add_values = args[0].add_values
            radd_values = args[0].radd_values
            result = I18nString(func(*args, **kwargs))
            result.add_values = add_values
            result.radd_values = radd_values

            return result
        return func(*args, **kwargs)

    return wrapper


inspect.cleandoc = escape_caller(inspect.cleandoc)


class TranslateContext:
    dictionary: dict = {}

    def add_translation(translation: dict):
        for k, v in translation.items():
            if k not in TranslateContext.dictionary:
                TranslateContext.dictionary[k] = {}
            TranslateContext.dictionary[k].update(v)

    def get_available_languages():
        return list(TranslateContext.dictionary.keys())

    lang_per_session = {}

    def get_current_language(request: gr.Request):
        return TranslateContext.lang_per_session.get(
            request.session_hash, TranslateContext.default_language
        )

    def set_current_language(request: gr.Request, lang: str):
        TranslateContext.lang_per_session[request.session_hash] = lang

    default_language = "en"

    def get_default_language():
        return TranslateContext.default_language

    def set_default_language(lang: str):
        TranslateContext.default_language = lang


def get_lang_from_request(request: gr.Request):
    lang = request.headers["Accept-Language"].split(",")[0]
    lang, _ = langcodes.closest_match(lang, TranslateContext.get_available_languages())

    if not lang:
        return "en"
    return lang


class I18nString(str):
    def __new__(cls, value):
        request: gr.Request = LocalContext.request.get()
        if request is None:
            return super().__new__(cls, value)

        lang = TranslateContext.get_current_language(request)
        result = TranslateContext.dictionary.get(lang, {}).get(value, value)
        return result

    def __init__(self, value):
        if isinstance(value, I18nString):
            self.add_values = value.add_values
            self.radd_values = value.radd_values
        else:
            self.add_values = []
            self.radd_values = []

    def __str__(self):
        request: gr.Request = LocalContext.request.get()
        if request is None:
            return self

        lang = TranslateContext.get_current_language(request)
        result = TranslateContext.dictionary.get(lang, {}).get(self, super().__str__())

        for v in self.radd_values:
            result = str(v) + result

        for v in self.add_values:
            result = result + str(v)

        # hotfix, remove unexpected single quotes
        while len(result) >= 2 and result.startswith("'") and result.endswith("'"):
            result = result[1:-1]

        return result

    def __add__(self, other):
        v = str(self)
        if isinstance(v, I18nString):
            self.add_values.append(other)
            return self
        return v.__add__(other)

    def __radd__(self, other):
        v = str(self)
        if isinstance(v, I18nString):
            self.radd_values.append(other)
            return self
        return other.__add__(v)

    def __hash__(self) -> int:
        return super().__hash__()

    def format(self, *args, **kwargs) -> str:
        v = str(self)
        if isinstance(v, I18nString):
            return super().format(*args, **kwargs)
        return v.format(*args, **kwargs)

    def unwrap(self):
        return super().__str__()

    @staticmethod
    def unwrap_strings(obj):
        """Unwrap all keys in I18nStrings in the object"""
        if isinstance(obj, I18nString):
            yield obj.unwrap()
            for v in obj.add_values:
                yield from I18nString.unwrap_strings(v)
            for v in obj.radd_values:
                yield from I18nString.unwrap_strings(v)
            return
        yield obj


def gettext(key: str):
    """Wrapper text string to return I18nString
    :param key: The key of the I18nString
    """
    return I18nString(key)


def iter_i18n_choices(choices):
    """Iterate all I18nStrings in the choice, returns the indices of the I18nStrings"""
    if not isinstance(choices, list) or len(choices) == 0:
        return

    if isinstance(choices[0], tuple):
        for i, (k, v) in enumerate(choices):
            if isinstance(k, I18nString):
                yield i

    else:
        for i, v in enumerate(choices):
            if isinstance(v, I18nString):
                yield i


def iter_i18n_fields(component: gr.components.Component):
    """Iterate all I18nStrings in the component"""
    for name, value in inspect.getmembers(component):
        if name == "value" and hasattr(component, "choices"):
            # for those components with choices, the value will be kept as is
            continue
        if isinstance(value, I18nString):
            yield name
        elif name == "choices" and any(iter_i18n_choices(value)):
            yield name


def iter_i18n_components(block: Block):
    """Iterate all I18nStrings in the block"""
    if isinstance(block, BlockContext):
        for component in block.children:
            for c in iter_i18n_components(component):
                yield c

    if any(iter_i18n_fields(block)):
        yield block


def has_new_i18n_fields(block: Block, langs=["en"], existing_translation={}):
    """Check if there are new I18nStrings in the block
    :param block: The block to check
    :param langs: The languages to check
    :param existing_translation: The existing translation dictionary
    :return: True if there are new I18nStrings, False otherwise
    """
    components = list(iter_i18n_components(block))

    for lang in langs:
        for component in components:
            for field in iter_i18n_fields(component):
                if field == "choices":
                    for idx in iter_i18n_choices(component.choices):
                        if isinstance(component.choices[idx], tuple):
                            value = component.choices[idx][0]
                        else:
                            value = component.choices[idx]
                        if value not in existing_translation.get(lang, {}):
                            return True
                else:
                    value = getattr(component, field)
                    if value not in existing_translation.get(lang, {}):
                        return True

    return False


def dump_blocks(block: Block, langs=["en"], include_translations={}):
    """Dump all I18nStrings in the block to a dictionary
    :param block: The block to dump
    :param langs: The languages to dump
    :param include_translations: The existing translation dictionary
    :return: The dumped dictionary
    """
    components = list(iter_i18n_components(block))

    def translate(lang, key):
        return include_translations.get(lang, {}).get(key, key)

    ret = {}

    for lang in langs:
        ret[lang] = {}
        for component in components:
            for field in iter_i18n_fields(component):
                if field == "choices":
                    for idx in iter_i18n_choices(component.choices):
                        if isinstance(component.choices[idx], tuple):
                            value = component.choices[idx][0]
                        else:
                            value = component.choices[idx]
                        for key in I18nString.unwrap_strings(value):
                            ret[lang][key] = translate(lang, key)
                else:
                    value = getattr(component, field)
                    for key in I18nString.unwrap_strings(value):
                        ret[lang][key] = translate(lang, key)

    return ret


def translate_blocks(
    block: gr.Blocks = None,
    translation={},
    lang: gr.components.Component = None,
    persistant=False,
):
    """Translate all I18nStrings in the block
    :param block: The block to translate, default is the root block
    :param translation: The translation dictionary
    :param lang: The language component to change the language
    :param persistant: Whether to persist the language
    """
    if block is None:
        block = Context.root_block

    """Translate all I18nStrings in the block"""
    if not isinstance(block, gr.Blocks):
        raise ValueError("block must be an instance of gradio.Blocks")

    components = list(iter_i18n_components(block))
    TranslateContext.add_translation(translation)

    hidden = gr.HTML(
        value="""<style>
                    gradio-app {
                        visibility: hidden;
                    }
                     </style>"""
    )

    if persistant:
        try:
            from gradio import BrowserState
        except ImportError:
            raise ValueError("gradio>=5.6.0 is required for persistant language")

    def on_lang_change(request: gr.Request, lang: str, saved_lang: str):
        if not lang:
            if saved_lang:
                lang = saved_lang
            else:
                lang = get_lang_from_request(request)

        outputs = [lang, lang, ""]
        # if TranslateContext.get_current_language(request) == lang:
        #     for _ in components:
        #         outputs.append(gr.update())
        #     return outputs

        TranslateContext.set_current_language(request, lang)

        for component in components:
            fields = list(iter_i18n_fields(component))
            if component == lang and "value" in fields:
                raise ValueError("'lang' component can't has I18nStrings as value")

            modified = {}

            for field in fields:
                if field == "choices":
                    choices = component.choices.copy()
                    for idx in iter_i18n_choices(choices):
                        if isinstance(choices[idx], tuple):
                            k, v = choices[idx]
                            # We don't need to translate the value
                            choices[idx] = (str(k), next(I18nString.unwrap_strings(v)))
                        else:
                            v = choices[idx]
                            choices[idx] = (str(v), next(I18nString.unwrap_strings(v)))
                    modified[field] = choices
                else:
                    modified[field] = str(getattr(component, field))

            new_comp = gr.update(**modified)
            outputs.append(new_comp)

        if len(outputs) == 1:
            return outputs[0]

        return outputs

    if lang is None:
        lang = gr.State()

    if persistant:
        saved_lang = gr.BrowserState(storage_key="lang")
    else:
        saved_lang = gr.State()

    gr.on(
        [block.load, lang.change],
        on_lang_change,
        inputs=[lang, saved_lang],
        outputs=[lang, saved_lang, hidden] + components,
    )


@contextmanager
def Translate(
    translation,
    lang: gr.components.Component = None,
    placeholder_langs=[],
    persistant=False,
):
    """Translate all I18nStrings in the block
    :param translation: The translation dictionary or file path
    :param lang: The language component to change the language
    :param placeholder_langs: The placeholder languages to create a new translation file if translation is a file path
    :param persistant: Whether to persist the language
    :return: The language component
    """
    if lang is None:
        lang = gr.State()
    yield lang

    if isinstance(translation, dict):
        # Static translation
        translation_dict = translation
        pass
    elif isinstance(translation, str):
        if os.path.exists(translation):
            # Regard as a file path
            with open(translation, "r", encoding="utf-8") as f:  # Force utf-8 encoding
                if translation.endswith(".json"):
                    translation_dict = json.load(f)
                elif translation.endswith(".yaml"):
                    translation_dict = yaml.safe_load(f)
                else:
                    raise ValueError("Unsupported file format")
        else:
            translation_dict = {}
    else:
        raise ValueError("Unsupported translation type")

    block = Context.block
    translate_blocks(
        block=block, translation=translation_dict, lang=lang, persistant=persistant
    )

    if (
        placeholder_langs
        and isinstance(translation, str)
        and has_new_i18n_fields(
            block, langs=placeholder_langs, existing_translation=translation_dict
        )
    ):
        merged = dump_blocks(
            block, langs=placeholder_langs, include_translations=translation_dict
        )

        with open(translation, "w") as f:
            if translation.endswith(".json"):
                json.dump(merged, f, indent=2, ensure_ascii=False)
            elif translation.endswith(".yaml"):
                yaml.dump(merged, f, allow_unicode=True, sort_keys=False)
