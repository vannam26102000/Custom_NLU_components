import json
import re
import logging

from rasa.shared.core.domain import Domain

from config.hyper_params import (
    DEFAULT_NAME_INTENT_FALLBACK,
    DEFAULT_ENCODING,
    MAX_INTENTS
)

from config.config_path import (
    PATH_CONFIG_REGEX,
    PATH_CONFIG_FALLBACK_WORDS,
    PATH_DOMAIN
)
from core.policies.graph_loader import GraphLoader

logger = logging.getLogger(__name__)


class RegexIntentClassifier(object):
    """Module classification using lib re(Regex) with config in file: `path_regex_file`"""

    def __init__(self, path_regex_file=PATH_CONFIG_REGEX):
        with open(path_regex_file, encoding=DEFAULT_ENCODING) as regex_file:
            self.regex_config = json.load(regex_file)
        with open(PATH_CONFIG_FALLBACK_WORDS, encoding=DEFAULT_ENCODING) as fallback_words_file:
            self.fallback_words = [line.replace("\n", "").strip() for line in fallback_words_file.readlines()]
            if "" in self.fallback_words:
                self.fallback_words.remove("")

        if PATH_DOMAIN is None:
            logger.info("path domain not exits")
        else:
            self.domain_intents = Domain.load(PATH_DOMAIN).intents

        for intent_name in self.domain_intents:
            self.regex_config.append({
                "name": intent_name,
                "regex": [
                    f"^{intent_name}$",
                ]
            })
        super(RegexIntentClassifier, self).__init__()

    def predict(self, message):
        if message.startswith("_"):
            message, _ = GraphLoader.extract_edge(message[1:])
            if not message:
                return None
        for intent in self.regex_config:
            name_intent = intent.get("name")
            for re_str in intent.get("regex"):
                if re.match(".*{}.*".format(re_str), message, re.IGNORECASE):
                    return name_intent
        if message in self.fallback_words:
            return DEFAULT_NAME_INTENT_FALLBACK
        return None

    def predict_name_intent(self, message, name_intent):
        if message.startswith("_"):
            message, _ = GraphLoader.extract_edge(message[1:])
            if not message:
                return None, MAX_INTENTS
        for idx, intent in enumerate(self.regex_config):
            if name_intent == intent.get("name"):
                for re_str in intent.get("regex"):
                    if re.match(".*{}.*".format(re_str), message, re.IGNORECASE):
                        return name_intent, idx
        if name_intent == DEFAULT_NAME_INTENT_FALLBACK:
            if message in self.fallback_words:
                return name_intent, 0
        return None, MAX_INTENTS
