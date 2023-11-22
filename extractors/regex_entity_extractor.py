import logging
import os
import re
from typing import Any, Dict, List, Optional, Text

import rasa.shared.utils.io
import rasa.nlu.utils.pattern_utils as pattern_utils
from core.nlu.extractors.regex_selector import remove_overlap
from rasa.nlu.model import Metadata
from rasa.nlu.config import RasaNLUModelConfig
from rasa.shared.nlu.training_data.training_data import TrainingData
from rasa.shared.nlu.training_data.message import Message
from rasa.shared.nlu.constants import (
    ENTITIES,
    ENTITY_ATTRIBUTE_VALUE,
    ENTITY_ATTRIBUTE_START,
    ENTITY_ATTRIBUTE_END,
    TEXT,
    ENTITY_ATTRIBUTE_TYPE,
)
from rasa.nlu.extractors.extractor import EntityExtractor

import core.nlu.extractors.regex_selector as regex_selector
from core.policies.graph_loader import GraphLoader

logger = logging.getLogger(__name__)


class RegexEntityExtractor(EntityExtractor):
    """Searches for entities in the user's message using the lookup tables and regexes
    defined in the training data."""

    defaults = {
        # text will be processed with case insensitive as default
        "case_sensitive": False,
        # use lookup tables to extract entities
        "use_lookup_tables": True,
        # use regexes to extract entities
        "use_regexes": True,
        # use match word boundaries for lookup table
        "use_word_boundaries": True,
    }

    def __init__(
            self,
            component_config: Optional[Dict[Text, Any]] = None,
            patterns: Optional[List[Dict[Text, Text]]] = None,
    ):
        """Extracts entities using the lookup tables and/or regexes defined."""
        super(RegexEntityExtractor, self).__init__(component_config)

        self.case_sensitive = self.component_config["case_sensitive"]
        self.patterns = patterns or []

    def train(
            self,
            training_data: TrainingData,
            config: Optional[RasaNLUModelConfig] = None,
            **kwargs: Any,
    ) -> None:
        self.patterns = pattern_utils.extract_patterns(
            training_data,
            use_lookup_tables=self.component_config["use_lookup_tables"],
            use_regexes=self.component_config["use_regexes"],
            use_only_entities=False,
            use_word_boundaries=self.component_config["use_word_boundaries"],
        )

        if not self.patterns:
            rasa.shared.utils.io.raise_warning(
                "No lookup tables or regexes defined in the training data that have "
                "a name equal to any entity in the training data. In order for this "
                "component to work you need to define valid lookup tables or regexes "
                "in the training data."
            )

    def process(self, message: Message, **kwargs: Any) -> None:
        if not self.patterns:
            return

        extracted_entities = self._extract_entities(message)
        extracted_entities = self.add_extractor_name(extracted_entities)

        message.set(
            ENTITIES, message.get(ENTITIES, []) + extracted_entities, add_to_output=True
        )

    def _extract_entities(self, message: Message) -> List[Dict[Text, Any]]:
        """Extract entities of the given type from the given user message."""
        entities = {}
        message = message.get(TEXT)
        if message.startswith("_"):
            extracted = []
            message = message[1:]
            _, entities_name = GraphLoader.extract_edge(message)
            if entities_name:
                for name in entities_name:
                    entity = {
                        "start": 0,
                        "end": 0,
                        "value": "test",
                        "confidence": 1.0,
                        "entity": name,
                    }
                    extracted.append(entity)
            return extracted

        for d in self.patterns:
            pattern = re.compile(d['pattern'])
            for match in pattern.finditer(message.lower()):
                name = d['name'].strip()
                entity = {
                    "start": match.start(),
                    "end": match.end(),
                    "value": match.group(),
                    "confidence": 1.0,
                    "entity": name,
                }
                if name not in entities:
                    entities[name] = []
                entities[name].append(entity)
        extracted = []
        for name in entities:
            list_e = entities[name].copy()
            choose = list_e[0]
            for e in list_e[1:]:
                choose = regex_selector.better(e, choose)
            extracted.append(choose)
        extracted = remove_overlap(extracted)
        return extracted

    @classmethod
    def load(
            cls,
            meta: Dict[Text, Any],
            model_dir: Text,
            model_metadata: Optional[Metadata] = None,
            cached_component: Optional["RegexEntityExtractor"] = None,
            **kwargs: Any,
    ) -> "RegexEntityExtractor":
        """Loads trained component (see parent class for full docstring)."""
        file_name = meta.get("file")
        regex_file = os.path.join(model_dir, file_name)

        if os.path.exists(regex_file):
            patterns = rasa.shared.utils.io.read_json_file(regex_file)
            return cls(meta, patterns=patterns)

        return cls(meta)

    def persist(self, file_name: Text, model_dir: Text) -> Optional[Dict[Text, Any]]:
        """Persist this model into the passed directory.

        Return the metadata necessary to load the model again.
        """
        file_name = f"{file_name}.json"
        regex_file = os.path.join(model_dir, file_name)
        rasa.shared.utils.io.dump_obj_as_json_to_file(regex_file, self.patterns)

        return {"file": file_name}
