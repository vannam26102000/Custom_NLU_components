import re
from builtins import enumerate


def better(x, y):
    """
    Entity selection function
    Default: The choice to appear position and length priority
    Custom: The choice depends on the type of entity
    """

    # Default: position ahead & longer
    if x["end"] - 1 < y["start"]:
        return x
    if y["end"] - 1 < x["start"]:
        return y
    if len(x["value"]) > len(y["value"]):
        return x
    if len(y["value"]) > len(x["value"]):
        return y
    return y


def remove_overlap(entities):
    result = []
    for idx1, e1 in enumerate(entities):
        is_inside = False
        for idx2, e2 in enumerate(entities):
            if idx1 != idx2 and e1.get("start") >= e2.get("start") and e1.get("end") <= e2.get("end"):
                is_inside = True
                break
        if not is_inside:
            result.append(e1)
    return result


def check_inside_ignore_phrase(e1, phrases):
    for e2 in phrases:
        if e1.get("start") >= e2.get("start") and e1.get("end") <= e2.get("end"):
            return True
    return False


def remove_ignore_phrase(entities, phrases):
    result = []
    for e1 in entities:
        if check_inside_ignore_phrase(e1, phrases):
            is_inside = True
        else:
            is_inside = False
        if not is_inside:
            result.append(e1)
    return result


def extract_entities(message, rule):
    """Extract entities of the given type from the given user message."""
    entities = []
    for pattern in rule:
        try:
            pattern = re.compile(pattern)
            for match in pattern.finditer(message.lower()):
                entity = {
                    "start": match.start(),
                    "end": match.end(),
                    "value": match.group(),
                    "confidence": 1.0,
                }
                entities.append(entity)
        except:
            pass
    return entities
