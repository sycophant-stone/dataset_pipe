import yaml
import utils.json_encoder as json_interface


def concat(v):
    res = ''
    for node in v:
        if isinstance(node.value, str):
            res += node.value
        else:
            res += concat(node.value)
    return res


class ConcatTag(yaml.YAMLObject, json_interface.JsonEncoderInterface):
    yaml_tag = u'!CONCAT'

    def __init__(self, str_list):
        self.str_list = str_list

    def __repr__(self):
        return concat(self.str_list)

    @classmethod
    def from_yaml(cls, loader, node):
        return concat(node.value)

    @classmethod
    def to_yaml(cls, dumper, data):
        return dumper.represent_scalar(cls.yaml_tag, concat(data.str_list))

    def to_json(self):
        return concat(self.str_list)
