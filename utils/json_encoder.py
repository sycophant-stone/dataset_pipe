import json


class JsonEncoderInterface:
    def to_json(self):
        raise Exception('need implement')


class JsonEncoder(json.JSONEncoder):
    def default(self, o):
        if issubclass(o.__class__, JsonEncoderInterface):
            return o.to_json()
        return json.JSONEncoder.default(self, o)
