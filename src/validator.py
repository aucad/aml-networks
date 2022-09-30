from __future__ import annotations

from collections import namedtuple


class NetworkProto:
    """Represents validatable instance"""

    def __init__(self, name: str, kind: namedtuple, **kwargs):
        self.name = name
        self.attributes = kind._fields or []
        defaults = dict([(a, 0) for a in self.attributes if a not in kwargs])
        self.record: namedtuple = kind(**defaults, **kwargs)

    @staticmethod
    def ensure_attrs():
        return {}

    @staticmethod
    def key_vals_dict(names, values):
        return dict([(a, b) for (a, b) in zip(names.split(' '), values)])

    def validation_model(self, attrs, kwargs) -> namedtuple:
        """validator model for current dataset"""
        req_keys = list(self.ensure_attrs().keys())
        instance_keys = list(kwargs.keys())
        attr_names = list(set((attrs or []) + req_keys + instance_keys))
        return namedtuple('xyz', ",".join(attr_names))

    @property
    def values(self):
        return [getattr(self.record, a) for a in self.attributes]

    @property
    def validation_rules(self):
        return False

    def check(self) -> bool:
        return self.validation_rules


class NbTCP(NetworkProto):
    def __init__(self, attrs=None, **kwargs):
        # noinspection PyTypeChecker
        super().__init__(
            'tcp', self.validation_model(attrs, kwargs), **kwargs)

    @staticmethod
    def ensure_attrs():
        """If values are undefined for some records, these are the defaults."""
        names = 'swin dwin synack ackdat tcprtt dbytes Dload dttl Djit Dpkts'
        return NetworkProto.key_vals_dict(names, [255, 255] + [0] * 8)

    @property
    def validation_rules(self):
        if self.record.swin != 255 or self.record.dwin != 255:
            return False
        if round(self.record.synack + self.record.ackdat, 3) != \
                round(self.record.tcprtt, 3):
            return False
        if self.record.dbytes == 0 and (
                self.record.Dload != 0 or self.record.dttl != 0 or
                self.record.Djit != 0 or self.record.Dpkts != 0):
            return False
        return True


class NbUDP(NetworkProto):
    def __init__(self, attrs=None, **kwargs):
        # noinspection PyTypeChecker
        super().__init__(
            'udp', self.validation_model(attrs, kwargs), **kwargs)

    @staticmethod
    def ensure_attrs():
        """If values are undefined for some records, these are the defaults."""
        names = 'smeansz Spkts sbytes dmeansz Dpkts dbytes'
        return NetworkProto.key_vals_dict(names, [0] * 6)

    @property
    def validation_rules(self):
        return self.record.smeansz * self.record.Spkts == \
               self.record.sbytes \
               and self.record.dmeansz * self.record.Dpkts == \
               self.record.dbytes


class Validator:
    NB15 = 'NB15'
    IOT23 = 'IOT23'

    @staticmethod
    def validate(instance: NetworkProto):
        return instance.check()
