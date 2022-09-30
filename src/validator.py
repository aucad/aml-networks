from __future__ import annotations

from collections import namedtuple


class NetworkProto:
    """Represents validatable instance"""

    def __init__(self, name: str, kind: namedtuple, **kwargs):
        self.name = name
        # noinspection PyProtectedMember
        self.attributes = kind._fields
        defaults = dict([(a, 0) for a in self.attributes if a not in kwargs])
        self.record: namedtuple = kind(**defaults, **kwargs)

    @staticmethod
    def ensure_attrs():
        return {}

    @staticmethod
    def kv_dict(names, values):
        return dict([(a, b) for (a, b) in zip(names.split(' '), values)])

    def v_model(self, attrs, kwargs) -> namedtuple:
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


# noinspection PyTypeChecker
class NbTCP(NetworkProto):
    def __init__(self, attrs: namedtuple = None, **kwargs):
        super().__init__('tcp', self.v_model(attrs, kwargs), **kwargs)

    @staticmethod
    def ensure_attrs():
        """If values are undefined for some records, these are the defaults."""
        names = 'swin dwin synack ackdat tcprtt dbytes Dload dttl Djit Dpkts'
        return NetworkProto.kv_dict(names, [255, 255] + [0] * 8)

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


# noinspection PyTypeChecker
class NbUDP(NetworkProto):
    def __init__(self, attrs=None, **kwargs):
        super().__init__('udp', self.v_model(attrs, kwargs), **kwargs)

    @staticmethod
    def ensure_attrs():
        """If values are undefined for some records, these are the defaults."""
        names = 'smeansz Spkts sbytes dmeansz Dpkts dbytes'
        return NetworkProto.kv_dict(names, [0] * 6)

    @property
    def validation_rules(self):
        return self.record.smeansz * self.record.Spkts == \
               self.record.sbytes \
               and self.record.dmeansz * self.record.Dpkts == \
               self.record.dbytes


# noinspection PyTypeChecker
class IotTCP(NetworkProto):
    def __init__(self, attrs=None, **kwargs):
        super().__init__('udp', self.v_model(attrs, kwargs), **kwargs)

    @staticmethod
    def ensure_attrs():
        return NetworkProto.kv_dict('', [])

    @property
    def validation_rules(self):
        return True


# noinspection PyTypeChecker
class IotUDP(NetworkProto):
    def __init__(self, attrs=None, **kwargs):
        super().__init__('udp', self.v_model(attrs, kwargs), **kwargs)

    @staticmethod
    def ensure_attrs():
        return NetworkProto.kv_dict('', [])

    @property
    def validation_rules(self):
        return True

    # TODO : implement these plus (?) rules from here
    # https://drive.google.com/drive/folders/1q5jfv0N-zWWi7e4qM3VSjrdIlN5QEMqd
    #
    #     # 0: valid 1: invalid
    #     print(att_dict)
    #     proto_sum = int(att_dict['proto_imcp']) +
    #     int(att_dict['proto_tcp']) + int(att_dict['proto_udp'])
    #     if proto_sum != 1:
    #         return 1
    #     state_sum = int(att_dict['conn_state_OTH']) +
    #     int(att_dict['conn_state_REJ']) +
    #     int(att_dict['conn_state_RSTO']) +
    #     int(att_dict['conn_state_RSTR']) +
    #     int(att_dict['conn_state_RSTRH']) +
    #     int(att_dict['conn_state_S0']) +
    #     int(att_dict['conn_state_S1']) +
    #     int(att_dict['conn_state_S2']) +
    #     int(att_dict['conn_state_SF']) +
    #     int(att_dict['conn_state_SH']) +
    #     int(att_dict['conn_state_SHR'])
    #
    #     if state_sum != 1:
    #         return 1
    #
    #     if int(att_dict['resp_pkts']) == 0:
    #         if int(att_dict['resp_ip_bytes']) != 0:
    #             return 1
    #
    #     if int(att_dict['proto_tcp']) == 1:
    #         if int(att_dict['orig_pkts']) < int(att_dict['resp_pkts']):
    #             return 1
    #
    #     if int(att_dict['proto_tcp']) == 1:
    #         if int(att_dict['orig_ip_bytes']) <
    #         int(att_dict['resp_ip_bytes']):
    #             if int(att_dict['conn_state_SF']) != 1:
    #                 return 1
    #
    #     if int(att_dict['proto_udp']) == 1:
    #         if int(att_dict['orig_ip_bytes']) <
    #         int(att_dict['resp_ip_bytes']):
    #             if int(att_dict['conn_state_SF']) != 1:
    #                 return 1
    #
    #     return 0


class Validator:
    NB15 = 'NB15'
    IOT23 = 'IOT23'

    @staticmethod
    def validate(instance: NetworkProto):
        return instance.check()
