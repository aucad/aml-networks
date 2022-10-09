from __future__ import annotations

from collections import namedtuple


class NetworkProto:
    """Represents validatable instance"""

    def __init__(self, name: str, kind: namedtuple, **kwargs):
        self.name = name
        # noinspection PyProtectedMember
        self.attributes = kind._fields
        defaults = [(a, 0) for a in self.attributes if a not in kwargs]
        self.record: namedtuple = kind(**dict(defaults), **kwargs)

    @staticmethod
    def ensure_attrs():
        return {}

    @staticmethod
    def kv_dict(names, values):
        names_ = names.split(' ')
        return dict([(a, b) for (a, b) in zip(names_, values)])

    def v_model(self, attrs, kwargs) -> namedtuple:
        """validator model for current dataset"""
        req_keys = list(self.ensure_attrs().keys())
        instance_keys = list(kwargs.keys())
        attr_names = list(set((attrs or []) + req_keys + instance_keys))
        return namedtuple('xyz', ",".join(attr_names))

    @property
    def values(self):
        return [getattr(self.record, a) for a in self.attributes]

    @staticmethod
    def validate(record):
        return False

    def check(self) -> bool:
        return self.validate(self.record)


# noinspection PyTypeChecker
class NbTCP(NetworkProto):
    def __init__(self, attrs: namedtuple = None, **kwargs):
        super().__init__('tcp', self.v_model(attrs, kwargs), **kwargs)

    @staticmethod
    def ensure_attrs():
        """Defaults for undefined attributes."""
        names = 'swin dwin ltime ' \
                'synack ackdat tcprtt dbytes Dload ' \
                'dttl Djit Dpkts dur state_INT stime'
        return NetworkProto.kv_dict(
            names, [255, 255, 1] + [0] * 11)

    @staticmethod
    def validate(record):
        """validation criteria for UNSW-NB15 TCP record"""
        if record.swin != 255 or record.dwin != 255:
            return False
        # synack + ackdat = tcprtt
        if round(record.synack + record.ackdat, 3) != \
                round(record.tcprtt, 3):
            return False
        if record.dur > 0:
            # if dur > 0 in state INT: dbytes = 0
            if record.state_INT == 1:
                if record.dbytes != 0:
                    return False
            # if dur > 0 then dbytes > 0
            elif not record.dbytes > 0:
                return False
        # if dur = 0 then dbytes = 0
        if record.dur == 0 and record.dbytes != 0:
            return False
        # if dbytes = 0 then everything destinations related is 0
        if record.dbytes == 0 and (
                record.Dload != 0 or record.dttl != 0 or
                record.Djit != 0 or record.Dpkts != 0):
            return False
        # stime + dur + (some small value) = ltime
        # can have stime==ltime
        if record.ltime < record.stime + record.dur:
            return False
        return True


# noinspection PyTypeChecker
class NbUDP(NetworkProto):
    def __init__(self, attrs=None, **kwargs):
        super().__init__('udp', self.v_model(attrs, kwargs), **kwargs)

    @staticmethod
    def ensure_attrs():
        """Defaults for undefined attributes."""
        names = 'swin dwin stcpb dtcpb synack ' \
                'ackdat tcprtt smeansz Spkts sbytes ' \
                'dmeansz Dpkts dbytes sjit'
        return NetworkProto.kv_dict(names, [0] * 14)

    @staticmethod
    def validate(record):
        # TCP related fields must be 0
        if not (record.swin == 0 and record.dwin == 0
                and record.stcpb == 0 and record.dtcpb == 0
                and record.synack == 0 and record.ackdat == 0
                and record.tcprtt == 0):
            return False
        # Smeansz * Spkts = sbytes and Dmeansz * Dpkts = dpytes
        if (record.smeansz * record.Spkts != record.sbytes) \
                or (record.dmeansz * record.Dpkts != record.dbytes):
            return False
        # if sjit = 0 then (Smeansz * 8)/sload + something small = dur
        if record.sjit == 0 and record.Sload != 0 and \
                record.dur < (record.smeansz * 8 / record.Sload):
            return False
        return True


# noinspection PyTypeChecker
class NbOther(NetworkProto):
    def __init__(self, attrs=None, **kwargs):
        super().__init__('other', self.v_model(attrs, kwargs), **kwargs)

    @staticmethod
    def ensure_attrs():
        """Defaults for undefined attributes."""
        names = 'swin dwin stcpb dtcpb synack ackdat tcprtt'
        return NetworkProto.kv_dict(names, [0] * 7)

    @staticmethod
    def validate(record):
        # TCP related fields must be 0
        return (record.swin == 0 and record.dwin == 0
                and record.stcpb == 0 and record.dtcpb == 0
                and record.synack == 0 and record.ackdat == 0
                and record.tcprtt == 0)


# noinspection PyTypeChecker
class IotTCP(NetworkProto):
    def __init__(self, attrs=None, **kwargs):
        super().__init__('udp', self.v_model(attrs, kwargs), **kwargs)

    @staticmethod
    def ensure_attrs():
        return NetworkProto.kv_dict('', [])

    @staticmethod
    def validate(record):
        return True


# noinspection PyTypeChecker
class IotUDP(NetworkProto):
    def __init__(self, attrs=None, **kwargs):
        super().__init__('udp', self.v_model(attrs, kwargs), **kwargs)

    @staticmethod
    def ensure_attrs():
        return NetworkProto.kv_dict('', [])

    @staticmethod
    def validate(record):
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
