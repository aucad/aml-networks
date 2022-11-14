from __future__ import annotations

import os
from collections import namedtuple
from pathlib import Path
from typing import Tuple, Union, List

from src import Show, utility


class Protocol:
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
    def validate(record) -> Tuple[bool, Union[str, None]]:
        return True, None

    def check(self) -> Tuple[bool, Union[str, None]]:
        return self.validate(self.record)


# noinspection PyTypeChecker
class NbTCP(Protocol):
    def __init__(self, attrs: namedtuple = None, **kwargs):
        super().__init__('tcp', self.v_model(attrs, kwargs), **kwargs)

    @staticmethod
    def ensure_attrs():
        """Defaults for undefined attributes."""
        names = 'swin dwin ltime ' \
                'synack ackdat tcprtt dbytes Dload ' \
                'dttl Djit Dpkts dur state_INT stime state_FIN'
        return Protocol.kv_dict(
            names, [255, 255, 1] + [0] * 12)

    @staticmethod
    def validate(record) -> Tuple[bool, Union[str, None]]:
        if record.swin != 255 or record.dwin != 255:
            # OK if dbytes==0 then dwin==0 or state is FIN
            if not ((record.dbytes == 0 and record.dwin == 0)
                    or record.state_FIN == 1):
                return False, "invalid swin/dwin"
        # synack + ackdat = tcprtt
        if round(record.synack + record.ackdat, 3) != \
                round(record.tcprtt, 3):
            return False, "synack+ackdat != tcprtt"
        if record.state_FIN != 1:
            # if dur > 0 in state INT: dbytes = 0
            if record.dur > 0 and record.state_INT == 1 and \
                    record.dbytes != 0:
                return False, "dbytes nonzero when INT"
            # if dur = 0 then dbytes = 0
            if record.dur == 0 and record.dbytes != 0:
                return False, "dbytes nonzero when dur is 0"
        # if dbytes = 0 then everything destinations related is 0
        if record.dbytes == 0 and (
                record.Dload != 0 or record.dttl != 0 or
                record.Djit != 0 or record.Dpkts != 0):
            return False, "dest attrs nonzero"
        # stime + dur + (some small value) = ltime
        # can have stime == ltime
        if record.ltime < record.stime:
            return False, "ltime < stime"
        return True, None


# noinspection PyTypeChecker
class NbUDP(Protocol):
    def __init__(self, attrs=None, **kwargs):
        super().__init__('udp', self.v_model(attrs, kwargs), **kwargs)

    @staticmethod
    def ensure_attrs():
        """Defaults for undefined attributes."""
        names = 'smeansz Spkts sbytes dmeansz Dpkts dbytes ' \
                'swin dwin stcpb dtcpb synack ' \
                'ackdat tcprtt sjit Sload'
        return Protocol.kv_dict(names, [0] * 15)

    @staticmethod
    def validate(record) -> Tuple[bool, Union[str, None]]:
        # TCP related fields must be 0
        if not (record.swin == 0 and record.dwin == 0
                and record.stcpb == 0 and record.dtcpb == 0
                and record.synack == 0 and record.ackdat == 0
                and record.tcprtt == 0):
            return False, "nonzero tcp fields"
        # ensure all 3 are included as dataset features
        if record.smeansz > 0 and record.Spkts > 0 \
                and record.sbytes >= 0:
            # Smeansz * Spkts = sbytes
            if record.smeansz * record.Spkts != record.sbytes:
                return False, "invalid smeansz"
        # ensure all 3 are included as dataset features
        if record.dmeansz > 0 and record.Dpkts > 0 \
                and record.dbytes > 0:
            # Dmeansz * Dpkts = dpytes
            if record.dmeansz * record.Dpkts != record.dbytes:
                return False, "invalid dmeansz"
        # if sjit = 0 then (Smeansz * 8)/sload + something small = dur
        if record.sjit == 0 and record.Sload != 0 and \
                record.dur < (record.smeansz * 8 / record.Sload):
            return False, "invalid dur for sjit=0"
        return True, None


# noinspection PyTypeChecker
class NbOther(Protocol):
    def __init__(self, attrs=None, **kwargs):
        super().__init__('other', self.v_model(attrs, kwargs), **kwargs)

    @staticmethod
    def ensure_attrs():
        """Defaults for undefined attributes."""
        names = 'swin dwin stcpb dtcpb synack ackdat tcprtt'
        return Protocol.kv_dict(names, [0] * 7)

    @staticmethod
    def validate(record) -> Tuple[bool, Union[str, None]]:
        # TCP related fields must be 0
        if not (record.swin == 0 and record.dwin == 0
                and record.stcpb == 0 and record.dtcpb == 0
                and record.synack == 0 and record.ackdat == 0
                and record.tcprtt == 0):
            return False, "nonzero tcp fields"
        return True, None


# noinspection PyTypeChecker
class IotTCP(Protocol):
    def __init__(self, attrs=None, **kwargs):
        super().__init__('tcp', self.v_model(attrs, kwargs), **kwargs)

    @staticmethod
    def ensure_attrs():
        names = 'conn_state_S0 conn_state_SF conn_state_OTH ' \
                'conn_state_REJ history_ShADadFf history_DdA ' \
                'resp_ip_bytes resp_pkts orig_ip_bytes orig_pkts'
        return Protocol.kv_dict(names, [0] * 10)

    @staticmethod
    def validate(record) -> Tuple[bool, Union[str, None]]:
        # if record.orig_pkts < record.resp_pkts:
        #     return False, "ori pkts < resp pkts"
        # in S0 resp_pkts = 0 and resp_ip_bytes = 0
        if record.conn_state_S0 == 1:
            if record.resp_pkts != 0 or record.resp_ip_bytes != 0:
                return False, "S0 packets or bytes nonzero"
        # number of packets would be smaller than the bytes sent,
        if record.orig_pkts > record.orig_ip_bytes:
            return False, "ori packets > bytes"
        # this is also true for the receiving
        if record.resp_pkts > record.resp_ip_bytes:
            return False, "resp packets > bytes"
        # if conn state REJ then orig_ip_bytes=0 and resp_ip_bytes=0
        if record.conn_state_REJ == 1:
            if record.orig_ip_bytes != 0 or record.resp_ip_bytes != 0:
                return False, "REJ state bytes is 0"
        return True, None


# noinspection PyTypeChecker
class IotOther(Protocol):
    def __init__(self, attrs=None, **kwargs):
        super().__init__('other', self.v_model(attrs, kwargs), **kwargs)

    @staticmethod
    def validate(record) -> Tuple[bool, Union[str, None]]:
        # in S0 resp_pkts = 0 and resp_ip_bytes = 0
        if record.conn_state_S0 == 1:
            if record.resp_pkts != 0 or record.resp_ip_bytes != 0:
                return False, "S0: packets/bytes not 0"
        # number of packets would be smaller than the bytes sent,
        if record.orig_pkts > record.orig_ip_bytes:
            return False, "ori packets > bytes"
        # this is also true for the receiving
        if record.resp_pkts > record.resp_ip_bytes:
            return False, "resp packets > bytes"
        return True, None


# noinspection PyTypeChecker
class IotUDP(Protocol):
    def __init__(self, attrs=None, **kwargs):
        super().__init__('udp', self.v_model(attrs, kwargs), **kwargs)

    @staticmethod
    def ensure_attrs():
        names = 'conn_state_S0 conn_state_SF history_Dd ' \
                'resp_ip_bytes resp_pkts orig_ip_bytes orig_pkts'
        return Protocol.kv_dict(names, [0] * 7)

    @staticmethod
    def validate(record) -> Tuple[bool, Union[str, None]]:
        # orig_pkts is always greater or equal to resp_pkts
        # unless history is Dd and state is SF.
        if record.orig_pkts < record.resp_pkts:
            if not (record.history_Dd == 1 and
                    record.conn_state_SF == 1):
                return False, "history/conn_state mismatch"
        if record.orig_pkts >= record.resp_pkts:
            # orig_ip_bytes is also >= resp_ip_bytes unless state is SF
            if not (record.orig_ip_bytes >= record.resp_ip_bytes or
                    record.conn_state_SF == 1):
                return False, "packet-bytes mismatch"
        return IotOther.validate(record)


# noinspection PyTypeChecker
class IotICMP(Protocol):
    def __init__(self, attrs=None, **kwargs):
        super().__init__('icmp', self.v_model(attrs, kwargs), **kwargs)

    @staticmethod
    def ensure_attrs():
        names = 'conn_state_S0 resp_ip_bytes resp_pkts ' \
                'orig_ip_bytes orig_pkts'
        return Protocol.kv_dict(names, [0] * 5)

    @staticmethod
    def validate(record) -> Tuple[bool, Union[str, None]]:
        return IotOther.validate(record)


class Validator:
    # validator kinds
    NB15 = 'NB15'
    IOT23 = 'IOT23'

    @staticmethod
    def determine_proto_val(attr, record, i):
        try:
            return attr, int(record[i])
        except ValueError:
            if str(record[i]) in ['tcp', 'udp', 'icmp']:
                return str(record[i]), 1
        return attr, 0

    @staticmethod
    def determine_proto(validator_kind, attrs, record) \
            -> Union[Protocol, None]:
        """Determine protocol for a record, and instantiate
        protocol validator.

        This method scans the attributes values to find an active bit.
        Returns None when active bit is not found.
        """
        rec_nd = dict([(a, b) for a, b in zip(attrs, record)])
        proto_label = next(
            (a for a, b in
             [Validator.determine_proto_val(lbl, record, i) for i, lbl
              in enumerate(attrs) if 'proto' in lbl]
             if b == 1), 'other')
        if 'tcp' in proto_label:
            if validator_kind == Validator.NB15:
                return NbTCP(attrs, **rec_nd)
            if validator_kind == Validator.IOT23:
                return IotTCP(attrs, **rec_nd)
        if 'udp' in proto_label:
            if validator_kind == Validator.NB15:
                return NbUDP(attrs, **rec_nd)
            if validator_kind == Validator.IOT23:
                return IotUDP(attrs, **rec_nd)
        if 'icmp' in proto_label:
            if validator_kind == Validator.IOT23:
                return IotICMP(attrs, **rec_nd)
        if validator_kind == Validator.NB15:
            return NbOther(attrs, **rec_nd)
        if validator_kind == Validator.IOT23:
            return IotOther(attrs, **rec_nd)
        return None

    @staticmethod
    def validate_records(validator_kind, attrs, records) \
            -> Tuple[List[bool], dict]:
        temp_arr, reasons = [], {}
        for (index, record) in enumerate(records):
            instance = Validator.determine_proto(
                validator_kind, attrs, record)
            if not instance:
                temp_arr.append(True)
            else:
                is_valid, reason = instance.check()
                temp_arr.append(is_valid)
                if not is_valid:
                    key = f'{instance.name} {reason}'
                    reasons[key] = 1 + (
                        reasons[key] if key in reasons else 0)
        return temp_arr, reasons

    @staticmethod
    def validate_dataset(validator, dataset, capture=False, out=None):
        attrs, rows = utility.read_dataset(dataset)
        idx, reasons = Validator.validate_records(
            validator, attrs, rows[:, :-1])

        if sum(reasons.values()) > 0:
            invalid_idx = [i for i, v in enumerate(idx) if not v]
            Show('Validation', f'{dataset} is invalid')
            Show('Invalid reasons', utility.dump_num_dict(reasons))
            Show('Invalid records',
                 f'[{",".join([str(r + 2) for r in invalid_idx])}]')
            if capture:
                ds, ts = Path(dataset).stem, utility.ts_str()
                fn = os.path.join(out, f'{ts}_invalid_{ds}.csv')
                utility.write_dataset(fn, attrs, rows[invalid_idx, :])
                Show('Examples', fn)
        else:
            Show('Validated', f' ✓ {dataset}')
