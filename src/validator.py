from __future__ import annotations

import os
import time
from pathlib import Path
from collections import namedtuple
from typing import Tuple, Union, List

from src import Show, utility


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
    def validate(record) -> Tuple[bool, Union[str, None]]:
        return True, None

    def check(self) -> Tuple[bool, Union[str, None]]:
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
                'dttl Djit Dpkts dur state_INT stime state_FIN'
        return NetworkProto.kv_dict(
            names, [255, 255, 1] + [0] * 12)

    @staticmethod
    def validate(record) -> Tuple[bool, Union[str, None]]:
        # if record.swin != 255 or record.dwin != 255:
        #    return False, "swin-dwin mismatch"
        # synack + ackdat = tcprtt
        if round(record.synack + record.ackdat, 3) != \
                round(record.tcprtt, 3):
            return False, "synack+ackdat != tcprtt"
        if record.state_FIN != 1:
            if record.dur > 0:
                # if dur > 0 in state INT: dbytes = 0
                if record.state_INT == 1:
                    if record.dbytes != 0:
                        return False, "dbytes nonzero when INT"
                # if dur > 0 then dbytes > 0
                # elif not record.dbytes > 0:
                #     return False, "dbytes is 0 for dur > 0"
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
class NbUDP(NetworkProto):
    def __init__(self, attrs=None, **kwargs):
        super().__init__('udp', self.v_model(attrs, kwargs), **kwargs)

    @staticmethod
    def ensure_attrs():
        """Defaults for undefined attributes."""
        names = 'swin dwin stcpb dtcpb synack ' \
                'ackdat tcprtt smeansz Spkts sbytes ' \
                'dmeansz Dpkts dbytes sjit Sload'
        return NetworkProto.kv_dict(names, [0] * 15)

    @staticmethod
    def validate(record) -> Tuple[bool, Union[str, None]]:
        # TCP related fields must be 0
        if not (record.swin == 0 and record.dwin == 0
                and record.stcpb == 0 and record.dtcpb == 0
                and record.synack == 0 and record.ackdat == 0
                and record.tcprtt == 0):
            return False, "nonzero tcp fields"
        # Smeansz * Spkts = sbytes
        # if record.smeansz * record.Spkts != record.sbytes:
        #     return False, "invalid smeansz"
        # Dmeansz * Dpkts = dpytes
        # if record.dmeansz * record.Dpkts != record.dbytes:
        #     return False, "invalid dmeansz"
        # if sjit = 0 then (Smeansz * 8)/sload + something small = dur
        if record.sjit == 0 and record.Sload != 0 and \
                record.dur < (record.smeansz * 8 / record.Sload):
            return False, "invalid dur for sjit=0"
        return True, None


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
    def validate(record) -> Tuple[bool, Union[str, None]]:
        # TCP related fields must be 0
        if not (record.swin == 0 and record.dwin == 0
                and record.stcpb == 0 and record.dtcpb == 0
                and record.synack == 0 and record.ackdat == 0
                and record.tcprtt == 0):
            return False, "nonzero tcp fields"
        return True, None


# noinspection PyTypeChecker
class IotTCP(NetworkProto):
    def __init__(self, attrs=None, **kwargs):
        super().__init__('tcp', self.v_model(attrs, kwargs), **kwargs)

    @staticmethod
    def ensure_attrs():
        names = 'conn_state_S0 conn_state_SF conn_state_OTH ' \
                'conn_state_REJ history_ShADadFf history_DdA ' \
                'resp_ip_bytes resp_pkts orig_ip_bytes orig_pkts'
        return NetworkProto.kv_dict(names, [0] * 10)

    @staticmethod
    def validate(record) -> Tuple[bool, Union[str, None]]:
        # if record.orig_pkts < record.resp_pkts:
        #    return False, "ori pkts < resp pkts"
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
        # if record.orig_ip_bytes < record.resp_ip_bytes:
        #     # unless either history ShADadFf and state SF
        #     # or history DdA and state: OTH
        #     if not ((record.history_ShADadFf == 1 and
        #              record.conn_state_SF == 1) or
        #             (record.history_DdA == 1 and
        #              record.conn_state_OTH == 1)):
        #         return False, "hist conn state"
        # if conn state REJ then orig_ip_bytes=0 and resp_ip_bytes=0
        if record.conn_state_REJ == 1:
            if record.orig_ip_bytes != 0 or record.resp_ip_bytes != 0:
                return False, "REJ state bytes is 0"
        return True, None


# noinspection PyTypeChecker
class IotOther(NetworkProto):
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
class IotUDP(NetworkProto):
    def __init__(self, attrs=None, **kwargs):
        super().__init__('udp', self.v_model(attrs, kwargs), **kwargs)

    @staticmethod
    def ensure_attrs():
        names = 'conn_state_S0 conn_state_SF history_Dd ' \
                'resp_ip_bytes resp_pkts orig_ip_bytes orig_pkts'
        return NetworkProto.kv_dict(names, [0] * 7)

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
class IotICMP(NetworkProto):
    def __init__(self, attrs=None, **kwargs):
        super().__init__('icmp', self.v_model(attrs, kwargs), **kwargs)

    @staticmethod
    def ensure_attrs():
        names = 'conn_state_S0 resp_ip_bytes resp_pkts ' \
                'orig_ip_bytes orig_pkts'
        return NetworkProto.kv_dict(names, [0] * 5)

    @staticmethod
    def validate(record) -> Tuple[bool, Union[str, None]]:
        return IotOther.validate(record)


class Validator:
    # validator kinds
    NB15 = 'NB15'
    IOT23 = 'IOT23'

    # known protocols
    TCP = 'tcp'
    UDP = 'udp'
    ICMP = 'icmp'
    OTHER = 'unknown proto'

    @staticmethod
    def validate(instance: NetworkProto) -> Tuple[bool, str]:
        return instance.check()

    @staticmethod
    def determine_proto(attrs, record) -> str:
        """Determine protocol for a record.

        This method scans the attributes values to find an active bit.
        Returns `Validator.OTHER` when active bit is not found.
        """
        proto_label = next(
            (a for a, b in
             [(lbl, int(record[i])) for i, lbl
              in enumerate(attrs) if 'proto' in lbl]
             if b == 1), Validator.OTHER)
        if 'tcp' in proto_label:
            return Validator.TCP
        if 'udp' in proto_label:
            return Validator.UDP
        if 'icmp' in proto_label:
            return Validator.ICMP
        return Validator.OTHER

    @staticmethod
    def batch_validate(validator_kind, attrs, records) \
            -> Tuple[List[bool], dict]:
        temp_arr, reasons = [], {}
        for (index, record) in enumerate(records):
            # make a dictionary of record
            rec_nd = dict([(a, b) for a, b in zip(attrs, record)])
            proto = Validator.determine_proto(attrs, record)
            v_inst = None
            if validator_kind == Validator.NB15:
                if proto == Validator.TCP:
                    v_inst = NbTCP(attrs, **rec_nd)
                elif proto == Validator.UDP:
                    v_inst = NbUDP(attrs, **rec_nd)
                else:
                    v_inst = NbOther(attrs, **rec_nd)
            elif validator_kind == Validator.IOT23:
                if proto == Validator.TCP:
                    v_inst = IotTCP(attrs, **rec_nd)
                elif proto == Validator.UDP:
                    v_inst = IotUDP(attrs, **rec_nd)
                elif proto == Validator.ICMP:
                    v_inst = IotICMP(attrs, **rec_nd)
                else:
                    v_inst = IotOther(attrs, **rec_nd)
            if not v_inst:
                temp_arr.append(True)
            else:
                is_valid, reason = Validator.validate(v_inst)
                temp_arr.append(is_valid)
                if not is_valid:
                    res = f'{v_inst.name} {reason}'
                    if res not in reasons:
                        reasons[res] = 1
                    else:
                        reasons[res] += 1
        return temp_arr, reasons

    @staticmethod
    def check_dataset(ds_path, validator_kind):
        """Debug validator on some dataset"""
        import pandas as pd
        import numpy as np
        from src import Classifier
        df = pd.read_csv(ds_path).fillna(0)
        attrs = Classifier.attr_fix(
            [col for col in df.columns])
        records = np.array(df)[:, :-1]
        return Validator.batch_validate(
            validator_kind, attrs, records)

    @staticmethod
    def dump_reasons(reasons):
        return '\n'.join([txt for _, txt in sorted(
            [(v, f'{v} * {k}') for k, v in reasons.items()],
            reverse=True)])

    @staticmethod
    def validate_dataset(validator, dataset, capture, out):
        Show('Validating', dataset)
        idx, reasons = Validator.check_dataset(dataset, validator)
        if sum(reasons.values()) > 0:
            recs = [str(i + 2) for i, v in enumerate(idx) if not v]
            Show('Result', f'{dataset} is invalid')
            Show('Invalid reasons', Validator.dump_reasons(reasons))
            Show('Invalid records', f'[{",".join(recs)}]')
            if capture:
                ds = Path(dataset).stem
                ts = str(round(time.time() * 1000))[-4:]
                fn = os.path.join(out, f'invalid_{ds}_{ts}.csv')
                raw_data = open(dataset, "r").readlines()
                with open(fn, 'w', newline='') as cf:
                    cf.write(raw_data[0])
                    for i in [i + 1 for i, v in enumerate(idx) if
                              not v]:
                        cf.write(raw_data[i])
                Show('Examples', fn)
        else:
            utility.clear_one_line()
            Show('Validated', f' ✓ {dataset}')
