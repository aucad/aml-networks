from __future__ import annotations

import logging
from random import uniform, randint
from collections import Counter, namedtuple
import warnings

import joblib
import numpy as np
from sklearn.tree import DecisionTreeClassifier

from classifier.utility import show

logger = logging.getLogger(__name__)

warnings.filterwarnings("ignore")  # ignore feature names warning

VALID, INVALID = 0, 1

TCP_MODEL: DecisionTreeClassifier = joblib.load('data/DT_tcp.sav')
UDP_MODEL: DecisionTreeClassifier = joblib.load('data/DT_udp.sav')

afix = lambda attrs: ",".join([a.replace(' ', '') for a in attrs])

NBTCP = namedtuple('NBTCP', afix(TCP_MODEL.feature_names_in_))
NBUDP = namedtuple('NBUDP', afix(UDP_MODEL.feature_names_in_))


class NetworkProto:
    """Represents validatable instance"""

    def __init__(self, name: str, kind: namedtuple, **kwargs):
        self.name = name
        self.attributes = kind._fields or []
        defaults = dict(
            [(a, 0) for a in self.attributes if a not in kwargs])
        self.record = kind(**defaults, **kwargs)

    @property
    def values(self):
        return [getattr(self.record, a) for a in self.attributes]

    @property
    def is_valid(self):
        return INVALID

    def check(self, model: DecisionTreeClassifier):
        if not self.is_valid:
            return INVALID
        return model.predict(np.array(self.values).reshape(1, -1))[0]


class NbTCP(NetworkProto):
    def __init__(self, **kwargs):
        # noinspection PyTypeChecker
        super().__init__('NB15_tcp', NBTCP, **kwargs)

    @property
    def is_valid(self):
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
    def __init__(self, **kwargs):
        # noinspection PyTypeChecker
        super().__init__('NB15_udp', NBUDP, **kwargs)

    @property
    def is_valid(self):
        return self.record.smeansz * self.record.Spkts == \
               self.record.sbytes \
               and self.record.dmeansz * self.record.Dpkts == \
               self.record.dbytes


class Validator:

    def __init__(self):
        self.models = {'NB15_tcp': TCP_MODEL, 'NB15_udp': UDP_MODEL}

    def validate(self, instance: NetworkProto):
        model = self.models[instance.name]
        return instance.check(model)

    @staticmethod
    def random_test(n, proto=None):
        """Generate and validate N random records"""
        validator = Validator()
        item_counter, valid = [], 0
        for i in range(n):
            kind = proto if proto else 'tcp' if \
                randint(0, 1) == 0 else 'udp'
            inst = Generator.generate(kind)
            item_counter.append(inst.name)
            valid += 1 if validator.validate(inst) == VALID else 0
        show('Generated', n)
        show('Valid', valid)
        for item in Counter(item_counter).items():
            show(*item)


class Generator:

    @staticmethod
    def generate(proto):
        if proto == 'tcp':
            p, q = uniform(0, 1) < 0.5, uniform(0, 1) < 0.5
            ackdat = uniform(0.0, 5.512234)
            synack = uniform(0.0, 4.525272)
            return NbTCP(
                sbytes=randint(0, 14355774),
                dbytes=0 if q else randint(0, 14657531),
                sttl=randint(0, 255),
                dttl=0 if q else randint(0, 254),
                sloss=randint(0, 5319),
                dloss=randint(0, 5507),
                Sload=uniform(0, 5988000256.0),
                Dload=0 if q else uniform(0, 128761904.0),
                Spkts=randint(0, 10646),
                Dpkts=0 if q else randint(0, 11018),
                swin=255 if p else randint(0, 255),
                dwin=255 if p else randint(0, 255),
                stcpb=randint(0, 4294958913),
                dtcpb=randint(0, 4294953724),
                smeansz=randint(0, 1504),
                dmeansz=randint(0, 1500),
                trans_depth=randint(0, 172),
                res_bdy_len=randint(0, 6558056),
                Sjit=uniform(0, 1483830.917),
                Djit=0 if q else uniform(0, 781221.1183),
                Sintpkt=uniform(0, 84371.496),
                Dintpkt=uniform(0.0, 59485.32),
                tcprtt=(synack + ackdat)
                if p else uniform(0.0, 10.037506),
                synack=synack,
                ackdat=ackdat,
                is_sm_ips_ports=randint(0, 1),
                ct_state_ttl=randint(0, 6),
                ct_flw_http_mthd=randint(0, 36),
                is_ftp_login=randint(0, 4),
                ct_ftp_cmd=randint(0, 8),
                ct_srv_src=randint(1, 67),
                ct_srv_dst=randint(1, 67),
                ct_dst_ltm=randint(1, 67),
                ct_src_ltm=randint(1, 67),
                ct_src_dport_ltm=randint(1, 67),
                ct_dst_sport_ltm=randint(1, 60),
                ct_dst_src_ltm=randint(1, 67))
        if proto == 'udp':
            r = uniform(0, 1) < 0.5
            smeansz = randint(0, 1504)
            dmeansz = randint(0, 1500)
            spkts = randint(0, 10646)
            dpkts = randint(0, 11018)
            return NbUDP(
                sbytes=smeansz * spkts if r else randint(0, 14355774),
                dbytes=dmeansz * dpkts if r else randint(0, 14657531),
                sttl=randint(0, 255),
                dttl=randint(0, 254),
                sloss=randint(0, 5319),
                dloss=randint(0, 5507),
                Sload=uniform(0, 5988000256.0),
                Dload=uniform(0, 128761904.0),
                Spkts=spkts,
                Dpkts=dpkts,
                smeansz=smeansz,
                dmeansz=dmeansz,
                trans_depth=randint(0, 172),
                res_bdy_len=randint(0, 6558056),
                Sjit=uniform(0, 1483830.917),
                Djit=uniform(0, 781221.1183),
                Sintpkt=uniform(0, 84371.496),
                Dintpkt=uniform(0.0, 59485.32),
                is_sm_ips_ports=randint(0, 1),
                ct_state_ttl=randint(0, 6),
                ct_flw_http_mthd=randint(0, 36),
                is_ftp_login=randint(0, 4),
                ct_ftp_cmd=randint(0, 8),
                ct_srv_src=randint(1, 67),
                ct_srv_dst=randint(1, 67),
                ct_dst_ltm=randint(1, 67),
                ct_src_ltm=randint(1, 67),
                ct_src_dport_ltm=randint(1, 67),
                ct_dst_sport_ltm=randint(1, 60),
                ct_dst_src_ltm=randint(1, 67))


if __name__ == '__main__':
    Validator.random_test(1000)
