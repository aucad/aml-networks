from __future__ import annotations
from random import uniform, randint
from collections import Counter
import warnings

import joblib
import numpy as np
from sklearn.tree import DecisionTreeClassifier

warnings.filterwarnings("ignore")  # ignore feature names warning

VALID, INVALID = 0, 1

TCP_MODEL: DecisionTreeClassifier = joblib.load('data/DT_tcp.sav')
UDP_MODEL: DecisionTreeClassifier = joblib.load('data/DT_udp.sav')

TCP_ATTR = TCP_MODEL.feature_names_in_
UDP_ATTR = UDP_MODEL.feature_names_in_


class NetworkProto:
    """Represents validatable instance"""

    def __init__(self, name, **kwargs):
        self.name = name
        self.record = {}
        self.set_record(**kwargs)

    @property
    def attributes(self):
        return []

    @property
    def values(self):
        return list(self.record.values())

    @property
    def is_valid(self):
        return INVALID

    def generate(self):
        return self

    def set_record(self, **kwargs):
        self.record = dict(
            [(key, 0 if key not in kwargs else kwargs[key])
             for key in self.attributes])

    def check(self, model: DecisionTreeClassifier):
        if not self.is_valid:
            return INVALID
        return model.predict(np.array(self.values).reshape(1, -1))[0]


class NBRecord(NetworkProto):
    """Model for NB15 instance"""

    @property
    def swin(self):
        return self.record['swin']

    @property
    def dwin(self):
        return self.record['dwin']

    @property
    def synack(self):
        return self.record['synack']

    @property
    def ackdat(self):
        return self.record['ackdat']

    @property
    def tcprtt(self):
        return self.record['tcprtt']

    @property
    def dbytes(self):
        return self.record['dbytes']

    @property
    def Dload(self):
        return self.record['Dload']

    @property
    def dttl(self):
        return self.record['dttl']

    @property
    def Djit(self):
        return self.record['Djit']

    @property
    def Dpkts(self):
        return self.record['Dpkts']

    @property
    def smeansz(self):
        return self.record['smeansz']

    @property
    def Spkts(self):
        return self.record['Spkts']

    @property
    def sbytes(self):
        return self.record['sbytes']

    @property
    def dmeansz(self):
        return self.record['dmeansz']

    @property
    def Dpkts(self):
        return self.record['Dpkts']

    @property
    def dbytes(self):
        return self.record['dbytes']


class NbTCP(NBRecord):
    def __init__(self, **kwargs):
        super().__init__('NB15_tcp', **kwargs)

    @property
    def attributes(self):
        return TCP_ATTR

    @property
    def is_valid(self):
        if self.swin != 255 or self.dwin != 255:
            return False
        if round(self.synack + self.ackdat, 3) != round(self.tcprtt, 3):
            return False
        if self.dbytes == 0 and (
                self.Dload != 0 or self.dttl != 0 or
                self.Djit != 0 or self.Dpkts != 0):
            return False
        return True

    def generate(self):
        """Generates random NB15 TCP record"""
        p, q = uniform(0, 1) < 0.5, uniform(0, 1) < 0.5
        ackdat = uniform(0.0, 5.512234)
        synack = uniform(0.0, 4.525272)
        self.set_record(
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
            tcprtt=(synack + ackdat) if p else uniform(0.0, 10.037506),
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
        return self


class NbUDP(NBRecord):
    def __init__(self, **kwargs):
        super().__init__('NB15_udp', **kwargs)

    @property
    def attributes(self):
        return UDP_ATTR

    @property
    def is_valid(self):
        return self.smeansz * self.Spkts == self.sbytes \
               and self.dmeansz * self.Dpkts == self.dbytes

    def generate(self):
        """Generates random NB15 UDP record"""
        r = uniform(0, 1) < 0.5
        smeansz = randint(0, 1504)
        dmeansz = randint(0, 1500)
        spkts = randint(0, 10646)
        dpkts = randint(0, 11018)
        self.set_record(
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
        return self


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
        generator = lambda: (
            NbUDP() if proto == 'udp' else NbTCP() if proto == 'tcp'
            else NbTCP() if randint(0, 1) == 0 else NbUDP())
        for i in range(n):
            inst = generator().generate()
            item_counter.append(inst.name)
            valid += 1 if validator.validate(inst) == VALID else 0

        counts = [f'{k}: {v}' for k, v in Counter(item_counter).items()]
        print(f'Generated: {n}, valid: {valid}, {" ".join(counts)}')


if __name__ == '__main__':
    Validator.random_test(1000)
