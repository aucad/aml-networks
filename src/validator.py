from __future__ import annotations
from random import uniform, randint
import warnings

import joblib
import numpy as np
from sklearn.tree import DecisionTreeClassifier

warnings.filterwarnings("ignore")
VALID, INVALID = 0, 1


class NbProto:

    def __init__(self, name=None, path=None, **kwargs):
        self.name = name
        self.model_path = path
        self.record = {}
        for key in self.attributes:
            self.record[key] = 0 if key not in kwargs else kwargs[key]

    @property
    def attributes(self):
        return []

    @property
    def is_valid(self):
        return False

    @staticmethod
    def random_info() -> NbProto:
        return NbProto()

    def to_record(self, *values):
        return dict([(a, b) for a, b in zip(self.attributes, values)])

    def check(self, model: DecisionTreeClassifier):
        if not self.is_valid:
            return INVALID
        model_attr = list(model.feature_names_in_)
        rec_attr = list(self.record.keys())
        values = [self.record[k] if k in self.record else 0
                  for k in model_attr] \
            if model_attr != rec_attr else self.record.values()
        return model.predict(np.array(values).reshape(1, -1))[0]


class NbTCP(NbProto):
    def __init__(self, classifier='data/DT_tcp.sav', **kwargs):
        super().__init__('tcp', classifier, **kwargs)

    @property
    def attributes(self):
        return ('sbytes,dbytes,sttl,dttl,sloss,dloss,Sload,Dload,'
                'Spkts,Dpkts,swin,dwin,stcpb,dtcpb,smeansz,dmeansz,'
                'trans_depth,res_bdy_len,Sjit,Djit,Sintpkt,Dintpkt,'
                'tcprtt,synack,ackdat,is_sm_ips_ports,ct_state_ttl,'
                'ct_flw_http_mthd,is_ftp_login,ct_ftp_cmd,ct_srv_src,'
                'ct_srv_dst,ct_dst_ltm,ct_src_ ltm,ct_src_dport_ltm,'
                'ct_dst_sport_ltm,ct_dst_src_ltm').split(',')

    @property
    def is_valid(self):
        # 0: valid 1: invalid
        print(type(self.record['synack']))
        if str(self.record['swin']) != str(255) or \
                str(self.record['dwin']) != str(255):
            print('p')
            return False
        elif round(float(self.record['synack']) +
                   float(self.record['ackdat']), 3) != \
                round(float(self.record['tcprtt']), 3):
            print('pp')
            return False
        elif self.record['dbytes'] == 0 and \
                (self.record['Dload'] != 0 or self.record[
                    'dttl'] != 0 or
                 self.record['Djit'] != 0 or self.record['Dpkts'] != 0):
            print('ppp')
            return False
        return True

    def random_info(self):
        x = uniform(0, 1) < 0.5
        y = uniform(0, 1) < 0.5
        ackdat = uniform(0.0, 5.512234)
        synack = uniform(0.0, 4.525272)

        record = self.to_record(
            randint(0, 14355774),
            0 if y else randint(0, 14657531),  # dbytes
            randint(0, 255),  # sttl
            0 if y else randint(0, 254),  # dttl
            randint(0, 5319),  # sloss
            randint(0, 5507),  # dloss
            uniform(0, 5988000256.0),
            0 if y else uniform(0, 128761904.0),
            randint(0, 10646),
            0 if y else randint(0, 11018),
            255 if x else randint(0, 255),
            255 if x else randint(0, 255),
            randint(0, 4294958913),
            randint(0, 4294953724),
            randint(0, 1504),
            randint(0, 1500),
            randint(0, 172),
            randint(0, 6558056),
            uniform(0, 1483830.917),
            0 if y else uniform(0, 781221.1183),
            uniform(0, 84371.496),
            uniform(0.0, 59485.32),
            (synack + ackdat) if x else uniform(0.0, 10.037506),
            synack,
            ackdat,
            randint(0, 1),
            randint(0, 6),
            randint(0, 36),
            randint(0, 4),
            randint(0, 8),
            randint(1, 67),
            randint(1, 67),
            randint(1, 67),
            randint(1, 67),
            randint(1, 67),
            randint(1, 60),
            randint(1, 67))
        return NbTCP(self.model_path, **record)


class NbUDP(NbProto):
    def __init__(self, classifier='data/DT_udp.sav', **kwargs):
        super().__init__('udp', classifier, **kwargs)

    @property
    def attributes(self):
        return ('sbytes,dbytes,sttl,dttl,sloss,dloss,Sload,Dload,'
                'Spkts,Dpkts,smeansz,dmeansz,trans_depth,res_bdy_len,'
                'Suit,Djit,Sintpkt,Dintpkt,is_sm_ips_ports,'
                'ct_state_ttl,ct_flw_http_mthd,is_ftp_login,'
                'ct_ftp_cmd,ct_srv_src,ct_srv_dst,ct_dst_ltm,'
                'ct_src_ ltm,ct_src_dport_ltm,ct_dst_sport_ltm,'
                'ct_dst_src_ltm').split(',')

    @property
    def is_valid(self):
        # 0: valid 1: invalid
        return int(self.record['smeansz']) * \
               int(self.record['Spkts']) == \
               int(self.record['sbytes']) \
               and int(self.record['dmeansz']) * \
               int(self.record['Dpkts']) == \
               int(self.record['dbytes'])

    def random_info(self):
        x = uniform(0, 1) < 0.5
        Smeansz = randint(0, 1504)
        Dmeansz = randint(0, 1500)
        Spkts = randint(0, 10646)
        Dpkts = randint(0, 11018)

        record = self.to_record(
            Smeansz * Spkts if x else randint(0, 14355774),
            Dmeansz * Dpkts if x else randint(0, 14657531),
            randint(0, 255),
            randint(0, 254),
            randint(0, 5319),
            randint(0, 5507),
            uniform(0, 5988000256.0),
            uniform(0, 128761904.0),
            Spkts, Dpkts, Smeansz, Dmeansz,
            randint(0, 172),
            randint(0, 6558056),
            uniform(0, 1483830.917),
            uniform(0, 781221.1183),
            uniform(0, 84371.496),
            uniform(0.0, 59485.32),
            randint(0, 1),
            randint(0, 6),
            randint(0, 36),
            randint(0, 4),
            randint(0, 8),
            randint(1, 67),
            randint(1, 67),
            randint(1, 67),
            randint(1, 67),
            randint(1, 67),
            randint(1, 60),
            randint(1, 67))
        return NbUDP(self.model_path, **record)


class Validator:

    def __init__(self):
        self.models = {}

    def get_model(self, record: NbProto):
        if record.name in self.models:
            return self.models[record.name]
        self.models[record.name] = joblib.load(record.model_path)
        return self.models[record.name]

    def validate(self, instance: NbProto):
        model: DecisionTreeClassifier = self.get_model(instance)
        result = instance.check(model)
        print(result, end=' ')
        return result


if __name__ == '__main__':
    my_validator = Validator()
    for i in range(100):
        udp = NbUDP().random_info()
        my_validator.validate(udp)
