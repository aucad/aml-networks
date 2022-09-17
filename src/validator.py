# TODO: fix

import random

import csv
import joblib
import numpy as np
import pandas as pd


class Nb15Validator:
    def __init__(self):
        self.tcp_model = joblib.load('data/DT_tcp.sav')
        self.udp_model = joblib.load('data/DT_udp.sav')

    @staticmethod
    def check_tcp(att_dict):
        # 0: valid 1: invalid
        print(type(att_dict['synack']))
        if str(att_dict['swin']) != str(255) or \
                str(att_dict['dwin']) != str(255):
            print('p')
            return 1
        elif round(float(att_dict['synack']) +
                   float(att_dict['ackdat']), 3) != \
                round(float(att_dict['tcprtt']), 3):
            print('pp')
            return 1
        elif att_dict['dbytes'] == 0 and \
                (att_dict['Dload'] != 0 or att_dict['dttl'] != 0 or
                 att_dict['Djit'] != 0 or att_dict['Dpkts'] != 0):
            print('ppp')
            return 1
        else:
            return 0

    @staticmethod
    def check_udp(att_dict):
        # 0: valid 1: invalid
        if int(att_dict['smeansz']) * int(att_dict['Spkts']) == \
                int(att_dict['sbytes']) \
                and int(att_dict['dmeansz']) * \
                int(att_dict['Dpkts']) == int(att_dict['dbytes']):
            return 0
        else:
            return 1

    @staticmethod
    def generate_info_dict(info_list, protocol):
        if protocol == 'tcp':
            return {'sbytes': info_list[0],
                    'dbytes': info_list[1],
                    'dttl': info_list[2],
                    'sttl': info_list[3],
                    'sloss': info_list[4],
                    'dloss': info_list[5],
                    'Sload': info_list[6],
                    'Dload': info_list[7],
                    'Spkts': info_list[8],
                    'Dpkts': info_list[9],
                    'swin': info_list[10],
                    'dwin': info_list[11],
                    'stcpb': info_list[12],
                    'dtcpb': info_list[13],
                    'smeansz': info_list[14],
                    'dmeansz': info_list[15],
                    'trans_depth': info_list[16],
                    'res_bdy_len': info_list[17],
                    'Sjit': info_list[18],
                    'Djit': info_list[19],
                    'Sintpkt': info_list[20],
                    'Dintpkt': info_list[21],
                    'tcprtt': info_list[22],
                    'synack': info_list[23],
                    'ackdat': info_list[24],
                    'is_sm_ips_ports': info_list[25],
                    'ct_state_ttl': info_list[26],
                    'ct_flw_http_mthd': info_list[27],
                    'is_ftp_login': info_list[28],
                    'ct_ftp_cmd': info_list[29],
                    'ct_srv_src': info_list[30],
                    'ct_srv_dst': info_list[31],
                    'ct_dst_ltm': info_list[32],
                    'ct_src_ltm': info_list[33],
                    'ct_src_dport_ltm': info_list[34],
                    'ct_dst_sport_ltm': info_list[35],
                    'ct_dst_src_ltm': info_list[36]}
        else:
            return {'sbytes': info_list[0],
                    'dbytes': info_list[1],
                    'dttl': info_list[2],
                    'sttl': info_list[3],
                    'sloss': info_list[4],
                    'dloss': info_list[5],
                    'Sload': info_list[6],
                    'Dload': info_list[7],
                    'Spkts': info_list[8],
                    'Dpkts': info_list[9],
                    'smeansz': info_list[10],
                    'dmeansz': info_list[11],
                    'trans_depth': info_list[12],
                    'res_bdy_len': info_list[13],
                    'Sjit': info_list[14],
                    'Djit': info_list[15],
                    'Sintpkt': info_list[16],
                    'Dintpkt': info_list[17],
                    'is_sm_ips_ports': info_list[18],
                    'ct_state_ttl': info_list[19],
                    'ct_flw_http_mthd': info_list[20],
                    'is_ftp_login': info_list[21],
                    'ct_ftp_cmd': info_list[22],
                    'ct_srv_src': info_list[23],
                    'ct_srv_dst': info_list[24],
                    'ct_dst_ltm': info_list[25],
                    'ct_src_ltm': info_list[26],
                    'ct_src_dport_ltm': info_list[27],
                    'ct_dst_sport_ltm': info_list[28],
                    'ct_dst_src_ltm': info_list[29]}

    @staticmethod
    def random_info(protocol='tcp'):
        if protocol == 'tcp':
            if random.uniform(0, 1) < 0.5:
                swin = 255
                dwin = 255
                synack = random.uniform(0.0, 4.525272)
                ackdat = random.uniform(0.0, 5.512234)
                tcprtt = synack + ackdat
                if random.uniform(0, 1) < 0.5:
                    dbytes = 0
                    dload = 0
                    dttl = 0
                    djit = 0
                    dpkts = 0
                else:
                    dbytes = random.randint(0, 14657531)
                    dload = random.uniform(0, 128761904.0)
                    dttl = random.randint(0, 254)
                    djit = random.uniform(0, 781221.1183)
                    dpkts = random.randint(0, 11018)
            else:
                swin = random.randint(0, 255)
                dwin = random.randint(0, 255)
                synack = random.uniform(0.0, 4.525272)
                ackdat = random.uniform(0.0, 5.512234)
                tcprtt = random.uniform(0.0, 10.037506)
                if random.uniform(0, 1) < 0.5:
                    dbytes = 0
                    dload = 0
                    dttl = 0
                    djit = 0
                    dpkts = 0
                else:
                    dbytes = random.randint(0, 14657531)
                    dload = random.uniform(0, 128761904.0)
                    dttl = random.randint(0, 254)
                    djit = random.uniform(0, 781221.1183)
                    dpkts = random.randint(0, 11018)

            return {'rand_sbytes': random.randint(0, 14355774),
                    'rand_dbytes': dbytes,
                    'rand_sttl': random.randint(0, 255),
                    'rand_dttl': dttl,
                    'rand_sloss': random.randint(0, 5319),
                    'rand_dloss': random.randint(0, 5507),
                    'rand_Sload': random.uniform(0, 5988000256.0),
                    'rand_Dload': dload,
                    'rand_Spkts': random.randint(0, 10646),
                    'rand_Dpkts': dpkts,
                    'rand_swin': swin,
                    'rand_dwin': dwin,
                    'rand_stcpb': random.randint(0, 4294958913),
                    'rand_dtcpb': random.randint(0, 4294953724),
                    'rand_smeansz': random.randint(0, 1504),
                    'rand_dmeansz': random.randint(0, 1500),
                    'rand_trans_depth': random.randint(0, 172),
                    'rand_res_bdy_len': random.randint(0, 6558056),
                    'rand_Sjit': random.uniform(0, 1483830.917),
                    'rand_Djit': djit,
                    'rand_Sintpkt': random.uniform(0, 84371.496),
                    'rand_Dintpkt': random.uniform(0.0, 59485.32),
                    'rand_tcprtt': tcprtt,
                    'rand_synack': synack,
                    'rand_ackdat': ackdat,
                    'rand_is_sm_ips_ports': random.randint(0, 1),
                    'rand_ct_state_ttl': random.randint(0, 6),
                    'rand_ct_flw_http_mthd': random.randint(0.0, 36.0),
                    'rand_is_ftp_login': random.randint(0, 4),
                    'rand_ct_ftp_cmd': random.randint(0, 8),
                    'rand_ct_srv_src': random.randint(1, 67),
                    'rand_ct_srv_dst': random.randint(1, 67),
                    'rand_ct_dst_ltm': random.randint(1, 67),
                    'rand_ct_src_ltm': random.randint(1, 67),
                    'rand_ct_src_dport_ltm': random.randint(1, 67),
                    'rand_ct_dst_sport_ltm': random.randint(1, 60),
                    'rand_ct_dst_src_ltm': random.randint(1, 67)}

        elif protocol == 'udp':
            if random.uniform(0, 1) < 0.5:
                Smeansz = random.randint(0, 1504)
                Dmeansz = random.randint(0, 1500)
                Spkts = random.randint(0, 10646)
                Dpkts = random.randint(0, 11018)
                sbytes = Smeansz * Spkts
                dpytes = Dmeansz * Dpkts
            else:
                Smeansz = random.randint(0, 1504)
                Dmeansz = random.randint(0, 1500)
                Spkts = random.randint(0, 10646)
                Dpkts = random.randint(0, 11018)
                sbytes = random.randint(0, 14355774)
                dpytes = random.randint(0, 14657531)

            return {'rand_sbytes': sbytes,
                    'rand_dbytes': dpytes,
                    'rand_sttl': random.randint(0, 255),
                    'rand_dttl': random.randint(0, 254),
                    'rand_sloss': random.randint(0, 5319),
                    'rand_dloss': random.randint(0, 5507),
                    'rand_Sload': random.uniform(0, 5988000256.0),
                    'rand_Dload': random.uniform(0, 128761904.0),
                    'rand_Spkts': Spkts,
                    'rand_Dpkts': Dpkts,
                    'rand_smeansz': Smeansz,
                    'rand_dmeansz': Dmeansz,
                    'rand_trans_depth': random.randint(0, 172),
                    'rand_res_bdy_len': random.randint(0, 6558056),
                    'rand_Sjit': random.uniform(0, 1483830.917),
                    'rand_Djit': random.uniform(0, 781221.1183),
                    'rand_Sintpkt': random.uniform(0, 84371.496),
                    'rand_Dintpkt': random.uniform(0.0, 59485.32),
                    'rand_is_sm_ips_ports': random.randint(0, 1),
                    'rand_ct_state_ttl': random.randint(0, 6),
                    'rand_ct_flw_http_mthd': random.randint(0.0, 36.0),
                    'rand_is_ftp_login': random.randint(0, 4),
                    'rand_ct_ftp_cmd': random.randint(0, 8),
                    'rand_ct_srv_src': random.randint(1, 67),
                    'rand_ct_srv_dst': random.randint(1, 67),
                    'rand_ct_dst_ltm': random.randint(1, 67),
                    'rand_ct_src_ltm': random.randint(1, 67),
                    'rand_ct_src_dport_ltm': random.randint(1, 67),
                    'rand_ct_dst_sport_ltm': random.randint(1, 60),
                    'rand_ct_dst_src_ltm': random.randint(1, 67)}

    def check(self, proto='tcp'):
        if proto == 'tcp':
            att_list = [self.sbytes_txt.text(), self.dbytes_txt.text(),
                        self.sttl_txt.text(), self.dttl_txt.text(), \
                        self.sloss_txt.text(), self.dloss_txt.text(),
                        self.Sload_txt.text(), self.Dload_txt.text(),
                        self.Spkts_txt.text(), self.Dpkts_txt.text(), \
                        self.swin_txt.text(), self.dwin_txt.text(),
                        self.stcpb_txt.text(), self.dtcpb_txt.text(),
                        self.smeansz_txt.text(),
                        self.dmeansz_txt.text(), \
                        self.trans_depth_txt.text(),
                        self.res_bdy_len_txt.text(),
                        self.Sjit_txt.text(), self.Djit_txt.text(), \
                        self.Sintpkt_txt.text(),
                        self.Dintpkt_txt.text(), self.tcprtt_txt.text(),
                        self.synack_txt.text(), self.ackdat_txt.text(),
                        self.is_sm_ips_ports_txt.text(), \
                        self.ct_state_ttl_txt.text(),
                        self.ct_flw_http_mthd_txt.text(),
                        self.is_ftp_login_txt.text(),
                        self.ct_ftp_cmd_txt.text(),
                        self.ct_srv_src_txt.text(), \
                        self.ct_srv_dst_txt.text(),
                        self.ct_dst_ltm_txt.text(),
                        self.ct_src_ltm_txt.text(),
                        self.ct_src_dport_ltm_txt.text(),
                        self.ct_dst_sport_ltm_txt.text(), \
                        self.ct_dst_src_ltm_txt.text()]

            attributes = list()
            for i in att_list:
                if '.' in i:
                    attributes.append(float(i))
                else:
                    attributes.append(int(i))

            att_dict = Nb15Validator.generate_info_dict(att_list, 'tcp')
            res_check = Nb15Validator.check_tcp(att_dict)

            if res_check == 0:
                attributes = np.array(attributes)
                attributes = attributes.reshape(1, -1)
                predict = self.tcp_model.predict(attributes)
                if predict[0] == 1:
                    self.status_txt.setText(f'valid packet\n{att_dict}')
                else:
                    self.status_txt.setText(
                        f'invalid packet\n{att_dict}')
            else:
                self.status_txt.setText(f'invalid packet\n{att_dict}')
        else:
            att_list = [self.sbytes_txt.text(), self.dbytes_txt.text(),
                        self.sttl_txt.text(), self.dttl_txt.text(), \
                        self.sloss_txt.text(), self.dloss_txt.text(),
                        self.Sload_txt.text(), self.Dload_txt.text(),
                        self.Spkts_txt.text(), self.Dpkts_txt.text(), \
                        self.smeansz_txt.text(),
                        self.dmeansz_txt.text(),
                        self.trans_depth_txt.text(),
                        self.res_bdy_len_txt.text(),
                        self.Sjit_txt.text(), self.Djit_txt.text(), \
                        self.Sintpkt_txt.text(),
                        self.Dintpkt_txt.text(),
                        self.is_sm_ips_ports_txt.text(), \
                        self.ct_state_ttl_txt.text(),
                        self.ct_flw_http_mthd_txt.text(),
                        self.is_ftp_login_txt.text(),
                        self.ct_ftp_cmd_txt.text(),
                        self.ct_srv_src_txt.text(), \
                        self.ct_srv_dst_txt.text(),
                        self.ct_dst_ltm_txt.text(),
                        self.ct_src_ltm_txt.text(),
                        self.ct_src_dport_ltm_txt.text(),
                        self.ct_dst_sport_ltm_txt.text(), \
                        self.ct_dst_src_ltm_txt.text()]

            attributes = list()
            for i in att_list:
                if '.' in i:
                    attributes.append(float(i))
                else:
                    attributes.append(int(i))

            att_dict = Nb15Validator.generate_info_dict(att_list, 'udp')
            res_check = Nb15Validator.check_udp(att_dict)

            if res_check == 0:
                attributes = np.array(attributes)
                attributes = attributes.reshape(1, -1)
                predict = self.udp_model.predict(attributes)
                if predict[0] == 1:
                    self.status_txt.setText(f'valid packet\n{att_dict}')
                else:
                    self.status_txt.setText(
                        f'invalid packet\n{att_dict}')
            else:
                self.status_txt.setText(f'invalid packet\n{att_dict}')

    def valid_generator_func(self, proto):
        n_gen = 100
        self.status_txt.setText('')
        res_text = str()
        generated = 0
        while generated < n_gen:
            info = Nb15Validator.random_info()

            if proto == 'tcp':
                att_list = [self.sbytes_txt.text(),
                            self.dbytes_txt.text(),
                            self.sttl_txt.text(), self.dttl_txt.text(), \
                            self.sloss_txt.text(),
                            self.dloss_txt.text(),
                            self.Sload_txt.text(),
                            self.Dload_txt.text(),
                            self.Spkts_txt.text(),
                            self.Dpkts_txt.text(), \
                            self.swin_txt.text(), self.dwin_txt.text(),
                            self.stcpb_txt.text(),
                            self.dtcpb_txt.text(),
                            self.smeansz_txt.text(),
                            self.dmeansz_txt.text(), \
                            self.trans_depth_txt.text(),
                            self.res_bdy_len_txt.text(),
                            self.Sjit_txt.text(), self.Djit_txt.text(), \
                            self.Sintpkt_txt.text(),
                            self.Dintpkt_txt.text(),
                            self.tcprtt_txt.text(),
                            self.synack_txt.text(),
                            self.ackdat_txt.text(),
                            self.is_sm_ips_ports_txt.text(), \
                            self.ct_state_ttl_txt.text(),
                            self.ct_flw_http_mthd_txt.text(),
                            self.is_ftp_login_txt.text(),
                            self.ct_ftp_cmd_txt.text(),
                            self.ct_srv_src_txt.text(), \
                            self.ct_srv_dst_txt.text(),
                            self.ct_dst_ltm_txt.text(),
                            self.ct_src_ltm_txt.text(),
                            self.ct_src_dport_ltm_txt.text(),
                            self.ct_dst_sport_ltm_txt.text(), \
                            self.ct_dst_src_ltm_txt.text()]

                attributes = list()
                for i in att_list:
                    if '.' in i:
                        attributes.append(float(i))
                    else:
                        attributes.append(int(i))
                att_dict = Nb15Validator.generate_info_dict(att_list, 'tcp')

                res_check = Nb15Validator.check_tcp(att_dict)
                if res_check == 0:
                    attributes = np.array(attributes)
                    attributes = attributes.reshape(1, -1)
                    predict = self.tcp_model.predict(attributes)

                    if predict[0] == 1:
                        res_text += f'\nvalid packet\n{att_dict}'
                        generated += 1
                    else:
                        res_text += f'\ninvalid packet\n{att_dict}'
                else:
                    res_text += f'\ninvalid packet\n{att_dict}'

            else:
                att_list = [self.sbytes_txt.text(),
                            self.dbytes_txt.text(),
                            self.sttl_txt.text(), self.dttl_txt.text(), \
                            self.sloss_txt.text(),
                            self.dloss_txt.text(),
                            self.Sload_txt.text(),
                            self.Dload_txt.text(),
                            self.Spkts_txt.text(),
                            self.Dpkts_txt.text(), \
                            self.smeansz_txt.text(),
                            self.dmeansz_txt.text(),
                            self.trans_depth_txt.text(),
                            self.res_bdy_len_txt.text(),
                            self.Sjit_txt.text(), self.Djit_txt.text(), \
                            self.Sintpkt_txt.text(),
                            self.Dintpkt_txt.text(),
                            self.is_sm_ips_ports_txt.text(), \
                            self.ct_state_ttl_txt.text(),
                            self.ct_flw_http_mthd_txt.text(),
                            self.is_ftp_login_txt.text(),
                            self.ct_ftp_cmd_txt.text(),
                            self.ct_srv_src_txt.text(), \
                            self.ct_srv_dst_txt.text(),
                            self.ct_dst_ltm_txt.text(),
                            self.ct_src_ltm_txt.text(),
                            self.ct_src_dport_ltm_txt.text(),
                            self.ct_dst_sport_ltm_txt.text(), \
                            self.ct_dst_src_ltm_txt.text()]

                attributes = list()
                for i in att_list:
                    if '.' in i:
                        attributes.append(float(i))
                    else:
                        attributes.append(int(i))
                att_dict = Nb15Validator.generate_info_dict(att_list, 'udp')

                res_check = Nb15Validator.check_udp(att_dict)
                if res_check == 0:
                    attributes = np.array(attributes)
                    attributes = attributes.reshape(1, -1)
                    predict = self.udp_model.predict(attributes)
                    if predict[0] == 1:
                        res_text += f'\nvalid packet\n{att_dict}'
                        generated += 1
                    else:
                        res_text += f'\ninvalid packet\n{att_dict}'
                else:
                    res_text += f'\ninvalid packet\n{att_dict}'

        self.status_txt.setText(res_text)

        text = self.status_txt.toPlainText()

        text = text.split('\n')

        all_dat = list()

        if text[0] == '':
            text = text[1:]
        for num, i in enumerate(text):

            if num % 2 == 1:
                val_inval = text[num - 1]
                new_dict = dict()
                field_names = list()
                for k, kk in eval(i).items():
                    new_dict[k] = kk
                new_dict['validation'] = val_inval
                all_dat.append(new_dict)

        with open('temp.csv', 'w') as csvfile:
            writer = csv.DictWriter(csvfile,
                                    fieldnames=list(all_dat[0].keys()),
                                    lineterminator='\n')
            writer.writeheader()
            for data in all_dat:
                writer.writerow(data)

        df = pd.read_csv('temp.csv', sep=',')

        df = df[df['validation'] == 'valid packet']

        df.to_csv('Valid_100.csv')

        self.show_dialog()

    def file_check(self, proto, file_path):
        res = pd.read_csv(file_path)
        res_text = str()

        if proto == 'tcp':
            for att_list in res:
                att_dict = Nb15Validator.generate_info_dict(att_list, 'tcp')
                res_check = Nb15Validator.check_tcp(att_dict)
                if res_check == 0:
                    attributes = np.array(att_list)
                    attributes = attributes.reshape(1, -1)
                    predict = self.tcp_model.predict(attributes)
                    if predict[0] == 1:
                        res_text += f'\nvalid packet\n{att_dict}'
                    else:
                        res_text += f'\ninvalid packet\n{att_dict}'
                else:
                    res_text += f'\ninvalid packet\n{att_dict}'

        else:
            for att_list in res:
                att_dict = Nb15Validator.generate_info_dict(att_list, 'udp')
                res_check = Nb15Validator.check_udp(att_dict)
                if res_check == 0:
                    attributes = np.array(att_list)
                    attributes = attributes.reshape(1, -1)
                    predict = self.udp_model.predict(attributes)
                    if predict[0] == 1:
                        res_text += f'\nvalid packet\n{att_dict}'
                    else:
                        res_text += f'\ninvalid packet\n{att_dict}'
                else:
                    res_text += f'\ninvalid packet\n{att_dict}'

        self.status_txt.setText(res_text)
