# TODO: fix

import sys
from os.path import expanduser
from PyQt5 import QtCore, QtGui, QtWidgets
from PyQt5.QtWidgets import QMessageBox
from  random_info_generator import random_info, generate_info_dict
import joblib
import sklearn
import numpy as np
from text_info_reader import read_text_info
# from check_rule import check_tcp, check_udp
import csv
import pandas as pd



def check_tcp(att_dict):
    # 0: valid 1: invalid
    print(type(att_dict['synack']))
    if str(att_dict['swin']) != str(255) or str(att_dict['dwin']) != str(255):
        print('p')
        return 1
    elif round(float(att_dict['synack']) + float(att_dict['ackdat']), 3) != round(float(att_dict['tcprtt']), 3):
        print('pp')
        return 1    
    elif att_dict['dbytes'] == 0 and (att_dict['Dload'] != 0 or att_dict['dttl'] != 0 or att_dict['Djit'] != 0 or att_dict['Dpkts'] != 0):
        print('ppp')
        return 1
    else:
        return 0


def check_udp(att_dict):
    # 0: valid 1: invalid
    if int(att_dict['smeansz']) * int(att_dict['Spkts']) == int(att_dict['sbytes']) and int(att_dict['dmeansz']) * int(att_dict['Dpkts']) == int(att_dict['dbytes']):
        return 0
    else:
        return 1

class Ui_MainWindow(QtWidgets.QMainWindow):
    def setupUi(self, MainWindow):
        MainWindow.setObjectName("MainWindow")
        MainWindow.resize(860, 556)
        self.file_tcp = './DT_tcp.sav'
        self.file_udp = './DT_udp.sav'
        self.tcp_model = joblib.load(self.file_tcp)
        self.udp_model = joblib.load(self.file_udp)

        self.centralwidget = QtWidgets.QWidget(MainWindow)
        self.centralwidget.setObjectName("centralwidget")
        self.sbytes_lbl = QtWidgets.QLabel(self.centralwidget)
        self.sbytes_lbl.setGeometry(QtCore.QRect(10, 60, 67, 17))
        self.sbytes_lbl.setObjectName("sbytes_lbl")
        self.dbytes_lbl = QtWidgets.QLabel(self.centralwidget)
        self.dbytes_lbl.setGeometry(QtCore.QRect(10, 100, 67, 17))
        self.dbytes_lbl.setObjectName("dbytes_lbl")
        self.sttl_lbl = QtWidgets.QLabel(self.centralwidget)
        self.sttl_lbl.setGeometry(QtCore.QRect(10, 140, 51, 17))
        self.sttl_lbl.setObjectName("sttl_lbl")
        self.dttl_lbl = QtWidgets.QLabel(self.centralwidget)
        self.dttl_lbl.setGeometry(QtCore.QRect(10, 180, 67, 17))
        self.dttl_lbl.setObjectName("dttl_lbl")
        self.dloss_lbl = QtWidgets.QLabel(self.centralwidget)
        self.dloss_lbl.setGeometry(QtCore.QRect(10, 260, 67, 17))
        self.dloss_lbl.setObjectName("dloss_lbl")
        self.Spkts_lbl = QtWidgets.QLabel(self.centralwidget)
        self.Spkts_lbl.setGeometry(QtCore.QRect(220, 100, 51, 17))
        self.Spkts_lbl.setObjectName("Spkts_lbl")
        self.Dload_lbl = QtWidgets.QLabel(self.centralwidget)
        self.Dload_lbl.setGeometry(QtCore.QRect(10, 340, 67, 17))
        self.Dload_lbl.setObjectName("Dload_lbl")
        self.sloss_lbl = QtWidgets.QLabel(self.centralwidget)
        self.sloss_lbl.setGeometry(QtCore.QRect(10, 220, 67, 17))
        self.sloss_lbl.setObjectName("sloss_lbl")
        self.Sload_lbl = QtWidgets.QLabel(self.centralwidget)
        self.Sload_lbl.setGeometry(QtCore.QRect(10, 300, 67, 17))
        self.Sload_lbl.setObjectName("Sload_lbl")
        self.Dpkts_lbl = QtWidgets.QLabel(self.centralwidget)
        self.Dpkts_lbl.setGeometry(QtCore.QRect(220, 140, 51, 17))
        self.Dpkts_lbl.setObjectName("Dpkts_lbl")
        self.random_btn = QtWidgets.QPushButton(self.centralwidget)
        self.random_btn.setGeometry(QtCore.QRect(430, 510, 191, 41))
        self.random_btn.setObjectName("random_btn")
        self.swin_lbl = QtWidgets.QLabel(self.centralwidget)
        self.swin_lbl.setGeometry(QtCore.QRect(220, 340, 41, 17))
        self.swin_lbl.setObjectName("swin_lbl")
        self.dwin_lbl = QtWidgets.QLabel(self.centralwidget)
        self.dwin_lbl.setGeometry(QtCore.QRect(220, 380, 51, 17))
        self.dwin_lbl.setObjectName("dwin_lbl")
        self.smeansz_lbl = QtWidgets.QLabel(self.centralwidget)
        self.smeansz_lbl.setGeometry(QtCore.QRect(390, 260, 67, 17))
        self.smeansz_lbl.setObjectName("smeansz_lbl")
        self.stcpb_lbl = QtWidgets.QLabel(self.centralwidget)
        self.stcpb_lbl.setGeometry(QtCore.QRect(400, 180, 67, 17))
        self.stcpb_lbl.setObjectName("stcpb_lbl")
        self.res_bdy_len_lbl = QtWidgets.QLabel(self.centralwidget)
        self.res_bdy_len_lbl.setGeometry(QtCore.QRect(610, 330, 101, 17))
        self.res_bdy_len_lbl.setObjectName("res_bdy_len_lbl")
        self.dtcpb_lbl = QtWidgets.QLabel(self.centralwidget)
        self.dtcpb_lbl.setGeometry(QtCore.QRect(400, 220, 67, 17))
        self.dtcpb_lbl.setObjectName("dtcpb_lbl")
        self.dmeansz_lbl = QtWidgets.QLabel(self.centralwidget)
        self.dmeansz_lbl.setGeometry(QtCore.QRect(390, 300, 67, 17))
        self.dmeansz_lbl.setObjectName("dmeansz_lbl")
        self.trans_depth_lbl = QtWidgets.QLabel(self.centralwidget)
        self.trans_depth_lbl.setGeometry(QtCore.QRect(610, 370, 91, 17))
        self.trans_depth_lbl.setObjectName("trans_depth_lbl")
        self.Djit_lbl = QtWidgets.QLabel(self.centralwidget)
        self.Djit_lbl.setGeometry(QtCore.QRect(220, 220, 41, 17))
        self.Djit_lbl.setObjectName("Djit_lbl")
        self.Sintpkt_lbl = QtWidgets.QLabel(self.centralwidget)
        self.Sintpkt_lbl.setGeometry(QtCore.QRect(210, 260, 51, 17))
        self.Sintpkt_lbl.setObjectName("Sintpkt_lbl")
        self.Dintpkt_lbl = QtWidgets.QLabel(self.centralwidget)
        self.Dintpkt_lbl.setGeometry(QtCore.QRect(210, 300, 51, 17))
        self.Dintpkt_lbl.setObjectName("Dintpkt_lbl")
        self.synack_lbl = QtWidgets.QLabel(self.centralwidget)
        self.synack_lbl.setGeometry(QtCore.QRect(400, 380, 67, 17))
        self.synack_lbl.setObjectName("synack_lbl")
        self.Sjit_lbl = QtWidgets.QLabel(self.centralwidget)
        self.Sjit_lbl.setGeometry(QtCore.QRect(220, 180, 41, 17))
        self.Sjit_lbl.setObjectName("Sjit_lbl")
        self.tcprtt_lbl = QtWidgets.QLabel(self.centralwidget)
        self.tcprtt_lbl.setGeometry(QtCore.QRect(220, 60, 51, 17))
        self.tcprtt_lbl.setObjectName("tcprtt_lbl")
        self.export_btn = QtWidgets.QPushButton(self.centralwidget)
        self.export_btn.setGeometry(QtCore.QRect(310, 510, 111, 41))
        self.export_btn.setObjectName("export_btn")
        self.is_sm_ips_ports_lbl = QtWidgets.QLabel(self.centralwidget)
        self.is_sm_ips_ports_lbl.setGeometry(QtCore.QRect(600, 410, 111, 17))
        self.is_sm_ips_ports_lbl.setObjectName("is_sm_ips_ports_lbl")
        self.ackdat_lbl = QtWidgets.QLabel(self.centralwidget)
        self.ackdat_lbl.setGeometry(QtCore.QRect(400, 340, 67, 17))
        self.ackdat_lbl.setObjectName("ackdat_lbl")
        self.ct_state_ttl_lbl = QtWidgets.QLabel(self.centralwidget)
        self.ct_state_ttl_lbl.setGeometry(QtCore.QRect(0, 380, 81, 17))
        self.ct_state_ttl_lbl.setObjectName("ct_state_ttl_lbl")
        self.ct_flw_http_mthd_lbl = QtWidgets.QLabel(self.centralwidget)
        self.ct_flw_http_mthd_lbl.setGeometry(QtCore.QRect(600, 290, 121, 17))
        self.ct_flw_http_mthd_lbl.setObjectName("ct_flw_http_mthd_lbl")
        self.ct_ftp_cmd_lbl = QtWidgets.QLabel(self.centralwidget)
        self.ct_ftp_cmd_lbl.setGeometry(QtCore.QRect(610, 250, 91, 17))
        self.ct_ftp_cmd_lbl.setObjectName("ct_ftp_cmd_lbl")
        self.is_ftp_login_lbl = QtWidgets.QLabel(self.centralwidget)
        self.is_ftp_login_lbl.setGeometry(QtCore.QRect(610, 220, 91, 17))
        self.is_ftp_login_lbl.setObjectName("is_ftp_login_lbl")
        self.ct_src_ltm_lbl = QtWidgets.QLabel(self.centralwidget)
        self.ct_src_ltm_lbl.setGeometry(QtCore.QRect(610, 180, 91, 17))
        self.ct_src_ltm_lbl.setObjectName("ct_src_ltm_lbl")
        self.ct_dst_ltm_lbl = QtWidgets.QLabel(self.centralwidget)
        self.ct_dst_ltm_lbl.setGeometry(QtCore.QRect(390, 50, 71, 17))
        self.ct_dst_ltm_lbl.setObjectName("ct_dst_ltm_lbl")
        self.ct_srv_src_lbl = QtWidgets.QLabel(self.centralwidget)
        self.ct_srv_src_lbl.setGeometry(QtCore.QRect(390, 90, 81, 20))
        self.ct_srv_src_lbl.setObjectName("ct_srv_src_lbl")
        self.ct_srv_dst_lbl = QtWidgets.QLabel(self.centralwidget)
        self.ct_srv_dst_lbl.setGeometry(QtCore.QRect(390, 140, 71, 17))
        self.ct_srv_dst_lbl.setObjectName("ct_srv_dst_lbl")
        self.file_btn = QtWidgets.QPushButton(self.centralwidget)
        self.file_btn.setGeometry(QtCore.QRect(630, 510, 131, 41))
        self.file_btn.setObjectName("file_btn")
        self.ct_src_dport_ltm_lbl = QtWidgets.QLabel(self.centralwidget)
        self.ct_src_dport_ltm_lbl.setGeometry(QtCore.QRect(600, 50, 121, 17))
        self.ct_src_dport_ltm_lbl.setObjectName("ct_src_dport_ltm_lbl")
        self.ct_dst_sport_ltm_lbl = QtWidgets.QLabel(self.centralwidget)
        self.ct_dst_sport_ltm_lbl.setGeometry(QtCore.QRect(600, 100, 131, 17))
        self.ct_dst_sport_ltm_lbl.setObjectName("ct_dst_sport_ltm_lbl")
        self.ct_dst_src_ltm_lbl = QtWidgets.QLabel(self.centralwidget)
        self.ct_dst_src_ltm_lbl.setGeometry(QtCore.QRect(600, 140, 111, 17))
        self.ct_dst_src_ltm_lbl.setObjectName("ct_dst_src_ltm_lbl")
        self.sbytes_txt = QtWidgets.QLineEdit(self.centralwidget)
        self.sbytes_txt.setGeometry(QtCore.QRect(90, 60, 113, 25))
        self.sbytes_txt.setObjectName("sbytes_txt")
        self.dbytes_txt = QtWidgets.QLineEdit(self.centralwidget)
        self.dbytes_txt.setGeometry(QtCore.QRect(90, 100, 113, 25))
        self.dbytes_txt.setObjectName("dbytes_txt")
        self.sttl_txt = QtWidgets.QLineEdit(self.centralwidget)
        self.sttl_txt.setGeometry(QtCore.QRect(90, 140, 113, 25))
        self.sttl_txt.setObjectName("sttl_txt")
        self.dttl_txt = QtWidgets.QLineEdit(self.centralwidget)
        self.dttl_txt.setGeometry(QtCore.QRect(90, 180, 113, 25))
        self.dttl_txt.setObjectName("dttl_txt")
        self.sloss_txt = QtWidgets.QLineEdit(self.centralwidget)
        self.sloss_txt.setGeometry(QtCore.QRect(90, 220, 113, 25))
        self.sloss_txt.setObjectName("sloss_txt")
        self.dloss_txt = QtWidgets.QLineEdit(self.centralwidget)
        self.dloss_txt.setGeometry(QtCore.QRect(90, 260, 113, 25))
        self.dloss_txt.setObjectName("dloss_txt")
        self.Sload_txt = QtWidgets.QLineEdit(self.centralwidget)
        self.Sload_txt.setGeometry(QtCore.QRect(90, 300, 113, 25))
        self.Sload_txt.setObjectName("Sload_txt")
        self.Dload_txt = QtWidgets.QLineEdit(self.centralwidget)
        self.Dload_txt.setGeometry(QtCore.QRect(90, 340, 113, 25))
        self.Dload_txt.setObjectName("Dload_txt")
        self.tcprtt_txt = QtWidgets.QLineEdit(self.centralwidget)
        self.tcprtt_txt.setGeometry(QtCore.QRect(270, 50, 113, 25))
        self.tcprtt_txt.setObjectName("tcprtt_txt")
        self.Spkts_txt = QtWidgets.QLineEdit(self.centralwidget)
        self.Spkts_txt.setGeometry(QtCore.QRect(270, 90, 113, 25))
        self.Spkts_txt.setObjectName("Spkts_txt")
        self.Dpkts_txt = QtWidgets.QLineEdit(self.centralwidget)
        self.Dpkts_txt.setGeometry(QtCore.QRect(270, 130, 113, 25))
        self.Dpkts_txt.setObjectName("Dpkts_txt")
        self.Sjit_txt = QtWidgets.QLineEdit(self.centralwidget)
        self.Sjit_txt.setGeometry(QtCore.QRect(270, 170, 113, 25))
        self.Sjit_txt.setObjectName("Sjit_txt")
        self.Djit_txt = QtWidgets.QLineEdit(self.centralwidget)
        self.Djit_txt.setGeometry(QtCore.QRect(270, 210, 113, 25))
        self.Djit_txt.setObjectName("Djit_txt")
        self.Sintpkt_txt = QtWidgets.QLineEdit(self.centralwidget)
        self.Sintpkt_txt.setGeometry(QtCore.QRect(270, 250, 113, 25))
        self.Sintpkt_txt.setObjectName("Sintpkt_txt")
        self.Dintpkt_txt = QtWidgets.QLineEdit(self.centralwidget)
        self.Dintpkt_txt.setGeometry(QtCore.QRect(270, 290, 113, 25))
        self.Dintpkt_txt.setObjectName("Dintpkt_txt")
        self.swin_txt = QtWidgets.QLineEdit(self.centralwidget)
        self.swin_txt.setGeometry(QtCore.QRect(270, 330, 113, 25))
        self.swin_txt.setObjectName("swin_txt")
        self.dwin_txt = QtWidgets.QLineEdit(self.centralwidget)
        self.dwin_txt.setGeometry(QtCore.QRect(270, 370, 113, 25))
        self.dwin_txt.setObjectName("dwin_txt")
        self.ct_srv_src_txt = QtWidgets.QLineEdit(self.centralwidget)
        self.ct_srv_src_txt.setGeometry(QtCore.QRect(470, 90, 113, 25))
        self.ct_srv_src_txt.setObjectName("ct_srv_src_txt")
        self.dtcpb_txt = QtWidgets.QLineEdit(self.centralwidget)
        self.dtcpb_txt.setGeometry(QtCore.QRect(470, 210, 113, 25))
        self.dtcpb_txt.setObjectName("dtcpb_txt")
        self.stcpb_txt = QtWidgets.QLineEdit(self.centralwidget)
        self.stcpb_txt.setGeometry(QtCore.QRect(470, 170, 113, 25))
        self.stcpb_txt.setObjectName("stcpb_txt")
        self.smeansz_txt = QtWidgets.QLineEdit(self.centralwidget)
        self.smeansz_txt.setGeometry(QtCore.QRect(470, 250, 113, 25))
        self.smeansz_txt.setObjectName("smeansz_txt")
        self.ct_dst_ltm_txt = QtWidgets.QLineEdit(self.centralwidget)
        self.ct_dst_ltm_txt.setGeometry(QtCore.QRect(470, 50, 113, 25))
        self.ct_dst_ltm_txt.setObjectName("ct_dst_ltm_txt")
        self.ct_srv_dst_txt = QtWidgets.QLineEdit(self.centralwidget)
        self.ct_srv_dst_txt.setGeometry(QtCore.QRect(470, 130, 113, 25))
        self.ct_srv_dst_txt.setObjectName("ct_srv_dst_txt")
        self.dmeansz_txt = QtWidgets.QLineEdit(self.centralwidget)
        self.dmeansz_txt.setGeometry(QtCore.QRect(470, 290, 113, 25))
        self.dmeansz_txt.setObjectName("dmeansz_txt")
        self.ct_flw_http_mthd_txt = QtWidgets.QLineEdit(self.centralwidget)
        self.ct_flw_http_mthd_txt.setGeometry(QtCore.QRect(730, 290, 113, 25))
        self.ct_flw_http_mthd_txt.setObjectName("ct_flw_http_mthd_txt")
        self.is_ftp_login_txt = QtWidgets.QLineEdit(self.centralwidget)
        self.is_ftp_login_txt.setGeometry(QtCore.QRect(730, 210, 113, 25))
        self.is_ftp_login_txt.setObjectName("is_ftp_login_txt")
        self.ct_dst_src_ltm_txt = QtWidgets.QLineEdit(self.centralwidget)
        self.ct_dst_src_ltm_txt.setGeometry(QtCore.QRect(730, 130, 113, 25))
        self.ct_dst_src_ltm_txt.setObjectName("ct_dst_src_ltm_txt")
        self.ct_src_ltm_txt = QtWidgets.QLineEdit(self.centralwidget)
        self.ct_src_ltm_txt.setGeometry(QtCore.QRect(730, 170, 113, 25))
        self.ct_src_ltm_txt.setObjectName("ct_src_ltm_txt")
        self.ct_dst_sport_ltm_txt = QtWidgets.QLineEdit(self.centralwidget)
        self.ct_dst_sport_ltm_txt.setGeometry(QtCore.QRect(730, 90, 113, 25))
        self.ct_dst_sport_ltm_txt.setObjectName("ct_dst_sport_ltm_txt")
        self.ct_ftp_cmd_txt = QtWidgets.QLineEdit(self.centralwidget)
        self.ct_ftp_cmd_txt.setGeometry(QtCore.QRect(730, 250, 113, 25))
        self.ct_ftp_cmd_txt.setObjectName("ct_ftp_cmd_txt")
        self.is_sm_ips_ports_txt = QtWidgets.QLineEdit(self.centralwidget)
        self.is_sm_ips_ports_txt.setGeometry(QtCore.QRect(730, 410, 113, 25))
        self.is_sm_ips_ports_txt.setObjectName("is_sm_ips_ports_txt")
        self.ct_src_dport_ltm_txt = QtWidgets.QLineEdit(self.centralwidget)
        self.ct_src_dport_ltm_txt.setGeometry(QtCore.QRect(730, 50, 113, 25))
        self.ct_src_dport_ltm_txt.setObjectName("ct_src_dport_ltm_txt")
        self.res_bdy_len_txt = QtWidgets.QLineEdit(self.centralwidget)
        self.res_bdy_len_txt.setGeometry(QtCore.QRect(730, 330, 113, 25))
        self.res_bdy_len_txt.setObjectName("res_bdy_len_txt")
        self.trans_depth_txt = QtWidgets.QLineEdit(self.centralwidget)
        self.trans_depth_txt.setGeometry(QtCore.QRect(730, 370, 113, 25))
        self.trans_depth_txt.setObjectName("trans_depth_txt")
        self.ct_state_ttl_txt = QtWidgets.QLineEdit(self.centralwidget)
        self.ct_state_ttl_txt.setGeometry(QtCore.QRect(90, 380, 113, 25))
        self.ct_state_ttl_txt.setObjectName("ct_state_ttl_txt")
        self.ackdat_txt = QtWidgets.QLineEdit(self.centralwidget)
        self.ackdat_txt.setGeometry(QtCore.QRect(470, 330, 113, 25))
        self.ackdat_txt.setObjectName("ackdat_txt")
        self.synack_txt = QtWidgets.QLineEdit(self.centralwidget)
        self.synack_txt.setGeometry(QtCore.QRect(470, 370, 113, 25))
        self.synack_txt.setObjectName("synack_txt")
        self.rule_check_btn = QtWidgets.QPushButton(self.centralwidget)
        self.rule_check_btn.setGeometry(QtCore.QRect(170, 510, 131, 41))
        self.rule_check_btn.setObjectName("rule_check_btn")
        self.valid_generator = QtWidgets.QPushButton(self.centralwidget)
        self.valid_generator.setGeometry(QtCore.QRect(50, 510, 110, 41))
        self.valid_generator.setObjectName("valid_generator")
        self.status_txt = QtWidgets.QTextEdit(self.centralwidget)
        self.status_txt.setGeometry(QtCore.QRect(90, 450, 711, 51))
        self.status_txt.setObjectName("status_txt")
        self.comboBox = QtWidgets.QComboBox(self.centralwidget)
        self.comboBox.setGeometry(QtCore.QRect(80, 10, 141, 25))
        self.comboBox.setObjectName("comboBox")
        self.comboBox.addItem("")
        self.comboBox.addItem("")
        MainWindow.setCentralWidget(self.centralwidget)

        self.retranslateUi(MainWindow)
        QtCore.QMetaObject.connectSlotsByName(MainWindow)


    def combobox_act(self):
        combo = str(self.comboBox.currentText())
        if combo == 'udp':
            self.swin_txt.setDisabled(True)
            self.dwin_txt.setDisabled(True)
            self.stcpb_txt.setDisabled(True)
            self.dtcpb_txt.setDisabled(True)
            self.tcprtt_txt.setDisabled(True)
            self.synack_txt.setDisabled(True)
            self.ackdat_txt.setDisabled(True)

    def random_generate(self):
        self.status_txt.setText('')
        combo = str(self.comboBox.currentText())
        info = random_info(combo)
        if combo == 'tcp':
            self.sbytes_txt.setText(str(info['rand_sbytes']))
            self.dbytes_txt.setText(str(info['rand_dbytes']))
            self.sttl_txt.setText(str(info['rand_sttl']))
            self.dttl_txt.setText(str(info['rand_dttl']))
            self.sloss_txt.setText(str(info['rand_sloss']))
            self.dloss_txt.setText(str(info['rand_dloss']))
            self.Sload_txt.setText(str(info['rand_Sload']))
            self.Dload_txt.setText(str(info['rand_Dload']))
            self.Spkts_txt.setText(str(info['rand_Spkts']))
            self.Dpkts_txt.setText(str(info['rand_Dpkts']))
            self.swin_txt.setText(str(info['rand_swin']))
            self.dwin_txt.setText(str(info['rand_dwin']))
            self.stcpb_txt.setText(str(info['rand_stcpb']))
            self.dtcpb_txt.setText(str(info['rand_dtcpb']))
            self.smeansz_txt.setText(str(info['rand_smeansz']))
            self.dmeansz_txt.setText(str(info['rand_dmeansz']))
            self.trans_depth_txt.setText(str(info['rand_trans_depth']))
            self.res_bdy_len_txt.setText(str(info['rand_res_bdy_len']))
            self.Sjit_txt.setText(str(info['rand_Sjit']))
            self.Djit_txt.setText(str(info['rand_Djit']))
            self.Sintpkt_txt.setText(str(info['rand_Sintpkt']))
            self.Dintpkt_txt.setText(str(info['rand_Dintpkt']))
            self.tcprtt_txt.setText(str(info['rand_tcprtt']))
            self.synack_txt.setText(str(info['rand_synack']))
            self.ackdat_txt.setText(str(info['rand_ackdat']))
            self.is_sm_ips_ports_txt.setText(str(info['rand_is_sm_ips_ports']))
            self.ct_state_ttl_txt.setText(str(info['rand_ct_state_ttl']))
            self.ct_flw_http_mthd_txt.setText(str(info['rand_ct_flw_http_mthd']))
            self.is_ftp_login_txt.setText(str(info['rand_is_ftp_login']))
            self.ct_ftp_cmd_txt.setText(str(info['rand_ct_ftp_cmd']))
            self.ct_srv_src_txt.setText(str(info['rand_ct_srv_src']))
            self.ct_srv_dst_txt.setText(str(info['rand_ct_srv_dst']))
            self.ct_dst_ltm_txt.setText(str(info['rand_ct_dst_ltm']))
            self.ct_src_ltm_txt.setText(str(info['rand_ct_src_ltm']))
            self.ct_src_dport_ltm_txt.setText(str(info['rand_ct_src_dport_ltm']))
            self.ct_dst_sport_ltm_txt.setText(str(info['rand_ct_dst_sport_ltm']))
            self.ct_dst_src_ltm_txt.setText(str(info['rand_ct_dst_src_ltm']))
        else:
            self.sbytes_txt.setText(str(info['rand_sbytes']))
            self.dbytes_txt.setText(str(info['rand_dbytes']))
            self.sttl_txt.setText(str(info['rand_sttl']))
            self.dttl_txt.setText(str(info['rand_dttl']))
            self.sloss_txt.setText(str(info['rand_sloss']))
            self.dloss_txt.setText(str(info['rand_dloss']))
            self.Sload_txt.setText(str(info['rand_Sload']))
            self.Dload_txt.setText(str(info['rand_Dload']))
            self.Spkts_txt.setText(str(info['rand_Spkts']))
            self.Dpkts_txt.setText(str(info['rand_Dpkts']))
            self.swin_txt.setText('')
            self.dwin_txt.setText('')
            self.stcpb_txt.setText('')
            self.dtcpb_txt.setText('')
            self.smeansz_txt.setText(str(info['rand_smeansz']))
            self.dmeansz_txt.setText(str(info['rand_dmeansz']))
            self.trans_depth_txt.setText(str(info['rand_trans_depth']))
            self.res_bdy_len_txt.setText(str(info['rand_res_bdy_len']))
            self.Sjit_txt.setText(str(info['rand_Sjit']))
            self.Djit_txt.setText(str(info['rand_Djit']))
            self.Sintpkt_txt.setText(str(info['rand_Sintpkt']))
            self.Dintpkt_txt.setText(str(info['rand_Dintpkt']))
            self.tcprtt_txt.setText('')
            self.synack_txt.setText('')
            self.ackdat_txt.setText('')
            self.is_sm_ips_ports_txt.setText(str(info['rand_is_sm_ips_ports']))
            self.ct_state_ttl_txt.setText(str(info['rand_ct_state_ttl']))
            self.ct_flw_http_mthd_txt.setText(str(info['rand_ct_flw_http_mthd']))
            self.is_ftp_login_txt.setText(str(info['rand_is_ftp_login']))
            self.ct_ftp_cmd_txt.setText(str(info['rand_ct_ftp_cmd']))
            self.ct_srv_src_txt.setText(str(info['rand_ct_srv_src']))
            self.ct_srv_dst_txt.setText(str(info['rand_ct_srv_dst']))
            self.ct_dst_ltm_txt.setText(str(info['rand_ct_dst_ltm']))
            self.ct_src_ltm_txt.setText(str(info['rand_ct_src_ltm']))
            self.ct_src_dport_ltm_txt.setText(str(info['rand_ct_src_dport_ltm']))
            self.ct_dst_sport_ltm_txt.setText(str(info['rand_ct_dst_sport_ltm']))
            self.ct_dst_src_ltm_txt.setText(str(info['rand_ct_dst_src_ltm']))

    def check(self):
        combo = str(self.comboBox.currentText())
        if combo == 'tcp':
            att_list = [self.sbytes_txt.text(), self.dbytes_txt.text(), self.sttl_txt.text(), self.dttl_txt.text(),\
            self.sloss_txt.text(), self.dloss_txt.text(), self.Sload_txt.text(), self.Dload_txt.text(), self.Spkts_txt.text(), self.Dpkts_txt.text(),\
            self.swin_txt.text(), self.dwin_txt.text(), self.stcpb_txt.text(), self.dtcpb_txt.text(), self.smeansz_txt.text(), self.dmeansz_txt.text(),\
            self.trans_depth_txt.text(), self.res_bdy_len_txt.text(), self.Sjit_txt.text(), self.Djit_txt.text(), \
            self.Sintpkt_txt.text(), self.Dintpkt_txt.text(), self.tcprtt_txt.text(), self.synack_txt.text(), self.ackdat_txt.text(), self.is_sm_ips_ports_txt.text(),\
            self.ct_state_ttl_txt.text(), self.ct_flw_http_mthd_txt.text(), self.is_ftp_login_txt.text(), self.ct_ftp_cmd_txt.text(), self.ct_srv_src_txt.text(),\
            self.ct_srv_dst_txt.text(), self.ct_dst_ltm_txt.text(), self.ct_src_ltm_txt.text(), self.ct_src_dport_ltm_txt.text(), self.ct_dst_sport_ltm_txt.text(),\
            self.ct_dst_src_ltm_txt.text()]

            attributes = list()
            for i in att_list:
                if '.' in i:
                    attributes.append(float(i))
                else:
                    attributes.append(int(i))
            att_dict = generate_info_dict(att_list, 'tcp')
            
            res_check = check_tcp(att_dict)
            if res_check == 0:
                attributes = np.array(attributes)
                attributes = attributes.reshape(1, -1)
                predict = self.tcp_model.predict(attributes)
                if predict[0] == 1:
                    self.status_txt.setText(f'valid packet\n{att_dict}')
                else:
                    self.status_txt.setText(f'invalid packet\n{att_dict}')
            else:
                self.status_txt.setText(f'invalid packet\n{att_dict}')
        else:
            att_list = [self.sbytes_txt.text(), self.dbytes_txt.text(), self.sttl_txt.text(), self.dttl_txt.text(),\
            self.sloss_txt.text(), self.dloss_txt.text(), self.Sload_txt.text(), self.Dload_txt.text(), self.Spkts_txt.text(), self.Dpkts_txt.text(),\
            self.smeansz_txt.text(), self.dmeansz_txt.text(),self.trans_depth_txt.text(), self.res_bdy_len_txt.text(), self.Sjit_txt.text(), self.Djit_txt.text(), \
            self.Sintpkt_txt.text(), self.Dintpkt_txt.text(), self.is_sm_ips_ports_txt.text(),\
            self.ct_state_ttl_txt.text(), self.ct_flw_http_mthd_txt.text(), self.is_ftp_login_txt.text(), self.ct_ftp_cmd_txt.text(), self.ct_srv_src_txt.text(),\
            self.ct_srv_dst_txt.text(), self.ct_dst_ltm_txt.text(), self.ct_src_ltm_txt.text(), self.ct_src_dport_ltm_txt.text(), self.ct_dst_sport_ltm_txt.text(),\
            self.ct_dst_src_ltm_txt.text()]

            attributes = list()
            for i in att_list:
                if '.' in i:
                    attributes.append(float(i))
                else:
                    attributes.append(int(i))
            att_dict = generate_info_dict(att_list, 'udp')
    
            res_check = check_udp(att_dict)
            if res_check == 0:
                attributes = np.array(attributes)
                attributes = attributes.reshape(1, -1)
                predict = self.udp_model.predict(attributes)
                if predict[0] == 1:
                    self.status_txt.setText(f'valid packet\n{att_dict}')
                else:
                    self.status_txt.setText(f'invalid packet\n{att_dict}')
            else:
                self.status_txt.setText(f'invalid packet\n{att_dict}')


    def valid_generator_func(self):
        n_gen = 100
        self.status_txt.setText('')
        res_text = str()
        generated = 0
        while generated < n_gen:
            self.random_generate()
            combo = str(self.comboBox.currentText())

            if combo == 'tcp':
                att_list = [self.sbytes_txt.text(), self.dbytes_txt.text(), self.sttl_txt.text(), self.dttl_txt.text(),\
                self.sloss_txt.text(), self.dloss_txt.text(), self.Sload_txt.text(), self.Dload_txt.text(), self.Spkts_txt.text(), self.Dpkts_txt.text(),\
                self.swin_txt.text(), self.dwin_txt.text(), self.stcpb_txt.text(), self.dtcpb_txt.text(), self.smeansz_txt.text(), self.dmeansz_txt.text(),\
                self.trans_depth_txt.text(), self.res_bdy_len_txt.text(), self.Sjit_txt.text(), self.Djit_txt.text(), \
                self.Sintpkt_txt.text(), self.Dintpkt_txt.text(), self.tcprtt_txt.text(), self.synack_txt.text(), self.ackdat_txt.text(), self.is_sm_ips_ports_txt.text(),\
                self.ct_state_ttl_txt.text(), self.ct_flw_http_mthd_txt.text(), self.is_ftp_login_txt.text(), self.ct_ftp_cmd_txt.text(), self.ct_srv_src_txt.text(),\
                self.ct_srv_dst_txt.text(), self.ct_dst_ltm_txt.text(), self.ct_src_ltm_txt.text(), self.ct_src_dport_ltm_txt.text(), self.ct_dst_sport_ltm_txt.text(),\
                self.ct_dst_src_ltm_txt.text()]

                attributes = list()
                for i in att_list:
                    if '.' in i:
                        attributes.append(float(i))
                    else:
                        attributes.append(int(i))
                att_dict = generate_info_dict(att_list, 'tcp')

                res_check = check_tcp(att_dict)
                if res_check == 0:
                    attributes = np.array(attributes)
                    attributes = attributes.reshape(1, -1)
                    predict = self.tcp_model.predict(attributes)
                    
                    if predict[0] == 1:
                        res_text += f'\nvalid packet\n{att_dict}'
                        generated+=1
                    else:
                        res_text += f'\ninvalid packet\n{att_dict}'
                else:
                    res_text += f'\ninvalid packet\n{att_dict}'
                

            else:
                att_list = [self.sbytes_txt.text(), self.dbytes_txt.text(), self.sttl_txt.text(), self.dttl_txt.text(),\
                self.sloss_txt.text(), self.dloss_txt.text(), self.Sload_txt.text(), self.Dload_txt.text(), self.Spkts_txt.text(), self.Dpkts_txt.text(),\
                self.smeansz_txt.text(), self.dmeansz_txt.text(),self.trans_depth_txt.text(), self.res_bdy_len_txt.text(), self.Sjit_txt.text(), self.Djit_txt.text(), \
                self.Sintpkt_txt.text(), self.Dintpkt_txt.text(), self.is_sm_ips_ports_txt.text(),\
                self.ct_state_ttl_txt.text(), self.ct_flw_http_mthd_txt.text(), self.is_ftp_login_txt.text(), self.ct_ftp_cmd_txt.text(), self.ct_srv_src_txt.text(),\
                self.ct_srv_dst_txt.text(), self.ct_dst_ltm_txt.text(), self.ct_src_ltm_txt.text(), self.ct_src_dport_ltm_txt.text(), self.ct_dst_sport_ltm_txt.text(),\
                self.ct_dst_src_ltm_txt.text()]

                attributes = list()
                for i in att_list:
                    if '.' in i:
                        attributes.append(float(i))
                    else:
                        attributes.append(int(i))
                att_dict = generate_info_dict(att_list, 'udp')
        
                res_check = check_udp(att_dict)
                if res_check == 0:
                    attributes = np.array(attributes)
                    attributes = attributes.reshape(1, -1)
                    predict = self.udp_model.predict(attributes)
                    if predict[0] == 1:
                        res_text += f'\nvalid packet\n{att_dict}'
                        generated+=1
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
                val_inval = text[num-1]
                new_dict = dict()
                field_names = list()
                for k,kk in eval(i).items():
                    new_dict[k] = kk
                new_dict['validation'] = val_inval
                all_dat.append(new_dict)

        with open('temp.csv', 'w') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=list(all_dat[0].keys()), lineterminator='\n')
            writer.writeheader()
            for data in all_dat:
                writer.writerow(data)

        df = pd.read_csv('temp.csv',sep=',')

        df = df[df['validation']=='valid packet']

        df.to_csv('Valid_100.csv')

        self.show_dialog()

    def file_check(self):
        self.status_txt.setText('')
        input_file = QtWidgets.QFileDialog.getOpenFileName(None, 'Select File:', expanduser("."))
        res = read_text_info(input_file[0])
        res_text = str()
        combo = str(self.comboBox.currentText())

        if combo == 'tcp':
            for att_list in res:
                att_dict = generate_info_dict(att_list, 'tcp')
                res_check = check_tcp(att_dict)
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
                att_dict = generate_info_dict(att_list, 'udp')
                res_check = check_udp(att_dict)
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

        with open('readme3.txt', 'w') as f:
            f.write(res_text)


    def show_dialog(self):
        msg_box = QMessageBox()
        msg_box.information(None, "Success Save", "File Saved Successfully.")

    def save_to_csv(self):
        text = self.status_txt.toPlainText()
        text = text.split('\n')
        if text[0] == '':
            text = text[1:]
        all_dat = list()
        for num, i in enumerate(text):
            if num % 2 == 1:
                val_inval = text[num-1]
                new_dict = dict()
                field_names = list()
                print(eval(i))
                for k,kk in eval(i).items():
                    new_dict[k] = kk
                new_dict['validation'] = val_inval
                all_dat.append(new_dict)
        with open('Result.csv', 'w') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=list(all_dat[0].keys()), lineterminator='\n')
            writer.writeheader()
            for data in all_dat:
                writer.writerow(data)
        self.show_dialog()

    def retranslateUi(self, MainWindow):
        _translate = QtCore.QCoreApplication.translate
        MainWindow.setWindowTitle(_translate("MainWindow", "MainWindow"))
        self.sbytes_lbl.setText(_translate("MainWindow", "sbytes"))
        self.dbytes_lbl.setText(_translate("MainWindow", "dbytes"))
        self.sttl_lbl.setText(_translate("MainWindow", "sttl"))
        self.dttl_lbl.setText(_translate("MainWindow", "dttl"))
        self.dloss_lbl.setText(_translate("MainWindow", "dloss"))
        self.Spkts_lbl.setText(_translate("MainWindow", "Spkts"))
        self.Dload_lbl.setText(_translate("MainWindow", "Dload"))
        self.sloss_lbl.setText(_translate("MainWindow", "sloss"))
        self.Sload_lbl.setText(_translate("MainWindow", "Sload"))
        self.Dpkts_lbl.setText(_translate("MainWindow", "Dpkts"))
        self.random_btn.setText(_translate("MainWindow", "Random Info Generator"))
        self.swin_lbl.setText(_translate("MainWindow", "swin"))
        self.dwin_lbl.setText(_translate("MainWindow", "dwin"))
        self.smeansz_lbl.setText(_translate("MainWindow", "smeansz"))
        self.stcpb_lbl.setText(_translate("MainWindow", "stcpb"))
        self.res_bdy_len_lbl.setText(_translate("MainWindow", "res_bdy_len"))
        self.dtcpb_lbl.setText(_translate("MainWindow", "dtcpb"))
        self.dmeansz_lbl.setText(_translate("MainWindow", "dmeansz"))
        self.trans_depth_lbl.setText(_translate("MainWindow", "trans_depth"))
        self.Djit_lbl.setText(_translate("MainWindow", "Djit"))
        self.Sintpkt_lbl.setText(_translate("MainWindow", "Sintpkt"))
        self.Dintpkt_lbl.setText(_translate("MainWindow", "Dintpkt"))
        self.synack_lbl.setText(_translate("MainWindow", "synack"))
        self.Sjit_lbl.setText(_translate("MainWindow", "Sjit"))
        self.tcprtt_lbl.setText(_translate("MainWindow", "tcprtt"))
        self.export_btn.setText(_translate("MainWindow", "Export Result "))
        self.is_sm_ips_ports_lbl.setText(_translate("MainWindow", "is_sm_ips_ports"))
        self.ackdat_lbl.setText(_translate("MainWindow", "ackdat"))
        self.ct_state_ttl_lbl.setText(_translate("MainWindow", "ct_state_ttl"))
        self.ct_flw_http_mthd_lbl.setText(_translate("MainWindow", "ct_flw_http_mthd"))
        self.ct_ftp_cmd_lbl.setText(_translate("MainWindow", "ct_ftp_cmd"))
        self.is_ftp_login_lbl.setText(_translate("MainWindow", "is_ftp_login"))
        self.ct_src_ltm_lbl.setText(_translate("MainWindow", "ct_src_ltm"))
        self.ct_dst_ltm_lbl.setText(_translate("MainWindow", "ct_dst_ltm"))
        self.ct_srv_src_lbl.setText(_translate("MainWindow", "ct_srv_src"))
        self.ct_srv_dst_lbl.setText(_translate("MainWindow", "ct_srv_dst"))
        self.file_btn.setText(_translate("MainWindow", "Analyse File"))
        self.ct_src_dport_ltm_lbl.setText(_translate("MainWindow", "ct_src_dport_ltm"))
        self.ct_dst_sport_ltm_lbl.setText(_translate("MainWindow", "ct_dst_sport_ltm"))
        self.ct_dst_src_ltm_lbl.setText(_translate("MainWindow", "ct_dst_src_ltm"))
        self.rule_check_btn.setText(_translate("MainWindow", "Validation Check"))
        self.comboBox.setItemText(0, _translate("MainWindow", "tcp"))
        self.comboBox.setItemText(1, _translate("MainWindow", "udp"))
        self.valid_generator.setText(_translate("MainWindow", "Valid Generator"))

        self.comboBox.currentIndexChanged.connect(self.combobox_act)
        self.random_btn.clicked.connect(self.random_generate)
        self.rule_check_btn.clicked.connect(self.check)
        self.valid_generator.clicked.connect(self.valid_generator_func) 
        self.export_btn.clicked.connect(self.save_to_csv)
        self.file_btn.clicked.connect(self.file_check)




if __name__ == "__main__":
    app = QtWidgets.QApplication(sys.argv)
    MainWindow = QtWidgets.QMainWindow()
    ui = Ui_MainWindow()
    ui.setupUi(MainWindow)
    MainWindow.show()
    sys.exit(app.exec_())
