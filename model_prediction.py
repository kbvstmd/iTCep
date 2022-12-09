#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2022/9/13 15:21
# @Author  : Yu Zhang
from tensorflow.keras.models import load_model
import numpy as np
import pandas as pd
from tcr_utils import TCRUtils as utils


class ModelPrediction:

    @classmethod
    def itcepphyA_encode(cls, peptide_seqs, cdr3_seqs):
        # physicochemical property
        phychem_dict = utils.get_phychem(normal=False)
        pep_feats = utils.feat_encode(peptide_seqs, utils.max_pep_len, phychem_dict)  # (XX, 11, 21)
        cdr3_feats = utils.feat_encode(cdr3_seqs, utils.max_cdr3_len, phychem_dict)  # (XX, 21, 21)
        seqs_phychem = np.concatenate((pep_feats, cdr3_feats), axis=1)  # (XX, 32, 21)
        seqs_phychem = seqs_phychem.reshape((len(seqs_phychem), seqs_phychem.shape[1], seqs_phychem.shape[2], 1))
        # AAPP
        seqs_aapp_tcr = utils.aapp_encode_TCR(peptide_seqs)  # (XX, 20, 20)
        seqs_aapp_tcr = seqs_aapp_tcr.reshape((len(seqs_aapp_tcr), seqs_aapp_tcr.shape[1], seqs_aapp_tcr.shape[2], 1))
        return seqs_phychem, seqs_aapp_tcr

    @classmethod
    def itcep_encode(cls, peptide_seqs, cdr3_seqs):
        # onehot
        oneh_dict = utils.get_onehot()
        pep_oneh = utils.feat_encode(peptide_seqs, utils.max_pep_len, oneh_dict)  # (XX, 11, 20)
        cdr3_oneh = utils.feat_encode(cdr3_seqs, utils.max_cdr3_len, oneh_dict)  # (XX, 21, 20)
        seqs_onehot = np.concatenate((pep_oneh, cdr3_oneh), axis=1)  # (XX, 32, 20)
        seqs_onehot = seqs_onehot.reshape((len(seqs_onehot), seqs_onehot.shape[1], seqs_onehot.shape[2], 1))
        # AAPP
        seqs_aapp_tcr = utils.aapp_encode_TCR(peptide_seqs)  # (XX, 20, 20)
        seqs_aapp_tcr = seqs_aapp_tcr.reshape((len(seqs_aapp_tcr), seqs_aapp_tcr.shape[1], seqs_aapp_tcr.shape[2], 1))
        return seqs_onehot, seqs_aapp_tcr

    @classmethod
    def bind_level(cls, prob):
        if prob < 0.50:
            return '/'
        if 0.50 <= prob < 0.80:
            return 'low'
        elif 0.80 <= prob < 0.95:
            return 'medium'
        return 'high'

    @classmethod
    def model_prediction(cls, file_df, model, sort=True):
        TCR_model = load_model('models/' + model + '.h5')
        peptide_seqs, cdr3_seqs = file_df.peptide.tolist(), file_df.CDR3.tolist()
        seqs_feat1, seqs_feat2 = cls.itcep_encode(peptide_seqs, cdr3_seqs) if model == 'iTCep' else cls.itcepphyA_encode(peptide_seqs, cdr3_seqs)
        file_df['Probability'] = TCR_model.predict([seqs_feat1, seqs_feat2])[:, 1]
        if sort:
            file_df.sort_values(by="Probability", inplace=True, ascending=False)
        file_df['Probability'] = file_df['Probability'].apply(lambda x: round(x, 4))  # 保留两位小数
        file_df['Interaction'] = file_df['Probability'].apply(lambda x: 'yes' if x >= 0.5 else 'no')
        file_df['Binding level'] = file_df['Probability'].apply(lambda x: cls.bind_level(x))
        return file_df

    @classmethod
    def one_pep_result(cls, pep, model):
        print(pep)
        tcr_df = pd.read_csv('data/unique_CDR3.csv')
        tcr_df['peptide'] = pep
        file_df = cls.model_prediction(tcr_df, model)
        file_df_10 = file_df.head(10)
        return file_df_10  # 取预测值排名前10的序列对

    @classmethod
    def model_prediction_TCR(cls, peptides, model):
        result_df = pd.DataFrame()
        for pep in peptides:
            result_df = pd.concat([result_df, cls.one_pep_result(pep, model)])
        order = ['peptide', 'CDR3', 'Probability', 'Interaction', 'Binding level']
        return result_df[order]