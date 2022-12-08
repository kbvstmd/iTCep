#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2022/9/13 15:17
# @Author  : Yu Zhang
import sys
import pandas as pd
from model_prediction import ModelPrediction as predict
from tcr_utils import TCRUtils as utils

if __name__ == '__main__':
    sys_args = sys.argv
    file = sys_args[1]
    # 1. read file
    file_df = pd.read_excel(file, sheet_name='Sheet1')
    file_df = utils.data_processing(file_df)
    # 2. predicting
    if sys_args[2] == 'pairs':
        predict_df = predict.model_prediction(file_df, sort=True)
        # 3. save results
        predict_df.to_csv('results/pairs_pred_output.csv', index=False)
    elif sys_args[2] == 'peponly':
        file_df = file_df.drop_duplicates(subset=['peptide'])
        peptides = file_df['peptide'].values.tolist()
        predict_df = predict.model_prediction_TCR(peptides)
        # 3. save results
        predict_df.to_csv('results/peptides_pred_output.csv', index=False)
    else:
        print('parameters error!')
