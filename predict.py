#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2022/9/13 15:17
# @Author  : Yu Zhang
import sys
import argparse
import pandas as pd
from model_prediction import ModelPrediction as predict
from tcr_utils import TCRUtils as utils

parser = argparse.ArgumentParser(description='manual to iTCep')
parser.add_argument("--input", type=str)
parser.add_argument("--mode", type=str, default="pairs")
parser.add_argument("--model", type=str, default="iTCep")
parser.add_argument("--dup", type=str, default="y")
parser.add_argument("--output", type=str, default="results/iTCep_pred_output.csv")


if __name__ == '__main__':
    args = parser.parse_args()
    file = args.input
    # 1. read file
    file_df = pd.DataFrame()
    file_class = file.split('.')[-1]
    if file_class == 'csv':
        file_df = pd.read_csv(file)
        file_df = utils.peptide_processing(file_df)
        if args.dup == 'y':
            file_df = file_df.drop_duplicates()
    elif file_class == 'xlsx':
        file_df = pd.read_excel(file, sheet_name='Sheet1')
        file_df = utils.data_processing(file_df)
        if args.dup == 'y':
            file_df = file_df.drop_duplicates(subset=['peptide', 'CDR3'])
    else:
        print('Parameters error: Unsupported file format!')
        exit(1)
    outfile = args.output if args.output.split('.')[-1] == 'csv' else args.output + '.csv'
    # 2. predicting
    model_list = ['iTCep', 'iTCep-PhyA']
    if args.model in model_list:
        if args.mode == 'pairs':
            predict_df = predict.model_prediction(file_df, args.model, sort=True)
            # 3. save results
            predict_df.to_csv(outfile, index=False)
        elif args.mode == 'peponly':
            peptides = file_df.peptide.tolist()
            predict_df = predict.model_prediction_TCR(peptides, args.model)
            # 3. save results
            predict_df.to_csv(outfile, index=False)
        else:
            print('Parameters error: Mode unsupported by iTCep!')
            exit(1)
    else:
        print('Parameters error: Unsupported model name!')
        exit(1)
    # print success message
    print('Success! The The iTCep program has been successfully executed.')
