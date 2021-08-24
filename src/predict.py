import os

from fairseq.models.roberta import RobertaModel
from fairseq.data.data_utils import collate_tokens
from scipy.special import softmax
import sys
import pandas as pd
import numpy as np
import torch

model_folder = './current_best_checkpoint'
binarized_path = 'test_binarized'
classification_head = 'protein_interaction_prediction'
output_path = 'test_predictions'
has_cuda = torch.cuda.device_count() > 0

model = RobertaModel.from_pretrained(model_folder, "checkpoint_best.pt", binarized_path, bpe=None)
model.eval()

if has_cuda:
    model.cuda()

data = pd.DataFrame()


def convert_list_to_dataframe(infile, dataframe, col=None):
    with open(infile, 'r') as f:
        list_file = f.readlines()
        dataframe[col] = list_file
        dataframe[col] = dataframe[col].str.replace(r'\n', '', regex=True)
    return dataframe
from_col = 0
to_col = 1
label_col = 2
tuple_col = 3
softmax_col = 4
pred_col = 5
from_file = 'split_tokenized/from/from_test.txt'
data = convert_list_to_dataframe(from_file, data, col=from_col)
to_file = 'split_tokenized/to/to_test.txt'
data = convert_list_to_dataframe(to_file, data, col=to_col)
label_file = 'split_tokenized/label/label_test.txt'
data = convert_list_to_dataframe(label_file, data, col=label_col)
data[label_col] = data[label_col].str.replace(" ", '_')
data.to_csv('input.tsv', sep='\t', index=None, header=None)





print(data.columns)
print(data.head())


