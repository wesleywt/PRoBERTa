import pandas as pd
import sys
import sentencepiece as spm
import math
import os
import bash_script_runs
from pathlib import Path

"""Input data for tokenization will be for training and evaluation. So we need to call which one it is"""
print('Argument list: ',str(sys.argv))
if sys.argv[1] == 'csv' or sys.argv == 'excel':
    input_type = sys.argv[1]
else:
    input_type = None
    print('Input type must be either csv or excel')
print(input_type)
file_path_1 = sys.argv[2]
file_path_2 = sys.argv[3]
bpe_model_path = sys.argv[4]


def from_csv(path1, path2):
    positive_dataframe = pd.read_csv(path1, usecols=['HUMAN_SEQ', 'VIRUS_SEQ'])
    positive_dataframe['label'] = 'positive'
    negative_dataframe = pd.read_csv(path2, usecols=['HUMAN_SEQ', 'VIRUS_SEQ'])
    negative_dataframe['label'] = 'negative'
    output_dataframe = pd.concat([positive_dataframe, negative_dataframe])
    output_dataframe = output_dataframe.rename(columns={'HUMAN_SEQ': 'from', 'VIRUS_SEQ': 'to'})

    return output_dataframe


def from_excel(filepath, positive_sheet, negative_sheet):
    positive_dataframe = pd.read_excel(filepath, sheet_name=positive_sheet)
    negative_dataframe = pd.read_excel(filepath, sheet_name=negative_sheet)
    positive_dataframe['label'] = 'positive'
    negative_dataframe['label'] = 'negative'
    output_dataframe = pd.concat([positive_dataframe, negative_dataframe])
    output_dataframe = output_dataframe.rename(columns={'HUMAN_SEQ': 'from', 'VIRUS_SEQ': 'to'})
    return output_dataframe


def load_bpe(path_to_model):
    model_path = path_to_model
    model = spm.SentencePieceProcessor()
    model.load(model_path)
    return model


def tokenize(df, bpe_model):
    dfv = df[['from', 'to', 'label']].values
    out = []
    for row in dfv:
        out.append([" ".join(bpe_model.encode_as_pieces(row[0])), " ".join(bpe_model.encode_as_pieces(row[1])), row[2]])
    print(out[101])
    out_df = pd.DataFrame(out, columns=['from', 'to', 'label'])
    print(out_df.head())
    #
    # out_df.to_csv(output_file_path + "tokenized_test.csv", index=False)
    # print(f"Tokenized data created in {output_file_path}")
    return out_df


def shuffle_dataframe(out_put_df):
    out_put_df_shuffled = out_put_df.sample(frac=1)
    train = math.ceil(len(out_put_df_shuffled) * 0.8)
    test = math.ceil(len(out_put_df_shuffled) * 0.8) + math.ceil(len(out_put_df_shuffled) * 0.1)

    tokenized_shuffled_80_train = out_put_df_shuffled[0:train]
    tokenized_shuffled_10_valid = out_put_df_shuffled[train:test]
    tokenized_shuffled_10_test = out_put_df_shuffled[test:]

    return tokenized_shuffled_80_train, tokenized_shuffled_10_valid, tokenized_shuffled_10_test


def create_from_to_label_lists(train, valid, test):
    from_trained = train['from'].tolist()
    from_valid = valid['from'].tolist()
    from_test = test['from'].tolist()

    from_dict = {'from_train': from_trained, 'from_valid': from_valid, 'from_test': from_test}
    for k, v in from_dict.items():
        os.makedirs('./split_tokenized/from', exist_ok=True)
        with open(f'./split_tokenized/from/{k}.txt', 'w+') as f:
            for item in v:
                f.write('%s\n' % item)

    to_trained = train['to'].tolist()
    to_valid = valid['to'].tolist()
    to_test = test['to'].tolist()

    to_dict = {'to_train': to_trained, 'to_valid': to_valid, 'to_test': to_test}
    for k, v in to_dict.items():
        os.makedirs('./split_tokenized/to', exist_ok=True)
        with open(f'./split_tokenized/to/{k}.txt', 'w') as f:
            for item in v:
                f.write('%s\n' % item)

    label_trained = train['label'].tolist()
    label_valid = valid['label'].tolist()
    label_test = test['label'].tolist()

    label_dict = {'label_train': label_trained, 'label_valid': label_valid, 'label_test': label_test}
    for k, v in label_dict.items():
        os.makedirs('./split_tokenized/label', exist_ok=True)
        with open(f'./split_tokenized/label/{k}.txt', 'w+') as f:
            for item in v:
                f.write('%s\n' % item)


def input_import(path_1, path_2='', input_type='csv'):
    if input_type == 'csv':
        input_df = from_csv(path_1, path_2)
    elif input_type == 'excel':
        input_df = from_excel(path_1, "Train_POS", "Train_NEG")
        print(input_df.head())
    else:
        input_df = None
        print('Input type not recognized')
    return input_df


def main():
    data = input_import(file_path_1, file_path_2)
    bpe_model = load_bpe(bpe_model_path)
    tokenized_df = tokenize(data, bpe_model)
    training, validate, testing = shuffle_dataframe(tokenized_df)
    create_from_to_label_lists(training, validate, testing)
    print("Training, Validation and Testing Lists written....")
    bash_script_runs.binary()
    bash_script_runs.train() # doesn't currently work


if __name__ == '__main__':
    main()
