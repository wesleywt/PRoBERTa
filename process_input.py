import sentencepiece as spm
import pandas as pd

def from_csv(positive_path, negative_path):
    positive_dataframe = pd.read_csv(positive_path, usecols=['HUMAN_SEQ', 'VIRUS_SEQ'])
    positive_dataframe['label'] = 'positive'
    negative_dataframe = pd.read_csv(negative_path, usecols=['HUMAN_SEQ', 'VIRUS_SEQ'])
    negative_dataframe['label'] = 'negative'
    output_dataframe = pd.concat([positive_dataframe, negative_dataframe])
    output_dataframe = output_dataframe.rename(columns={'HUMAN_SEQ':'from', 'VIRUS_SEQ':'to'})
    return output_dataframe

def tokenize(dataframe, model):
    dfv = dataframe.values
    out = []
    for row in dfv:
        out.append([" ".join(model.encode_as_pieces(row[0])), " ".join(model.encode_as_pieces(row[1])),row[2]])
    print(out[101])
    out_df = pd.DataFrame(out, columns=dataframe.columns)
    print(out_df.head())
    out_df.to_csv('Data/H1N1_interact_tokenized.csv', index=False)



if __name__ == '__main__':
    model_path = 'BPE_model/m_reviewed.model'
    model = spm.SentencePieceProcessor()
    model.load(model_path)
    positive = 'Data/h1n1_data/training_set/H1N1_human_pos_training.csv'
    negative = 'Data/h1n1_data/training_set/H1N1_human_neg_training.csv'

    df = from_csv(positive, negative)

    print(df.head())
    tokenize(df, model)