
# PRoBERTa
Ananthan Nambiar, Maeve Heflin, Simon Liu, Sergei Maslov, Mark Hopkins, Anna Ritz

## Notes
- Links to Google Drive folders:

[BPE model](https://drive.google.com/drive/folders/1lJkG4IAWxSs8mGqSk-MjsaBQFV4Y3dhq?usp=sharing)

[pretraining data](https://drive.google.com/drive/folders/1glEeWDS0HoE_kYXqwySqoPpTuYstG_Dw?usp=sharing)

[family data](https://drive.google.com/drive/folders/1iXMEZuzDV1D0tstP6o_MjKzGab56_bOU?usp=sharing)

[conservative ppi data](https://drive.google.com/drive/folders/1XTCaBZOygCwoQmOZIt17dsS1XGVBq2Hu?usp=sharing)

[aggressive data](https://drive.google.com/drive/folders/1f2Z2fddQmwmVC30owNseE0lC8MSDEXBG?usp=sharing)

[pretrained weights](https://drive.google.com/drive/u/2/folders/1TbFjyRfbkLgJ_rlvO1SFB-ZvwQyykvK7)

[protein family finetuned weights](https://drive.google.com/drive/folders/1aro0V5yQgR53kLqKr9wYxjv_tVdbiw_m?usp=sharing)

[ppi conservative finetuned (20%) weights](https://drive.google.com/drive/folders/1D58Bzxm_t-MaRu7QQmb7VRlwfEvxytnD?usp=sharing)

[ppi conservative finetuned (100%) weights](https://drive.google.com/drive/folders/1djwhZE66N6SGT7qh-0yhzplpM4_H6CE-?usp=sharing)

[ppi aggressive finetuned (20%) weights](https://drive.google.com/drive/folders/1n58G7b_2ks_TPKh52FbuRrZW0LUT80t8?usp=sharing)

[ppi aggressive finetuned (100%) weights](https://drive.google.com/drive/folders/1lkzIa2DwYKiyP6TIglE2mAbSM1NnDDrB?usp=sharing)



## Requirements and Installation
sentencepiece tokenizer
```bash
pip3 install sentencepiece
```
Build [fairseq from linked repo source](https://github.com/imonlius/fairseq.git).
```bash
git clone https://github.com/imonlius/fairseq.git
cd fairseq
pip3 install --editable . --no-binary cffi
```
### tokenizer.py
Train a tokenizer and tokenize data for protein family and interaction fine-tuning

#### Example usage:
```bash
python3 tokenizer.py
```

- To change

| Name | Description |
| ----- | --------------------------------------- |
| path | Path to the protein family data. This should be a .tab file with "Sequence" and "Protein families" as two of the columns |
| int_path | Path to protein interaction data. This should be a json file with 'from', 'to' and 'link' for each interaction |

### pRoBERTa_pretrain.sh
 Pre-train RoBERTa model

#### Example Usage:
```bash
bash pRoBERTa_pretrain.sh pretrain 4 pretrained_model \
        pretraining/split_binarized/ \
        768 5 125000 3125 0.0025 32 64 3
```
- Arguments

| Name | Description | Example |
| ----- | ------------------------------------------ | ------ |
| PREFIX | Prefix for the model output files | pretrain |
| NUM_GPUS | Number of GPUs to be used during pretraining | 4 |
| OUTPUT_DIR | Output directory | [pretrained_model](https://drive.google.com/drive/u/2/folders/1fyb3RklnVWAUwajv20BP5smq9ypDgMl9) |
| DATA_DIR | Binarized input data directory | [pretraining/split_binarized/](https://drive.google.com/drive/u/2/folders/1inKxRuf5f3JBM2YDO1dQc-gTsdMn6VGR) |
| ENCODER_EMBED_DIM | Dimension of embedding generated by the encoders | 768 |
| ENCODER_LAYERS | Number of encoder layers in the model | 5 |
| TOTAL_UPDATES | Total (maximum) number of updates during training | 125000 |
| WARMUP_UPDATES | Total number of LR warm-up updates during training | 3125 |
| PEAK_LEARNING_RATE | Peak learning rate for training | 0.0025 |
| MAX_SENTENCES | Maximum number of sequences in each batch | 32 |
| UPDATE_FREQ | Updates the model every UPDATE_FREQ batches | 64 |
| PATIENCE | Early stop training if valid performance doesn’t improve for PATIENCE consecutive validation runs | 3 |



### pRoBERTa_finetune_ppi.sh: 
Fine-tune RoBERTa model for Protein Interaction Prediction Task

#### Example Usage:
```bash
bash pRoBERTa_finetune_ppi.sh ppi 4 ppi_prediction \
        ppi_prediction/split_binarized/robustness_minisplits/0.80/ \
        768 5 12500 312 0.0025 32 64 2 3 \
        pretraining/checkpoint_best.pt \
        no
```
- Arguments

| Name | Description | Example |
| ----- | ------------------------------------------ | ------ |
| PREFIX | Prefix for the model output files | ppi |
| NUM_GPUS | Number of GPUs to use for finetuning | 4 |
| OUTPUT_DIR | Model output directory | [ppi_prediction](https://drive.google.com/drive/u/2/folders/1mS34_2YTBh2wZuvn9QF7m0254bnc2LE_) |
| DATA_DIR | Binarized input data directory | [ppi_prediction/split_binarized/robustness_minisplits/1.00](https://drive.google.com/drive/u/2/folders/1kjNnud51AIPu_eeuqdapHHE-GVoaHfZm) |
| ENCODER_EMBED_DIM | Dimension of embedding generated by the encoders | 768 |
| ENCODER_LAYERS | Number of encoder layers in the model | 5 |
| TOTAL_UPDATES | Total (maximum) number of updates during training | 12500 |
| WARMUP_UPDATES | Total number of LR warm-up updates during training | 3125 |
| PEAK_LEARNING_RATE | Peak learning rate for training | 0.0025 |
| MAX_SENTENCES | Maximum number of sequences in each batch | 32 |
| UPDATE_FREQ | Updates the model every UPDATE_FREQ batches | 64 |
| PATIENCE | Early stop training if valid performance doesn’t improve for PATIENCE consecutive validation runs | 3 |
| PRETRAIN_CHECKPOINT | Path to pretrained model checkpoint | [pretraining/checkpoint_best.pt](https://drive.google.com/drive/u/2/folders/1TbFjyRfbkLgJ_rlvO1SFB-ZvwQyykvK7) |
| RESUME_TRAINING | Whether to resume training from previous finetuned model checkpoints | no |

### pRoBERTa_finetune_pfamclass.sh:
Fine-tune RoBERTa model for Family Classification Task

#### Example Usage:
```bash
bash pRoBERTa_finetune_pfamclass.sh family 4 family_classification \
        family_classification/split_binarized/robustness_minisplits/1.00 \
        768 5 12500 312 0.0025 32 64 4083 3 \
        pretraining/checkpoint_best.pt \
        no
```
- Arguments

| Name | Description | Example |
| ----- | ------------------------------------------ | ------ |
| PREFIX | Prefix for the model output files | family |
| NUM_GPUS | Number of GPUs to use for finetuning | 4 |
| OUTPUT_DIR | Model output directory | [family_classification](https://drive.google.com/drive/u/2/folders/1EGvJEAVDfPb1gcxPUsr92Tan9rPgasGm) |
| DATA_DIR | Binarized input data directory | [family_classification/split_binarized/robustness_minisplits/1.00](https://drive.google.com/drive/u/2/folders/1VxNHbwWqVZsnnZwA-6gjFtxkXB55tX3y) |
| ENCODER_EMBED_DIM | Dimension of embedding generated by the encoders | 768 |
| ENCODER_LAYERS | Number of encoder layers in the model | 5 |
| TOTAL_UPDATES | Total (maximum) number of updates during training | 12500 |
| WARMUP_UPDATES | Total number of LR warm-up updates during training | 3125 |
| PEAK_LEARNING_RATE | Peak learning rate for training | 0.0025 |
| MAX_SENTENCES | Maximum number of sequences in each batch | 32 |
| UPDATE_FREQ | Updates the model every UPDATE_FREQ batches | 64 |
| PATIENCE | Early stop training if valid performance doesn’t improve for PATIENCE consecutive validation runs | 3 |
| PRETRAIN_CHECKPOINT | Path to pretrained model checkpoint | [pretraining/checkpoint_best.pt](https://drive.google.com/drive/u/2/folders/1TbFjyRfbkLgJ_rlvO1SFB-ZvwQyykvK7) |
| RESUME_TRAINING | Whether to resume training from previous finetuned model checkpoints | no |


### Clustering/protein_family_clustering_loop.py
Cluster proteins using k-means and calculate the normalized mutual information (NMI) with protein families. Before running this make sure to download roberta.base and the relevant checkpoints.

#### Example Usage:
```bash
python3 protein_family_clustering_loop.py
```
- To change

| Name | Description |
| ----- | ----------------------------------- |
| tokenized_data_filepath | Input data filepath. This file has to contain tokenized protein sequences in a 'Tokenized Sequence' column, and the family each protein belongs to in a 'Protein families' column. Any other columns in this file will be ignored. |
| roberta_weights | depending on whether you're using a pretrained or fine-tuned model, choose the appropriate weights |
| EMBEDDING_SIZE | Should match the PRoBERTa model size |
| USE_NULL_MODEL | Whether to use random cluster prediction instead of k-means clustering |


### pRoBERTa_evaluate_family_batch.py: 
Predict families using fine-tuned RoBERTa model

#### Example Usage:
```bash
python3 pRoBERTa_evaluate_family_batch.py family_classification/split_tokenized/full/Finetune_fam_data.split.test.10 \
	family_classification/split_binarized/robustness_minisplits/1.00/ \
	predictions.tsv \
	family_classification/checkpoints/ \
	protein_family_classification 256
```
- Arguments


| Name | Description | Example |
| ----- | ------------------------------------------ | ------ |
| DATA | Path to input examples to predict. This should be formatted as a CSV with the columns, in order: tokenized sequence, true family label | [family_classification/split_tokenized/full/Finetune_fam_data.split.test.10](https://drive.google.com/drive/u/2/folders/1CvZPrtqs_JqxJVG3Fk-2FwUEC5R7NNUU) |
| BINARIZED_DATA | Path to binarized family data | [family_classification/split_binarized/robustness_minisplits/1.00/](https://drive.google.com/drive/u/2/folders/1VxNHbwWqVZsnnZwA-6gjFtxkXB55tX3y) |
| OUTPUT | Path to output file with model predictions | [predictions.tsv](https://drive.google.com/drive/u/2/folders/10gpJUzyjPCT12GfqUexOFcjFoTW9Rcr4) |
| MODEL_FOLDER | Model checkpoints folder. Will use checkpoint_best.pt file in the folder. | [family_classification/checkpoints/](https://drive.google.com/drive/u/2/folders/1JgEfybT6wT8MGzaxgAUKI7dH6W0oWLBn) |
| CLASSIFICATION_HEAD_NAME | Name of the trained classification head | protein_family_classification |
| BATCH_SIZE | Batch size for prediction | 256 |

### pRoBERTa_evaluate_ppi_batch.py: 
Predict PPI using fine-tuned RoBERTa model

#### Example Usage:
```bash
python3 pRoBERTa_evaluate_ppi_batch.py ppi_prediction/split_tokenized/full/Finetune_interact_tokenized.split.test.10 \
	ppi_prediction/split_binarized/robustness_minisplits/1.00/ \
	predictions.tsv \
	ppi_prediction/checkpoints/ \
	protein_interaction_prediction 256
```
- Arguments:


| Name | Description | Example |
| ----- | ------------------------------------------ | ------ |
| DATA | Path to input examples to predict. This should be formatted as a CSV with the columns, in order: tokenized from sequence, tokenized to sequence, true label | [ppi_prediction/split_tokenized/full/Finetune_interact_tokenized.split.test.10](https://drive.google.com/drive/u/2/folders/1GxGGOqQz5LvlLoTW3EnuEEr7fKwmu8ju) |
| BINARIZED_DATA | Path to binarized PPI data | [ppi_prediction/split_binarized/robustness_minisplits/1.00/](https://drive.google.com/drive/u/2/folders/1kjNnud51AIPu_eeuqdapHHE-GVoaHfZm) |
| OUTPUT | Path to output file with model predictions | [predictions.tsv](https://drive.google.com/drive/u/2/folders/1mS34_2YTBh2wZuvn9QF7m0254bnc2LE_) |
| MODEL_FOLDER | Model checkpoints folder. Will use checkpoint_best.pt file in the folder. | [ppi_prediction/checkpoints/](https://drive.google.com/drive/u/2/folders/1PvcqbJbgjUNMgoYhTNCsZ_a2oEAIBjxV) |
| CLASSIFICATION_HEAD_NAME | Name of the trained classification head | protein_interaction_prediction |
| BATCH_SIZE | Batch size for prediction | 256 |

### shuffle_and_split_pretrain.sh:
Shuffle and split pretraining data file into training, validation, and test data files.

#### Example Usage:
```bash
bash shuffle_and_split_pretrain.sh pretraining/tokenized_seqs_v1.txt \
	pretraining/split_tokenized/ \
	tokenized_seqs_v1
```
- Arguments:


| Name | Description | Example |
| ----- | ------------------------------------------ | ------ |
| INPUT | Input file. Each line should be an example. | [pretraining/tokenized_seqs_v1.txt](https://drive.google.com/drive/u/2/folders/1glEeWDS0HoE_kYXqwySqoPpTuYstG_Dw) |
| OUTPUT | Output directory | [pretraining/split_tokenized/](https://drive.google.com/drive/u/2/folders/10HwZrooDzT3wsY3w7GNRgwA5UFzVJ0Dj) |
| PREFIX | Prefix for output files | tokenized_seqs_v1 |

### shuffle_and_split.sh:
Shuffle and split finetuning data file into training, validation, and test data files.

#### Example Usage:
```bash
bash shuffle_and_split.sh family_classification/Finetune_fam_data.csv \
	family_classification/split_tokenized/full/ \
	Finetune_fam_data
```
- Arguments:

| Name | Description | Example |
| ----- | ------------------------------------------ | ------ |
| INPUT | Input file. Each line should be an example. | [family_classification/Finetune_fam_data.csv](https://drive.google.com/drive/u/2/folders/1mcDfv_rHsYltIW5CimMmzu7_BF7rHiN2) |
| OUTPUT | Output directory | [family_classification/split_tokenized/full/](https://drive.google.com/drive/u/2/folders/1CvZPrtqs_JqxJVG3Fk-2FwUEC5R7NNUU) |
| PREFIX | Prefix for output files | Finetune_fam_data |

### percentage_splits.sh
Generate output files with a certain percentage of the input data file

#### Example Usage:
```bash
bash percentage_splits.sh family_classification/split_tokenized/full/Finetune_fam_data.split.train.80 \
	family_classification/split_tokenized/full/robustness_split
	Finetune_fam_data
```
- Arguments:

| Name | Description | Example |
| ----- | ------------------------------------------ | ------ |
| INPUT | Input file | [family_classification/split_tokenized/full/Finetune_fam_data.split.train.80](https://drive.google.com/drive/u/2/folders/1CvZPrtqs_JqxJVG3Fk-2FwUEC5R7NNUU) |
| OUTPUT | Output directory | [family_classification/split_tokenized/full/robustness_split](https://drive.google.com/drive/u/2/folders/1EVWrfF9GVUb_b9MnNjyEHyFz0SuIapcy) |
| PREFIX | Prefix for output files| Finetune_fam_data |

### Preprocess/binarize pretraining data:
```bash
fairseq-preprocess \
	--only-source \
	--trainpref tokenized_seqs_v1.split.train.80 \
	--validpref tokenized_seqs_v1.split.valid.10 \
	--testpref tokenized_seqs_v1.split.test.10 \
	--destdir pretraining/split_binarized \
	--workers 60
```

### Preprocess/binarize family classification finetuning data:
```bash
# Split data into sequence and family files
for f in family_classification/split_tokenized/full/Finetune*; do
	cut -f1 -d',' "$f" > family_classification/split_tokenized/sequence/$(basename "$f").sequence
	cut -f2 -d',' "$f" > family_classification/split_tokenized/family/$(basename "$f").family
done

# Replace all spaces in family names with underscores
for f in family_classification/split_tokenized/family/*.family; do
	sed -i 's/ /_/g' "$f"
done

# Generate family label dictionary file
awk '{print $0,0}' family_classification/split_tokenized/family/*.family | sort | uniq > \
	family_classification/split_tokenized/family/families.txt

# Binarize sequences
fairseq-preprocess \
	--only-source \
	--trainpref family_classification/split_tokenized/sequence/Finetune_fam_data.split.train.80.sequence
        --validpref family_classification/split_tokenized/sequence/Finetune_fam_data.split.valid.10.sequence
        --testpref family_classification/split_tokenized/sequence/Finetune_fam_data.split.test.10.sequence
	--destdir family_classification/split_binarized/input0
	--workers 60
	--srcdict pretraining/split_binarized/dict.txt

# Binarize labels
fairseq-preprocess \
	--only-source \
	--trainpref family_classification/split_tokenized/family/Finetune_fam_data.split.train.80.family
	--validpref family_classification/split_tokenized/family/Finetune_fam_data.split.valid.10.family
	--testpref family_classification/split_tokenized/family/Finetune_fam_data.split.test.10.family 
	--destdir family_classification/split_binarized/label
	--workers 60
	--srcdict family_classification/split_tokenized/family/families.txt
```

### Preprocess/binarize PPI data:
```bash
# Split data into from sequence, to sequence, and label files
for f in ppi_prediction/split_tokenized/full/Finetune*; do
        cut -f1 -d',' "$f" > ppi_prediction/split_tokenized/from/$(basename "$f").from
        cut -f2 -d',' "$f" > ppi_prediction/split_tokenized/to/$(basename "$f").to
	cut -f2 -d',' "$f" > ppi_prediction/split_tokenized/label/$(basename "$f").label
done

# Binarize sequences
fairseq-preprocess \
        --only-source \
        --trainpref ppi_prediction/split_tokenized/from/Finetune_interact_tokenized.split.train.80.from
        --validpref ppi_prediction/split_tokenized/from/Finetune_interact_tokenized.split.valid.10.from
        --testpref ppi_prediction/split_tokenized/from/Finetune_interact_tokenized.split.test.10.from
        --destdir ppi_prediction/split_binarized/input0
        --workers 60
        --srcdict pretraining/split_binarized/dict.txt

fairseq-preprocess \
        --only-source \
        --trainpref ppi_prediction/split_tokenized/to/Finetune_interact_tokenized.split.train.80.to
        --validpref ppi_prediction/split_tokenized/to/Finetune_interact_tokenized.split.valid.10.to
        --testpref ppi_prediction/split_tokenized/to/Finetune_interact_tokenized.split.test.10.to
        --destdir ppi_prediction/split_binarized/input1
        --workers 60
        --srcdict pretraining/split_binarized/dict.txt

# Binarize labels
fairseq-preprocess \
	--only-source \
	--trainpref ppi_prediction/split_tokenized/label/Finetune_interact_tokenized.split.train.80.label
        --validpref ppi_prediction/split_tokenized/label/Finetune_interact_tokenized.split.valid.10.label
        --testpref ppi_prediction/split_tokenized/label/Finetune_interact_tokenized.split.test.10.label
	--destdir ppi_prediction/split_binarized/label
	--workers 60
```

