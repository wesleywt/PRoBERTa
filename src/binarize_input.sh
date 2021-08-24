#!/bin/bash
fairseq-preprocess --only-source --trainpref split_tokenized/from/from_train.txt --validpref split_tokenized/from/from_valid.txt --testpref split_tokenized/from/from_test.txt --destdir split_binarized/input0 --workers 1 --srcdict pretraining/dict.txt

fairseq-preprocess --only-source --trainpref split_tokenized/to/to_train.txt --validpref split_tokenized/to/to_valid.txt --testpref split_tokenized/to/to_test.txt --destdir split_binarized/input1 --workers 1 --srcdict pretraining/dict.txt

fairseq-preprocess --only-source --trainpref split_tokenized/label/label_train.txt --validpref split_tokenized/label/label_valid.txt --testpref split_tokenized/label/label_test.txt --destdir split_binarized/label --workers 1





