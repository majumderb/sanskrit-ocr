# Post-OCR Text Correction in Romanised Sanskrit

This repository contains the data and the codes (implemented in tensorflow) for the CoNLL 2018 paper "***Upcycle* Your OCR: Reusing OCRs for Post-OCR Text Correction in Romanised Sanskrit**" 

## Data

nmt/nmt_data contains the data files. 
1. Training files (train_BPE.src and train_BPE.trg) contain the BPE encoded input strings which were the output of the OCR system.
2. Testing files (test_BPE.src and test_BPE.trg) contain the BPE encoded input strings for testing. These strings are taken from *Gita* and *Sahasranama* manuscripts.
3. Vocabulary file vocab_BPE.src contains the shared vocabulary obtained after BPE. This vocab file can contain specific tokens which are only to be copied. In our experiement, we used the complete shared vocabulary as tokens which need to be copied and/or generated.

## Commands

**Test while training -** 

python2.7 -m nmt.nmt --copynet=True --share_vocab=True --attention=scaled_luong --src=src --tgt=trg --vocab_prefix=nmt/nmt_data/vocab_BPE  --train_prefix=nmt/nmt_data/train_BPE  --dev_prefix=nmt/nmt_data/valid_BPE  --test_prefix=nmt/nmt_data/test_BPE --out_dir=nmt/copynet_models --num_train_steps=12000 --steps_per_stats=100 --encoder_type=bi --num_layers=4 --num_units=128 --dropout=0.4 --metrics=bleu --check_special_token=False

**Only training -** 

python2.7 -m nmt.nmt --copynet=True --share_vocab=True --attention=scaled_luong --src=src --tgt=trg --vocab_prefix=nmt/nmt_data/vocab_BPE  --train_prefix=nmt/nmt_data/train_BPE  --dev_prefix=nmt/nmt_data/valid_BPE --out_dir=nmt/copynet_models --num_train_steps=12000 --steps_per_stats=100 --encoder_type=bi --num_layers=4 --num_units=128 --dropout=0.4 --metrics=bleu --check_special_token=False

**Test after training -** 

python2.7 -m nmt.nmt --out_dir=nmt/copynet_models --inference_input_file=nmt/my_infer_file.vi --inference_output_file=nmt/copynet_models/output_infer

**Requirements -** tesnorflow 1.5

## Remarks and Results

## CopyNet Implementation with Tensorflow and nmt

CopyNet Paper: [Incorporating Copying Mechanism in Sequence-to-Sequence Learning](https://arxiv.org/abs/1603.06393).

CopyNet mechanism is wrapped with an exsiting RNN cell and used as an normal RNN cell.

Official [nmt](https://github.com/tensorflow/nmt) is also modified to enable CopyNet  mechanism.

### Vocabulary Setting

Since in copynet scenarios the target sequence contains words from source sentences, the best choice is to use a **shared vocabulary** for source vocabulary and target vcabulary. And we also use a parameter **generated  vocabulary size**, namely, the number of target vocabulary excluding  words from source sequences (copied words), to indicate that the first N(=generated vocabulary size) words in shared vocabulary are in generate mode and target word indexes larger than N are copied.

In this codebase, `vocab_size` and `gen_vocab_size` are variables representing shared vocabulary size and generated vocabulalry size.

### Using tensorflow official nmt

Full nmt usages are in [nmt](https://github.com/tensorflow/nmt).

`--copynet` argument added to nmt command line to enable copy mechanism.

`--share_vocab` argument must be set.

`--gen_vocab_size` argument represents the size of generated vocabulary (excluding copy words from target vocabulary), if is not set, it equals the size of whole vocabulary.

```bash
python nmt.nmt.nmt.py --copynet --share_vocab --gen_vocab_size=2345 ...other_nmt_arguments
```

