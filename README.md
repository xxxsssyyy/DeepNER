# DeepNER
An Easy-to-use, Modular and Prolongable package of deep-learning based Named Entity Recognition Models.

This repository contains complex Deep Learning models for named entity recognition.

## Requirements
- [Tensorflow-gpu=1.14.0](https://github.com/tensorflow/tensorflow)
- [jieba=0.37](https://github.com/fxsjy/jieba)

## Train Models
* Train Transformer with CRF
```shell
python test_Transformer_CRF.py --num_blocks 2
```
* Train Bilstm with CRF
```shell
python test_BiLSTM_CRF.py
```
* fintuning bert with freezing bert Variables
```shell
python test_BiLSTM_CRF.py --freeze_bert True
```
* fintuning bert Variables simultaneously
```shell
python test_BiLSTM_CRF.py --freeze_bert False
```
## Performances Comparison
| models | Precision | Recall | F1-Score |
| :------| :------ | :------ | :------ |
| Transformer-CRF(2 Layers)  | 67.56% | 62.88% | 65.14% |
| BiGRU-CRF | 91.66% | 89.85% | 90.75% |
| BiLSTM-CRF | 91.90% | 89.85% | 90.87% |
| Bert-BiLSTM-CRF(freeze) | 94.56% |  95.09% | 94.82%  |
| Bert-BiLSTM-CRF(fintuning bert simultaneously) |  95.33% | 94.69% | 95.01% |

## TODO

* Lexicon enhance
* Label Attention Network for fine-gained NER
* Nested NER

## References

* [Bert](https://github.com/google-research/bert)
* [Chinese_NER](https://github.com/ProHiryu/bert-chinese-ner)
* [Transformer](https://github.com/DongjunLee/transformer-tensorflow)
