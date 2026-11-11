# PII Detection

Instructions to run the pipeline:
1. clone the TAB corpus: https://github.com/NorskRegnesentral/text-anonymization-benchmark
2. install requirements
3. run the following scripts:

- preprocessing
```
python3 pre_processing.py
```

- optional: train in test mode
```
python3 train.py --test_mode
```

- train
```
python3 train.py
```


- evaluate
```
python3 evaluate.py
```

4. find the confusion matrices and cross-validation summary in the `results` directory. Individual fold results are in each model's sub-directory.


### File structure

Note: models were too large to upload to git, but during training the models can optionally be saved to a `models` directory. 

```
.
├── evaluate.py
├── pre_processing.py
├── README.md
├── requirements.txt
├── train.py
├── data
│   ├── test.json
│   ├── train.json
│   └── valid.json
├── results
│   ├── bert_regular_best_cm.png
│   ├── bert_weighted_best_cm.png
│   ├── class_distr_test.txt
│   ├── class_distr_train.txt
│   ├── class_distr_valid.txt
│   ├── cv_summary.txt
│   ├── mbert_regular_best_cm.png
│   ├── mbert_weighted_best_cm.png
│   ├── bert_regular
│   │   ├── fold0.json
│   │   ├── fold1.json
│   │   ├── fold2.json
│   │   ├── fold3.json
│   │   └── fold4.json
│   ├── bert_weighted
│   │   ├── fold0.json
│   │   ├── fold1.json
│   │   ├── fold2.json
│   │   ├── fold3.json
│   │   └── fold4.json
│   ├── mbert_regular
│   │   ├── fold0.json
│   │   ├── fold1.json
│   │   ├── fold2.json
│   │   ├── fold3.json
│   │   └── fold4.json
│   └── mbert_weighted
│       ├── fold0.json
│       ├── fold1.json
│       ├── fold2.json
│       ├── fold3.json
│       └── fold4.json
└── utils
    ├── configs.py
    └── weighted_trainer.py
```
