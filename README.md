# pii-detection

Instructions to run the pipeline:
1. clone the TAB corpus: https://github.com/NorskRegnesentral/text-anonymization-benchmark
2. install requirements
3. run the following scripts:

preprocessing
```
python3 pre_processing.py
```

optional: train in test mode
```
python3 train.py --test_mode
```

train
```
python3 train.py
```

evaluate
```
python3 evaluate.py
```

4. find the confusion matrices and cross-validation summary in the `results` directory. Individual fold results are in each model's sub-directory.
