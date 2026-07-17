import os
import json
import numpy as np
from taxonomy_parser import TaxonomyParser

def main():
    artifacts = [
        'models/base_model.tflite',
        'models/initial_head_weights.npz',
        'models/model_schema.json',
        'output/bootstrap_eval_report.json',
        '../data/bootstrap_seed/labels_train.csv',
        '../data/bootstrap_seed/labels_val.csv'
    ]
    missing = [a for a in artifacts if not os.path.exists(a)]
    if missing:
        print('Missing artifacts:', missing)
    else:
        print('All artifacts present.')
        
        with open('output/bootstrap_eval_report.json', 'r') as f:
            rep = json.load(f)
        
        macro_f1 = rep.get('macro avg', {}).get('f1-score', 'N/A')
        print(f'Macro F1: {macro_f1}')
        
        classes = {k: v for k, v in rep.items() if isinstance(v, dict) and 'f1-score' in v and k not in ['micro avg', 'macro avg', 'weighted avg', 'samples avg']}
        sorted_classes = sorted(classes.items(), key=lambda item: item[1].get('support', 0), reverse=True)
        print('\nTop 10 Classes by Support:')
        for k, v in sorted_classes[:10]:
            print(f"{k}: F1 {v.get('f1-score', 0):.4f}, Support {v.get('support', 0)}")
        
        zero_support = [k for k, v in classes.items() if v.get('support', 0) == 0]
        print('\nZero support classes:', zero_support)
        
        train_lines = sum(1 for _ in open('../data/bootstrap_seed/labels_train.csv')) - 1
        val_lines = sum(1 for _ in open('../data/bootstrap_seed/labels_val.csv')) - 1
        print(f'\nTotal Train Images: {train_lines}')
        print(f'Total Val Images: {val_lines}')
        
        parser = TaxonomyParser("taxonomy.json")
        print(f'NUM_CLASSES inferred from taxonomy_parser: {parser.num_classes}')

if __name__ == '__main__':
    main()
