import os.path

import optuna
import numpy as np
import pandas as pd
import torch.optim
import torchvision.transforms

from models import *
from training import save_results, fit_multi, fit_binary
from database import save_results as save_to_db, create_db, get_db_params

def objective(trial):
    root = r'..\data\datasets\current\complete datasets\images'
    test_csv = pd.read_csv(r'..\data\datasets\current\complete datasets\test.csv')
    train_csv = pd.read_csv(r'..\data\datasets\current\complete datasets\train_reduced_nevuses.csv')

    models_path = r'../data/results/models'
    performance_info_path = r'../data/results/performance info'
    plots_path = r'../data/results/plots'
    summary_csv_path = r'../data/results/summary.csv'

    seed = 42
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True

    architectures = {
        'ResNet50': get_resnet,
        # 'DenseNet121': get_densenet,
        'EfficientNetB4': get_effnet,
    }

    optimizers = {
        'Adam': torch.optim.Adam,
    }

    schedulers = {
        'ExponentialLR': torch.optim.lr_scheduler.ExponentialLR,
    }

    random_crop = True
    auto_contrast_p = trial.suggest_float('autocontrast prob', 0.15, 0.45)
    arch_name = 'EfficientNetB4'
    opt_name = 'Adam'
    sch_name = 'ExponentialLR'
    wd = 0
    lr = trial.suggest_float('lr', 1e-5, 1e-3, log=True)
    gamma_substr = 0.002

    train_csv = train_csv[train_csv['label'].str.contains('SebK|DF|Nevus')]
    test_csv = test_csv[test_csv['label'].str.contains('SebK|DF|Nevus')]
    # train_csv['label'] = train_csv['label'].apply(lambda x: 'Cancer' if x == 'Melanoma' or x == 'BCC' else 'Benign')
    # test_csv['label'] = test_csv['label'].apply(lambda x: 'Cancer' if x == 'Melanoma' or x == 'BCC' else 'Benign')

    print(train_csv['label'].value_counts(), end='\n\n')

    weights = []
    for label in sorted(train_csv['label'].unique()):
        weights.append(train_csv.loc[train_csv['label'] == label, 'name'].count())
    m = min(weights)
    for i, weight in enumerate(weights):
        weights[i] = m/weights[i]
    weight = torch.tensor(weights).to('cuda:0')
    loss = torch.nn.CrossEntropyLoss(weight=weight)
    # weight = torch.tensor(weights[1]).to('cuda:0')
    # loss = torch.nn.BCEWithLogitsLoss(pos_weight=weight)

    test_transforms = torchvision.transforms.Compose([
        torchvision.transforms.CenterCrop(224),
        torchvision.transforms.PILToTensor(),
        torchvision.transforms.ConvertImageDtype(torch.float),
        torchvision.transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    train_transforms = torchvision.transforms.Compose([
        torchvision.transforms.RandomCrop(224),
        torchvision.transforms.RandomAutocontrast(p=auto_contrast_p),
        test_transforms,
    ])

    model = architectures[arch_name](out=3, freeze=False)
    optimizer = optimizers[opt_name]
    scheduler = schedulers[sch_name]

    hyperparams = {'csv': train_csv,
                   'test_csv': test_csv,
                   'transforms': train_transforms,
                   'test_transforms': test_transforms,
                   'epochs': 150,
                   'batch_size': 9,
                   'loss': loss,
                   'optimizer': optimizer,
                   'opt_params': {
                       'weight_decay': wd,
                       'lr': lr
                   },
                   'target_metric': 'rec',
                   'scheduler': scheduler,
                   'sch_params': {'gamma': 1 - gamma_substr},
                   'patience': 20,
                   'min_delta': 0.007,
                   }

    print(f'Architecture: {arch_name}, autocontrast_p: {"%.5f" % auto_contrast_p}, lr: {lr}'
          , end='\n\n')

    models_dict, res = fit_multi(model, root=root, **hyperparams,
                                 print_train_metrics=True, print_test_metrics=True)

    save_results(models_dict, res, models_path=models_path,
                 performance_info_path=performance_info_path, plots_path=plots_path, summary_csv_path=summary_csv_path,
                 architecture=arch_name)

    # last_epoch = res['last_epoch']
    # best_epoch = res['best_epoch']
    # info_test = {
    #     'train': [False] * last_epoch,
    #     'epoch': list(range(1, last_epoch + 1)),
    #     'loss': res['test']['loss'],
    #     'accuracy': res['test']['accuracy'],
    #     'recall': res['test']['recall'],
    # }
    # info_train = {
    #     'train': [True] * last_epoch,
    #     'epoch': list(range(1, last_epoch + 1)),
    #     'loss': res['train']['loss'],
    #     'accuracy': res['train']['accuracy'],
    #     'recall': res['train']['recall'],
    # }
    # info_train = pd.DataFrame(info_train)
    # info_test = pd.DataFrame(info_test)
    # info = pd.concat([info_test, info_train], ignore_index=True)
    #
    # hyp = {
    #     'optimizer': opt_name,
    #     'lr': hyperparams['opt_params']['lr'],
    #     'wd': hyperparams['opt_params']['weight_decay'],
    #     'scheduler': sch_name,
    #     'sch_param': hyperparams['sch_params']['gamma']
    # }
    #
    # exp = {
    #     'early_stopped': res['early_stopped'],
    #     'best_epoch': best_epoch,
    #     'last_epoch': last_epoch,
    #     'best_recall': res['test']['recall'][best_epoch - 1],
    # }
    #
    # results = {
    #     'info': info,
    #     'hyp': hyp,
    #     'exp': exp,
    #     'arch_name': arch_name,
    # }
    # save_to_db(results)

    return res['test']['recall'][res['best_epoch'] - 1]


if __name__ == "__main__":
    db_name = 'optuna'
    create_db(db_name)
    db_params = get_db_params()

    study_name = 'benign_types'
    #'benign_types' 'final_all' 'final_end-to-end_reduced' 'final_base_reduced' 'mono_multi' 'benign_types_narrow'
    storage_name = f"postgresql+psycopg2://{db_params['user']}:{db_params['password']}@{db_params['host']}/{db_name}"

    study = optuna.create_study(study_name=study_name, storage=storage_name, load_if_exists=True, direction='maximize')
    # study.optimize(objective, n_trials=10)

    trials = study.trials
    for x in trials:
        if x.value is None:
            trials.remove(x)
    for i, trial in enumerate(trials):
        print(i, trial.value, trial.params)
    print(study.best_params)
    print(study.best_value)
    fig = optuna.visualization.plot_param_importances(study)
    fig.show()