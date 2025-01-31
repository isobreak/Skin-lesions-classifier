import os
from typing import Union, Any

from PIL import Image
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import torch
import torchvision
from sklearn.metrics import accuracy_score, recall_score, confusion_matrix, roc_auc_score, roc_curve
from torch.nn import Module
from torch.utils.data import Dataset

from models import get_effnet


class ImageCsvDataset(Dataset):
    """Dataset based on folder with images and CSV file with labels. Csv must have 'name' and 'label' fields
    Csv and folder coherence is ensured"""

    def __init__(self, root: str, csv, transform: torchvision.transforms, binary: bool = False, rotate: bool = False):
        """
        :param root: path to the image folder
        :param csv: csv path or csv
        :param transform: transforms to be performed over image
        """
        if type(csv) is str:
            if os.path.exists(csv):
                df = pd.read_csv(csv).dropna()
            else:
                pass
        elif type(csv) is pd.DataFrame:
            df = csv.dropna()
            df.reset_index(drop=True, inplace=True)
        else:
            pass

        # drop inappropriate entries (not existing or with broken size)
        csv_images = df['name'].tolist()
        real_images = set(os.listdir(root))

        df.dropna(inplace=True)
        df.reset_index(inplace=True, drop=True)
        idxs = list()
        for i, image in enumerate(csv_images):
            if image not in real_images:
                idxs.append(i)

        df.drop(idxs, inplace=True)
        self.root = root
        self.df = df
        self.transform = transform
        self.classes = sorted(self.df['label'].unique().tolist())
        self.binary = binary
        self.rotate = rotate

    def __len__(self):
        return len(self.df)

    def __getitem__(self, i):
        if torch.is_tensor(i):
            i = i.tolist()
        name = self.df.iloc[i]['name']

        if self.rotate:
            r = np.random.choice(['', '_rot1', '_rot2', '_rot3'])
            f = np.random.choice(['-flip', ''])
            name = name.replace('.jpg', r + f + '.jpg')

        if os.path.exists(os.path.join(self.root, name)):
            image = Image.open(os.path.join(self.root, name))
        if self.transform:
            image = self.transform(image)

        label_number = self.classes.index(self.df.iloc[i]['label'])
        if self.binary:
            target = torch.tensor(label_number, dtype=torch.float32)
        else:
            li = [0] * len(self.classes)
            li[label_number] = 1
            target = torch.tensor(li, dtype=torch.float32)

        item = image, target

        return item


class EarlyStopper:
    def __init__(self, patience: int = 1, min_delta: float = 0):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.min_validation_loss = float('inf')

    def early_stop(self, validation_loss: float):
        if validation_loss < self.min_validation_loss:
            self.min_validation_loss = validation_loss
            self.counter = 0
        elif validation_loss > (self.min_validation_loss + self.min_delta):
            self.counter += 1
            if self.counter >= self.patience:
                return True
        return False


def fit_binary(model: torchvision.models, root: str, csv: str, test_root: str = None, test_csv: str = None,
               train_ratio: float = None,
               const_thresh: float = None,
               transforms: torchvision.transforms = None, test_transforms: torchvision.transforms = None,
               epochs: int = 25, batch_size: int = 50,
               loss: torch.nn = torch.nn.MSELoss, optimizer: torch.optim = torch.optim.Adam, opt_params: Any = None,
               scheduler: torch.optim.lr_scheduler.LRScheduler = None,
               sch_params: Any = None,
               patience: int = 20,
               min_delta: float = 0,
               print_train_metrics: bool = False, print_test_metrics: bool = False) \
               -> tuple[dict[str, Module], dict[
                   str, Union[dict[str, Union[list[float], list[Any]]], dict[str, Union[list[float], list[Any]]]]]]:
    """
    Fits model gathering information about metrics each epoch and returns tuple of dictionaries with results
    :param model: model to be trained
    :param root: path to a folder with images for both train and test (if they use separate folders, specify test_root)
    :param test_root: path to a folder with images for test. If is specified, test_csv must be specified too
    :param csv: path or csv with labels for both train and test (if they use separate csv, specify test_csv)
    :param test_csv: path or csv with labels for test
    :param train_ratio: ratio of train-test split (will be used if test_root and test_csv are not specified)
    :param const_thresh: use specified constant threshold instead of calculating it on train samples each epoch
    :param transforms: transforms to be performed over images (train or both train and test)
    :param test_transforms: transforms to be performed over test images (will be used if test_csv is specified)
    :param epochs:y number of epochs
    :param batch_size: batch size
    :param loss: loss function
    :param optimizer: optimizer to be used
    :param opt_params: dict with optimizer params
    :param scheduler: lr_scheduler to be used
    :param sch_params: params for lr_scheduler
    :param patience: patience for early stopping
    :param min_delta: min delta for early stopping
    :param print_train_metrics: whether it is necessary to print test metrics during training
    :param print_test_metrics: whether it is necessary to print test metrics during training
    :return: dict with ['best', 'last'] models, dict with ['test', 'train']['losses', 'accuracy', 'sensitivity', 'specificity', 'auc']
    """
    device = ('cuda:0' if torch.cuda.is_available() else 'cpu')
    print(f'Device: {device}')
    model = model.to(device)
    best_model = model
    best_loss = float('inf')

    # preparing dataloaders: train, test
    if test_transforms is None:
        test_transforms = transforms

    if test_csv is not None:
        if test_root is not None:
            train_dataset = ImageCsvDataset(root, csv, transforms, binary=True, rotate=True)
            test_dataset = ImageCsvDataset(test_root, test_csv, test_transforms, binary=True)
        else:
            train_dataset = ImageCsvDataset(root, csv, transforms, binary=True, rotate=True)
            test_dataset = ImageCsvDataset(root, test_csv, test_transforms, binary=True)
    else:
        if test_root is not None:
            pass
        else:
            dataset = ImageCsvDataset(root, csv, transforms, binary=True, rotate=True)
            train_size = int(len(dataset) * train_ratio)
            test_size = len(dataset) - train_size
            train_dataset, test_dataset = torch.utils.data.random_split(dataset, [train_size, test_size])

    train_size = len(train_dataset)
    test_size = len(test_dataset)

    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True,
                                               pin_memory=True, drop_last=True)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=True,
                                              pin_memory=True, drop_last=True)

    print(f'Train: {train_size}, test: {test_size}, epochs: {epochs}\nClasses:', end='')
    for c in train_dataset.classes:
        print(' ', c, end='')

    # train info
    train_losses = []
    test_losses = []
    train_acc = []
    test_acc = []
    train_rec = []
    test_rec = []
    train_auc = []
    test_auc = []

    optimizer = optimizer(model.parameters(), **opt_params)
    if scheduler is not None and sch_params is not None:
        scheduler = scheduler(optimizer, **sch_params)
    early_stopper = EarlyStopper(patience=patience, min_delta=min_delta)
    early_stopped = False

    #fitting
    best_epoch = last_epoch = 0
    for epoch in range(1, epochs + 1):
        last_epoch = epoch
        model.train()

        train_loss_sum = 0
        for b, (X_train, y_train) in enumerate(train_loader):
            optimizer.zero_grad()

            X_train = X_train.to(device)
            y_train = y_train.to(device)

            y_pred = model(X_train)
            y_pred = y_pred.reshape(batch_size)

            batch_loss = loss(y_pred, y_train)
            train_loss_sum += batch_loss.item()

            y_pred = y_pred.to('cpu')
            y_train = y_train.to('cpu').detach()
            probs_train_b = torch.nn.functional.sigmoid(y_pred).detach()
            if b == 0:
                ys_train = np.array(y_train)
                probs_train = np.array(probs_train_b)
            else:
                ys_train = np.concatenate((ys_train, y_train), axis=y_train.dim() - 1)
                probs_train = np.concatenate((probs_train, probs_train_b), axis=y_train.dim() - 1)

            batch_loss.backward()
            optimizer.step()

        if scheduler is not None:
            scheduler.step()

        auc = roc_auc_score(ys_train, probs_train)
        if const_thresh is not None:
            thresh = const_thresh
        else:
            fpr, tpr, thresholds = roc_curve(ys_train, probs_train)
            thresh = thresholds[np.argmax((1 - fpr)*tpr)]
        preds_train = (probs_train > thresh).astype(np.float32)
        acc = accuracy_score(ys_train, preds_train)
        rec = recall_score(ys_train, preds_train, average='macro')
        conf_mat = confusion_matrix(ys_train, preds_train)

        # gathering metrics
        avg_loss = train_loss_sum / train_size
        train_losses.append(avg_loss)
        train_acc.append(acc)
        train_rec.append(rec)
        train_auc.append(auc)

        if print_train_metrics is True:
            print(f'\nEpoch: {epoch}')
            print(f'Train metrics: loss = {"%.5f" % avg_loss}, acc = {"%.5f" % acc}, '
                  f'rec = {"%.5f" % rec}, auc = {"%.5f" % auc}, thresh := {"%.5f" % thresh}')
            print('Confusion matrix:\n', conf_mat)

        # estimating
        model.eval()
        with torch.no_grad():
            test_loss_sum = 0
            for b, (X_test, y_test) in enumerate(test_loader):
                X_test = X_test.to(device)
                y_test = y_test.to(device).to(torch.float32)

                y_val = model(X_test)
                y_val = y_val.reshape(batch_size)
                test_loss_sum += loss(y_val, y_test).item()

                y_val = y_val.to('cpu')
                y_test = y_test.to('cpu').detach()
                probs_test_b = torch.nn.functional.sigmoid(y_val).detach()
                if b == 0:
                    ys_test = np.array(y_test)
                    probs_test = np.array(probs_test_b)
                else:
                    ys_test = np.concatenate((ys_test, y_test), axis=y_test.dim() - 1)
                    probs_test = np.concatenate((probs_test, probs_test_b), axis=y_test.dim() - 1)

            auc = roc_auc_score(ys_test, probs_test)
            preds_test = (probs_test > thresh).astype(np.float32)
            acc = accuracy_score(ys_test, preds_test)
            rec = recall_score(ys_test, preds_test, average='macro')
            conf_mat = confusion_matrix(ys_test, preds_test)

            # gathering metrics
            avg_loss = test_loss_sum / test_size
            test_losses.append(avg_loss)
            test_acc.append(acc)
            test_rec.append(rec)
            test_auc.append(auc)

            if print_test_metrics is True:
                if print_train_metrics is not True:
                    print(f'\nEpoch: {epoch}')
                print(f'Test metrics: loss = {"%.5f" % avg_loss}, acc = {"%.5f" % acc}, '
                      f'rec = {"%.5f" % rec}, auc = {"%.5f" % auc} (thresh = {"%.5f" % thresh})')
                print('Confusion matrix:\n', conf_mat)

        # updating best model
        if  avg_loss < best_loss:
            best_epoch = epoch
            best_loss = avg_loss
            best_model = model

        if early_stopper.early_stop(avg_loss):
            early_stopped = True
            break

    return dict(best=best_model, last=model), dict(train={'loss': train_losses,
                                                          'accuracy': train_acc,
                                                          # 'auc': train_auc,
                                                          'recall': train_rec},
                                                   test={'loss': test_losses,
                                                         'accuracy': test_acc,
                                                         # 'auc': test_auc,
                                                         'recall': test_rec},
                                                   best_epoch=best_epoch,
                                                   last_epoch=last_epoch,
                                                   early_stopped=early_stopped)


def fit_multi(model: torchvision.models, root: str, csv: str, test_root: str = None, test_csv: str = None,
              train_ratio: float = None,
              transforms: torchvision.transforms = None, test_transforms: torchvision.transforms = None,
              epochs: int = 25, batch_size: int = 50,
              loss: torch.nn = torch.nn.MSELoss, optimizer: torch.optim = torch.optim.Adam, opt_params: Any = None,
              scheduler: torch.optim.lr_scheduler.LRScheduler = None,
              sch_params: Any = None,
              patience: int = 20,
              min_delta: float = 0,
              print_train_metrics: bool = False, print_test_metrics: bool = False):
    """
    Fits model gathering information about metrics each epoch and returns tuple of dictionaries with results
    :param model: model to be trained
    :param root: path to a folder with images for both train and test (if they use separate folders, specify test_root)
    :param test_root: path to a folder with images for test. If is specified, test_csv must be specified too
    :param csv: path or csv with labels for both train and test (if they use separate csv, specify test_csv)
    :param test_csv: path or csv with labels for test
    :param train_ratio: ratio of train-test split (will be used if test_root and test_csv are not specified)
    :param transforms: transforms to be performed over images (train or both train and test)
    :param test_transforms: transforms to be performed over test images (will be used if test_csv is specified)
    :param epochs:y number of epochs
    :param batch_size: batch size
    :param loss: loss function
    :param optimizer: optimizer to be used
    :param opt_params: dict with optimizer params
    :param scheduler: lr_scheduler to be used
    :param sch_params: params for lr_scheduler
    :param patience: patience for early stopping
    :param min_delta: min delta for early stopping
    :param print_train_metrics: whether it is necessary to print test metrics during training
    :param print_test_metrics: whether it is necessary to print test metrics during training
    :return: dict with ['best', 'last'] models, dict with ['test', 'train']['losses', 'accuracy', 'sensitivity', 'specificity', 'auc']
    """
    device = ('cuda:0' if torch.cuda.is_available() else 'cpu')
    print(f'Device: {device}')
    model = model.to(device)
    best_model = model
    best_loss = float('inf')

    # preparing dataloaders: train, test
    if test_transforms is None:
        test_transforms = transforms

    if test_csv is not None:
        if test_root is not None:
            train_dataset = ImageCsvDataset(root, csv, transforms, rotate=True)
            test_dataset = ImageCsvDataset(test_root, test_csv, test_transforms)
        else:
            train_dataset = ImageCsvDataset(root, csv, transforms, rotate=True)
            test_dataset = ImageCsvDataset(root, test_csv, test_transforms)
    else:
        if test_root is not None:
            pass
        else:
            dataset = ImageCsvDataset(root, csv, transforms, rotate=True)
            train_size = int(len(dataset) * train_ratio)
            test_size = len(dataset) - train_size
            train_dataset, test_dataset = torch.utils.data.random_split(dataset, [train_size, test_size])

    train_size = len(train_dataset)
    test_size = len(test_dataset)

    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True,
                                               pin_memory=True, drop_last=True)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=True,
                                              pin_memory=True, drop_last=True)
    print(f'Train: {train_size}, test: {test_size}, epochs: {epochs}\nClasses:', end='')
    for c in train_dataset.classes:
        print(' ', c, end='')

    # train info
    train_losses = []
    test_losses = []
    train_acc = []
    test_acc = []
    train_rec = []
    test_rec = []

    optimizer = optimizer(model.parameters(), **opt_params)
    if scheduler is not None and sch_params is not None:
        scheduler = scheduler(optimizer, **sch_params)
    early_stopper = EarlyStopper(patience=patience, min_delta=min_delta)
    early_stopped = False

    #fitting
    best_epoch = last_epoch = 0
    for epoch in range(1, epochs + 1):
        last_epoch = epoch
        model.train()

        train_loss_sum = 0
        for b, (X_train, y_train) in enumerate(train_loader):
            optimizer.zero_grad()

            X_train = X_train.to(device)
            y_train = y_train.to(device)

            logits = model(X_train)
            y_pred = torch.nn.functional.softmax(logits, dim=1)

            batch_loss = loss(logits, y_train)
            train_loss_sum += batch_loss.item()

            y_pred = y_pred.to('cpu')
            y_train = y_train.to('cpu')

            if b == 0:
                preds_train = torch.argmax(y_pred, dim=1).numpy()
                ys_train = torch.argmax(y_train, dim=1).numpy()
            else:
                preds_train = np.hstack([preds_train, torch.argmax(y_pred, dim=1).numpy()])
                ys_train = np.hstack([ys_train, torch.argmax(y_train, dim=1).numpy()])

            batch_loss.backward()
            optimizer.step()

        if scheduler is not None:
            scheduler.step()

        acc = accuracy_score(ys_train, preds_train)
        rec = recall_score(ys_train, preds_train, average='macro')
        conf_mat = confusion_matrix(ys_train, preds_train)

        # gathering metrics
        avg_loss = train_loss_sum / train_size
        train_losses.append(avg_loss)
        train_acc.append(acc)
        train_rec.append(rec)

        if print_train_metrics is True:
            print(f'\nEpoch: {epoch}')
            print(f'Train metrics: loss = {"%.5f" % avg_loss}, acc = {"%.5f" % acc}, '
                  f'rec = {"%.5f" % rec}')
            print('Confusion matrix:\n', conf_mat)

        # estimating
        model.eval()
        with torch.no_grad():
            test_loss_sum = 0
            for b, (X_test, y_test) in enumerate(test_loader):
                X_test = X_test.to(device)
                y_test = y_test.to(device)

                logits = model(X_test)
                y_pred = torch.nn.functional.softmax(logits, dim=1)

                test_loss_sum += loss(logits, y_test).item()

                y_pred = y_pred.to('cpu')
                y_test = y_test.to('cpu')

                if b == 0:
                    preds_test = torch.argmax(y_pred, dim=1).numpy()
                    ys_test = torch.argmax(y_test, dim=1).numpy()
                else:
                    preds_test = np.hstack([preds_test, torch.argmax(y_pred, dim=1).numpy()])
                    ys_test = np.hstack([ys_test, torch.argmax(y_test, dim=1).numpy()])

            acc = accuracy_score(ys_test, preds_test)
            rec = recall_score(ys_test, preds_test, average='macro')
            conf_mat = confusion_matrix(ys_test, preds_test)

            # gathering metrics
            avg_loss = test_loss_sum / test_size
            test_losses.append(avg_loss)
            test_acc.append(acc)
            test_rec.append(rec)

        if print_test_metrics is True:
            if print_train_metrics is not True:
                print(f'\nEpoch: {epoch}')
            print(f'Test metrics: loss = {"%.5f" % avg_loss}, acc = {"%.5f" % acc}, '
                  f'rec = {"%.5f" % rec}')
            print('Confusion matrix:\n', conf_mat)

        # updating best model
        if  avg_loss < best_loss:
            best_epoch = epoch
            best_loss = avg_loss
            best_model = model

        if early_stopper.early_stop(avg_loss):
            early_stopped = True
            break

    return dict(best=best_model, last=model), dict(train={'loss': train_losses,
                                                          'accuracy': train_acc,
                                                          'recall': train_rec},
                                                   test={'loss': test_losses,
                                                         'accuracy': test_acc,
                                                         'recall': test_rec},
                                                   best_epoch=best_epoch,
                                                   last_epoch=last_epoch,
                                                   early_stopped=early_stopped)


def save_results(models_dict: dict, res: dict,
                 models_path: str = None, performance_info_path: str = None, plots_path: str = None,
                 summary_csv_path: str = None, architecture: str = None, comment: str = None):
    """
    Saves given results according to paths
    :param models_dict: dict with 'best' and 'last' models
    :param res: dict with metrics
    :param models_path: models saving path
    :param plots_path: plots saving path
    :param performance_info_path: train info saving path
    :param summary_csv_path: csv with all models' summary info saving path
    :param architecture: architecture of used model
    :param comment: comment
    """
    # folders creation
    if models_path is not None and not os.path.exists(models_path):
        os.mkdir(models_path)
    if plots_path is not None and not os.path.exists(plots_path):
        os.mkdir(plots_path)
    if performance_info_path is not None and not os.path.exists(performance_info_path):
        os.mkdir(performance_info_path)

    # model name generation
    model_name = f'Model_0'
    if os.path.exists(summary_csv_path):
        model_name = f'Model_{len(pd.read_csv(summary_csv_path))}'
    # adding epochs info
    epochs = list(range(1, len(res['train']['loss']) + 1))
    res['train']['epoch'] = epochs
    res['test']['epoch'] = epochs

    # saving model
    if models_path is not None:
        torch.save(models_dict['best'].state_dict(), os.path.join(models_path, model_name + '_best.pt'))
        torch.save(models_dict['last'].state_dict(), os.path.join(models_path, model_name + '_last.pt'))
        print(f'Model {model_name} has been saved at {models_path}')

    # saving performance info
    if os.path.exists(performance_info_path):
        train_df = pd.DataFrame(res['train'])
        test_df = pd.DataFrame(res['test'])
        train_df.to_csv(os.path.join(performance_info_path, model_name + '_train.csv'), index=False)
        test_df.to_csv(os.path.join(performance_info_path, model_name + '_test.csv'), index=False)
        print(f'Performance info has been saved at {performance_info_path}')

    # saving plots
    metric_names = list(res['train'].keys())
    if 'epoch' in metric_names:
        metric_names.remove('epoch')
    if plots_path is not None:
        for metric_name in metric_names:
            plt.plot(epochs, res['train'][metric_name], label=f'Train {metric_name}')
            plt.plot(epochs, res['test'][metric_name], label=f'Test {metric_name}')
            plt.scatter(epochs, res['train'][metric_name])
            plt.scatter(epochs, res['test'][metric_name])
            plt.legend()
            plt.savefig(os.path.join(plots_path, model_name + f'_{metric_name}.png'))
            plt.clf()

        print(f'Plots have been saved at {plots_path}')

    # saving summary
    if summary_csv_path is not None:
        # updating existing info
        best_epoch = res['best_epoch'] - 1
        info = [res['test'][metric_name][best_epoch] for metric_name in metric_names]

        res_columns = sorted(list(res.keys()))
        for x in 'train test'.split():
            if x in res_columns:
                res_columns.remove(x)

        if os.path.exists(summary_csv_path):
            df = pd.read_csv(summary_csv_path)
        else:
            df = pd.DataFrame(columns=['model_name', 'architecture', 'comment',
                                       *metric_names, *res_columns])
        df.loc[len(df.index)] = ([model_name, architecture, comment] + info +
                                 [res[res_col] for res_col in res_columns])
        df.to_csv(summary_csv_path, index=False)
        print(f'Summary csv has been saved at {summary_csv_path}')


def train(root: str, train_csv: pd.DataFrame, test_csv: pd.DataFrame, single_neuron: bool = False):
    """
    Train model based on root, test.csv and train.csv using fixed hyperparams (lr, auto_contrast_p, gamma)
    :param root: root to image folder
    :param train_csv: df with 'name' and 'label' columns which includes information about samples for train
    :param test_csv: df with 'name' and 'label' columns which includes information about samples for test
    :param single_neuron: use 1 output neuron for prediction (only available for binary classification)
    """
    seed = 42
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True

    auto_contrast_p = 0.3
    lr = 5e-5
    gamma_substr = 0.002

    print(train_csv['label'].value_counts(), end='\n\n')

    weights = []
    for label in sorted(train_csv['label'].unique()):
        weights.append(train_csv.loc[train_csv['label'] == label, 'name'].count())
    m = min(weights)
    for i, weight in enumerate(weights):
        weights[i] = m/weights[i]

    if single_neuron:
        weight = torch.tensor(weights[1]).to('cuda:0')
        loss = torch.nn.BCEWithLogitsLoss(pos_weight=weight)
    else:
        weight = torch.tensor(weights).to('cuda:0')
        loss = torch.nn.CrossEntropyLoss(weight=weight)

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

    model = get_effnet(out=int(single_neuron or len(train_csv['label'].unique())), freeze=False)
    optimizer = torch.optim.Adam
    scheduler = torch.optim.lr_scheduler.ExponentialLR

    hyperparams = {'csv': train_csv,
                   'test_csv': test_csv,
                   'transforms': train_transforms,
                   'test_transforms': test_transforms,
                   'epochs': 65,
                   'batch_size': 9,
                   'loss': loss,
                   'optimizer': optimizer,
                   'opt_params': {
                       'lr': lr
                   },
                   'scheduler': scheduler,
                   'sch_params': {'gamma': 1 - gamma_substr},
                   'patience': 30,
                   'min_delta': 0.007,
                   }

    if single_neuron:
        models_dict, res = fit_binary(model, root=root, const_thresh=0.5,
                                      **hyperparams, print_train_metrics=True, print_test_metrics=True)
    else:
        models_dict, res = fit_multi(model, root=root,
                                     **hyperparams, print_train_metrics=True, print_test_metrics=True)

    return models_dict, res


if __name__ == "__main__":
    pass