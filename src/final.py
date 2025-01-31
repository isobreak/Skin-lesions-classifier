import os
from typing import Union, Any
import gc

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, recall_score, confusion_matrix, roc_auc_score, roc_curve
import cv2
from PIL import Image
import torch
import torchvision
from torch.nn import Module
from torch.utils.data import Dataset



def get_tumor_contour(img: np.ndarray,  solve_hair: bool = True,
                      area_ratio_thresh: float = 0.65,
                      indent_ratio_thresh: float = 0.18) -> np.ndarray:
    """
    :param img: the source image\n
    :param solve_hair: if hair-problem should be solved (decreases quality of segmentation, but ignores hair)
    :param area_ratio_thresh: max suspicious_area/total_area ratio of object to be considered as tumor\n
    :param indent_ratio_thresh: max x_indent/width (or y_indent/height) ratio of object  to be considered as tumor\n
    :return: tumor contour"""

    def get_k(img: np.ndarray, intensity_thresh: int = 145) -> float:
        """Returns coefficient for exposition filter matrix corresponding given target intensity\n
        - img - source image
        - target_intensity - average intensity of resulting image"""
        def calc_intensity(img):
            """Returns average intensity of image"""
            flatten = img.flatten()
            sum = np.sum(flatten)
            return sum / len(flatten)

        k = 0.3
        while k < 2.2:
            k += 0.05
            kernel = np.array([[-1, -1, -1],
                               [-1, 9, -1],
                               [-1, -1, -1]]) * k
            intensity = calc_intensity(cv2.filter2D(img, -1, kernel))
            if intensity > intensity_thresh:
                return k

        return k

    HEIGHT = img.shape[0]
    WIDTH = img.shape[1]
    AREA_THRESH = area_ratio_thresh * HEIGHT * WIDTH
    X_INDENT_LEFT = indent_ratio_thresh * WIDTH
    X_INDENT_RIGHT = WIDTH - X_INDENT_LEFT
    Y_INDENT_UPPER = indent_ratio_thresh * HEIGHT
    Y_INDENT_LOWER = HEIGHT - Y_INDENT_UPPER

    best_contours = []
    orig = img

    for mode in range(3):
        if mode == 0:
            img = cv2.cvtColor(orig, cv2.COLOR_BGR2GRAY)
        elif mode == 1:
            img, _, _ = cv2.split(orig)
        elif mode == 2:
            img_b, img_g, _ = cv2.split(orig)
            img = cv2.merge([img_b, img_g, img_b])
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        k = get_k(img, intensity_thresh=140)
        kernel = np.array([[-1, -1, -1],
                           [-1, 9, -1],
                           [-1, -1, -1]]) * k
        img = cv2.filter2D(img, -1, kernel)
        img = cv2.medianBlur(img, 15)

        _, mask = cv2.threshold(img, 110, 255, cv2.THRESH_BINARY_INV)
        if solve_hair is True:
            open_kernel = np.ones((9, 9))
            mask = cv2.erode(mask, open_kernel, iterations=5)
            mask = cv2.dilate(mask, open_kernel, iterations=2)

        contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_TC89_L1)
        max_area = 0
        i_max = -1
        for i, contour in enumerate(contours):
            area = cv2.contourArea(contour)
            if area > max_area and area <= AREA_THRESH:
                M = cv2.moments(contour)
                cx = int(M['m10'] / M['m00'])
                cy = int(M['m01'] / M['m00'])
                if X_INDENT_LEFT < cx < X_INDENT_RIGHT and Y_INDENT_UPPER < cy < Y_INDENT_LOWER:
                    max_area = area
                    i_max = i

        if i_max != -1:
            best_contours.append(contours[i_max])

    if len(best_contours) == 0:
        # generate contour of the entire image ?
        pass
    else:
        best_ratio = cv2.arcLength(best_contours[0], True) / np.sqrt(cv2.contourArea(best_contours[0]))
        best_mode = 0
        for mode, contour in enumerate(best_contours):
            ratio = cv2.arcLength(contour, True) / np.sqrt(cv2.contourArea(contour))
            if ratio <= best_ratio:
                best_ratio = ratio
                best_mode = mode
    res_contour = best_contours[best_mode]
    res_contour = cv2.convexHull(res_contour)

    return res_contour


def get_cropped_image(img: np.ndarray, size_step: int = 224, const_res: bool = True, solve_hair: bool = True,
                      apply_mask: bool = False, draw_contour: bool = False,
                      contour_color: tuple[int, int, int] = (255, 0, 0),
                      area_ratio_thresh: float = 0.65, indent_ratio_thresh: float = 0.18, contour = None) -> np.ndarray:
    """
    Returns cropped image of (N*size_step, N*size_step) resolution, where N is a min appropriate number.
    Optionally applies a mask to the image and/or draws a contour on it
    :param img: source image
    :param size_step: size increase step (equals to min of HEIGHT and WIDTH if is set up to 0 or less)
    :param const_res: resize all images to (size_step, size_step) regardless their resolution after processing
    :param solve_hair: whether hair-problem should be solved (decreases quality of segmentation,
    but more likely ignores hair)
    :param apply_mask: apply mask to the image
    :param draw_contour: draw selected contour
    :param contour_color: color of drawn contour
    :param area_ratio_thresh: max suspicious_area/total_area ratio of object to be considered as tumor
    :param indent_ratio_thresh: max x_indent/width (or y_indent/height) ratio of object to be considered as tumor
    :param contour: existing contour to use except calculating a new one
    :return cropped image of the minimal of the available discrete sizes:
    """
    HEIGHT = img.shape[0]
    WIDTH = img.shape[1]

    if contour is None:
        contour = get_tumor_contour(img, solve_hair=solve_hair, area_ratio_thresh=area_ratio_thresh,
                                    indent_ratio_thresh=indent_ratio_thresh)

    if contour is not None:
        if apply_mask and contour is not None:
            mask = np.zeros([img.shape[0], img.shape[1]], dtype='uint8')
            cv2.drawContours(mask, [contour], -1, (255, 255, 255), -1)
            img = cv2.bitwise_and(img, img, mask=mask)
        if draw_contour and contour is not None:
            cv2.drawContours(img, [contour], -1, color=contour_color, thickness=2)

        x, y, w, h = cv2.boundingRect(contour)
    else:
        cont_size = min(HEIGHT, WIDTH)
        x, y, w, h = (WIDTH - cont_size) // 2, (HEIGHT - cont_size) // 2, cont_size, cont_size

    if size_step <= 0:
        size_step = min(WIDTH, HEIGHT)
    size = size_step
    while w > size or h > size:
        size += size_step
        if size > HEIGHT or size > WIDTH:
            size -= size_step
            w -= size_step
            break

    x_left = (size - w) // 2
    x_right = size - w - x_left
    y_upper = (size - h) // 2
    y_lower = size - h - y_upper
    y1 = y - y_upper
    y2 = y + h + y_lower
    x1 = x - x_left
    x2 = x + w + x_right

    if x1 < 0:
        shift = 0 - x1
        x1 += shift
        x2 += shift
    elif x2 > WIDTH:
        shift = x2 - WIDTH
        x1 -= shift
        x2 -= shift
    if y1 < 0:
        shift = 0 - y1
        y1 += shift
        y2 += shift
    elif y2 > HEIGHT:
        shift = y2 - HEIGHT
        y1 -= shift
        y2 -= shift

    img = img[y1:y2, x1:x2]
    if const_res:
        img = cv2.resize(img, (size_step, size_step))

    return img


def get_effnet(out: int = 1, freeze: bool = False) -> torch.nn.Module:
    """
    Perform some changes in architecture and returns model, arch (name)
    :param out: number of output neurons
    :param freeze: freeze conv layers
    :return: model
    """
    weights = torchvision.models.EfficientNet_B4_Weights.DEFAULT
    model = torchvision.models.efficientnet_b4(weights=weights)
    model.classifier = torch.nn.Sequential(
                torch.nn.Dropout(p=0.2, inplace=True),
                torch.nn.Linear(1792, out),
            )
    if freeze:
        for param in model.features.parameters():
            param.requires_grad = False
    return model


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


def predict_one_stage(img: np.ndarray, preprocess: bool = True, st_dict_path: str = None, data_path: str = None,
                      name: str = 'name', label: str = 'label') -> str:
    """
    Predicts class: 'Nevus', 'SebK', 'DF', 'BCC', or 'Melanoma' based on prediction of one model
    :param img: BGR image
    :param preprocess: specifies if input image require preprocessing
    :param st_dict_path: path to state dict with weights for model to be used
    :param data_path: path to data (images, train.csv, test.csv) which will be used for training model if st_dict_path doesn't exist
    :param name: column in both train and test dataframes to be considered as image file name
    :param label: column in both train and test dataframes to be considered as image label
    :return: class
    """
    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    seed = 42
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True

    model = get_effnet(out=5)
    if os.path.exists(st_dict_path):
        model.load_state_dict(torch.load(st_dict_path, map_location=torch.device('cpu')))
    elif os.path.exists(data_path):
        root = os.path.join(data_path, 'images')
        train_csv = pd.read_csv(os.path.join(data_path, 'train.csv'))
        test_csv = pd.read_csv(os.path.join(data_path, 'test.csv'))

        for df in [train_csv, test_csv]:
            df.rename(columns={name: 'name', label: 'label'})
            df.reset_index(drop=True, inplace=True)

        models_dict, _ = train(root, train_csv, test_csv, single_neuron=False)
        model = models_dict['best']

        torch.save(model.state_dict(), st_dict_path)
        print(f'Model has been saved at {st_dict_path}')
    else:
        pass

    if preprocess:
        img = get_cropped_image(img, size_step=224, const_res=True)

    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = img.transpose((2, 0, 1))

    img = torch.tensor(img, device=device, dtype=torch.float)
    img /= 255
    img = torch.unsqueeze(img, 0)

    img.to(device)
    model.to(device)

    transform = torchvision.transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    img = transform(img)

    with torch.no_grad():
        model.eval()
        logits = model(img)
        logits = logits.to('cpu')
        num = torch.argmax(logits, dim=1)
        labels = ['BCC', 'DF', 'Melanoma', 'Nevus', 'SebK']

        res = labels[num]

    return res


def predict_hierarchical(img: np.ndarray, preprocess: bool = True, st_dict_folder: str = None, data_path: str = None,
                         name: str = 'name', label: str = 'label') -> str:
    """
    Predicts class: 'Nevus', 'SebK', 'DF', 'BCC', or 'Melanoma' based on several models' predictions (hierarchical clf)
    :param img: BGR image
    :param preprocess: specifies if input image require preprocessing
    :param st_dict_folder: path to a folder (binary.pt, cancer.pt, benign.pt) with state dictionaries with weights for models to be used
    :param data_path: path to data (images, train.csv, test.csv) which will be used for training model if st_dict_path doesn't exist
    :param name: column in both train and test dataframes to be considered as image file name
    :param label: column in both train and test dataframes to be considered as image label
    :return: class
    """

    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    seed = 42
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True

    nodes = {
        'binary': {
            'out': 1,
            'select': lambda x: x[x['label'].str.contains('SebK|DF|Nevus|Melanoma|BCC')],
            'convert': lambda x: 'Cancer' if x == 'Melanoma' or x == 'BCC' else 'Benign',
        },
        'benign': {
            'out': 3,
            'select': lambda x: x[x['label'].str.contains('SebK|DF|Nevus')],
            'convert': None,
        },
        'cancer': {
            'out': 1,
            'select': lambda x: x[x['label'].str.contains('Melanoma|BCC')],
            'convert': None,
        },
    }

    for node_name in nodes.keys():
        node = nodes[node_name]
        out = node['out']
        select = node['select']
        convert = node['convert']
        single_neuron = True if out == 1 else False

        st_dict_path = os.path.join(st_dict_folder, node_name + '.pt')
        if not os.path.exists(st_dict_path):
            if data_path is None:
                pass
            else:
                training_files = os.listdir(data_path)
                if 'train.csv' not in training_files or 'test.csv' not in training_files or 'images' not in training_files:
                    pass

            root = os.path.join(data_path, 'images')
            train_csv = pd.read_csv(os.path.join(data_path, 'train.csv'))
            test_csv = pd.read_csv(os.path.join(data_path, 'test.csv'))

            current = []
            for x in [train_csv, test_csv]:
                x.rename(columns={name: 'name', label: 'label'})
                x = select(x)
                x.reset_index(drop=True, inplace=True)
                if convert is not None:
                    x['label'] = x['label'].apply(convert)
                current.append(x)

            models_dict, training_results = train(root, current[0], current[1], single_neuron=single_neuron)
            model = models_dict['best']

            torch.save(model.state_dict(), st_dict_path)
            print(f'Model has been saved at {st_dict_path}')

    if preprocess:
        img = get_cropped_image(img, size_step=224, const_res=True)

    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = img.transpose((2, 0, 1))

    img = torch.tensor(img, device=device, dtype=torch.float)
    img /= 255
    img = torch.unsqueeze(img, 0)

    img.to(device)

    transform = torchvision.transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
    img = transform(img)

    with torch.no_grad():
        model = get_effnet(out=1)
        st_dict_path = os.path.join(st_dict_folder, 'binary.pt')
        model.load_state_dict(torch.load(st_dict_path, map_location=torch.device('cpu')))

        model.eval()
        model.to(device)
        logit = model(img)
        logit = logit.to('cpu')
        p = torch.sigmoid(logit)

        del model
        gc.collect()
        if p > 0.5:
            model = get_effnet(out=1)
            st_dict_path = os.path.join(st_dict_folder, 'cancer.pt')
            model.load_state_dict(torch.load(st_dict_path, map_location=torch.device('cpu')))

            model.eval()
            model.to(device)
            logit = model(img)
            logit = logit.to('cpu')
            p = torch.sigmoid(logit)

            res = 'Melanoma' if p > 0.5 else 'BCC'
        else:
            model = get_effnet(out=3)
            st_dict_path = os.path.join(st_dict_folder, 'benign.pt')
            model.load_state_dict(torch.load(st_dict_path, map_location=torch.device('cpu')))

            model.eval()
            model.to(device)
            logits = model(img)
            logits = logits.to('cpu')
            num = torch.argmax(logits, dim=1)
            labels = ['DF', 'Nevus', 'SebK']

            res = labels[num]

    return res


if __name__ == "__main__":
    file_path = r'..\data\demonstration\data\images'
    data_path = r'..\data\demonstration\data'

    state_dict_path = r'..\data\demonstration\one_stage\final.pt'  # one stage
    state_dict_folder = r'..\data\demonstration\hierarchical\models'      # hierarchical

    results = []
    csv = pd.read_csv(r'..\data\demonstration\data\test.csv')
    csv_images = csv['name'].tolist()
    for image in csv_images:
        path = os.path.join(file_path, image)
        image = cv2.imread(path)
        result = predict_hierarchical(image, st_dict_folder=state_dict_folder, data_path=data_path)
        print(result)
        results.append(result)
    csv['res'] = results
    csv.to_csv(os.path.join(data_path, 'results.csv'), index=False)
    labels = csv['label'].tolist()

    count = 0
    for i in range(len(labels)):
        if labels[i] == results[i]:
            count += 1
    print(count / len(labels))

    for i, image in enumerate(csv_images, 1):
        path = os.path.join(file_path, image)
        image = cv2.imread(path)
        result = predict_one_stage(image, st_dict_path=state_dict_path, data_path=data_path)
        print(i, result)
