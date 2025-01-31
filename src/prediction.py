import os
import gc

import cv2
import numpy as np
import pandas as pd
import torch
import torchvision

from processing import get_cropped_image
from models import get_effnet
from training import train


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


def predict(img: np.ndarray, preprocess: bool = True, st_dict_path: str = None, data_path: str = None,
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

    model = get_effnet(out=3)
    if os.path.exists(st_dict_path):
        model.load_state_dict(torch.load(st_dict_path, map_location=torch.device('cpu')))
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

    state_dict_path = r'..\data\demonstration\hierarchical\models\benign.pt'
    for i, image in enumerate(csv_images, 1):
        path = os.path.join(file_path, image)
        image = cv2.imread(path)
        result = predict_hierarchical(image, st_dict_folder=state_dict_folder, data_path=data_path)
        print(i, result)
        results.append(result)
    csv['res'] = results
    csv.to_csv(os.path.join(data_path, 'hierarchical.csv'), index=False)
    labels = csv['label'].tolist()