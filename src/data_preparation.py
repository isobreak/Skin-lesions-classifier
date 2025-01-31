import base64
import os
from io import BytesIO

from PIL import Image
import cv2
import numpy as np
import pandas as pd
import urllib3
import requests
from bs4 import BeautifulSoup

from processing import get_cropped_image


def parse_site(src: str) -> tuple[pd.DataFrame, requests.models.Response]:
    """
    Gathers information about name, diagnosis, href and represents in a pd.DataFrame
    (works correctly only for specific site used in project)
    :param src: url to a site
    :return: df, response
    """
    urllib3.disable_warnings()
    response = requests.get(src, verify=False)
    if response.status_code != 200:
        print('response is not 200')
        return None

    soap = BeautifulSoup(response.content, 'html.parser')
    names = list(map(lambda x: x.text, soap.find_all('td', {'style': 'color:#000;text-align:right;'})))
    diagnoses = list(map(lambda x: x.text, soap.find_all('td', {'class': 'gray'})))
    hrefs = list(map(lambda x: x['href'], soap.find_all('a', href=True)))

    if not len(names) == len(diagnoses) == len(hrefs):
        print('lens not equal')
        return None

    df = pd.DataFrame(list(zip(names, diagnoses, hrefs)), columns=['name', 'label', 'href'])
    return df, response


def get_generalized_df(df: pd.DataFrame, mode: str = 'accurate') -> pd.DataFrame:
    """
    Generalizes column 'label' in a given pd.DataFrame
    :param df: given df
    :param mode: 'cancer' for binary labeling, 'accurate' for multiclass labeling (ACCURATE IS NOT PROVIDED YET)
    :return: altered df
    """
    if mode == 'cancer':
        df.loc[df['label'].str.contains('рак|базальнокл.|меланома|мел|Боуэна', case=False), 'label'] = 'cancer'
        df['label'] = df['label'].apply(lambda x: 'cancer' if x == 'cancer' else 'benign')
    elif mode == 'accurate':
        pass

    return df


def get_coherent_df(df: pd.DataFrame, img_path: str) -> pd.DataFrame:
    """
    Ensure presence of appropriate images to all DataFrame entries (delete entries without appropriate image)
    :param df: given df
    :return: altered df
    """
    if not os.path.exists(img_path):
        print(f"Path {img_path} doesn't exist")
        return None

    csv_images = df['name'].tolist()
    real_images = set(os.listdir(img_path))

    df.dropna(inplace=True)
    for i, image in enumerate(csv_images):
        if image not in real_images:
            df.drop(i, axis=0, inplace=True)
    df.reset_index(inplace=True, drop=True)

    return df


def create_dataset(src: str, img_path: str, csv_path: str, name_suffix: str = '',
                   check_paths: bool = True, data_coherence: bool = True, generalize: bool = True,
                   res: int = None, apply_mask: bool = False):
    """
    Creates dataset (image folder and csv with labels) by parsing src site,
    optionally alters image names by adding suffix, optionally preprocess images before saving. Does not guarantee
    absense of image duplicates
    :param src: link to the site
    :param img_path: path to a folder in which images should be saved
    :param csv_path: path for saving csv
    :param name_suffix: suffix for image names (for distinction of images from different sources)
    :param check_paths: whether paths must be checked to be appropriate (folder is empty, csv doesn't exist)
    :param data_coherence: save only complementary image-label pairs
    :param generalize: generalize diagnoses
    :param res: determines resolution of image as (size_step x size_step)
    :param apply_mask: apply mask to images before saving
    """
    img_path_appr = True
    if os.path.exists(img_path) and len(os.listdir(img_path)):
        print(f'Folder {img_path} is not empty')
        img_path_appr = False

    csv_path_appr = True
    if os.path.exists(csv_path):
        print(f'CSV at {csv_path} already exists')
        csv_path_appr = False

    if check_paths and (not img_path_appr or not csv_path_appr):
        print('Can not create dataset')
        return None

    if not os.path.exists(img_path):
        os.mkdir(img_path)

    df, response = parse_site(src)
    cookies = response.cookies

    # coherence
    if data_coherence:
        df.dropna(inplace=True)

    # generalization
    if generalize:
        df = get_generalized_df(df, mode='accurate')

    # saving csv
    df['name'] = df['name'].apply(lambda x: name_suffix + x + '.jpg')
    df.drop('href', axis=1).to_csv(csv_path, encoding='UTF-16', index=False)
    print(csv_path)

    names = df['name']
    hrefs = df['href']

    # saving images
    for i in range(len(df)):
        image_name = names[i]
        path = os.path.join(img_path, image_name)
        if os.path.exists(path):
            print(image_name, 'already exists')
            continue
        image_url = src + '/' + hrefs[i].replace('teledermid', 'jpg')

        response = requests.get(image_url, verify=False, cookies=cookies)
        image = Image.open(BytesIO(response.content))
        if res is not None or apply_mask is True:
            size_step = res or 0
            image = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
            processed_image = get_cropped_image(image, size_step=size_step, const_res=True, apply_mask=apply_mask)
            cv2.imwrite(path, processed_image)
        else:
            image.save(path, format="JPEG")
        print(image_name)


def create_cropped_dataset(src_path: str, dst_path: str, size_step: int = 224, const_res: bool = True,
                           apply_mask: bool = False, solve_hair: bool = True,
                           presentation_size: int = 500, img_names: list[str] = None):
    """
    Saves cropped images from src directory to dst directory.
    Processed images by default (const_res = True) have (size_step, size_step) resolution: after resizing from
    (N * size_step, N * size_step) where N is a min appropriate
    number for tumor to fit into a cropped image
    :param src_path: path to src
    :param dst_path: path to dst
    :param size_step: size increase step (equals to min of HEIGHT and WIDTH if is set up to 0 or less)
    :param const_res: resize all images to (size_step, size_step) regardless their resolution after processing
    :param apply_mask: apply mask to the image
    :param solve_hair: if hair-problem should be solved (decreases quality of segmentation, but ignores hair)
    :param presentation_size: resolution of presentation during processing (no presentation if set up to 0 or less)
    :param img_names: only images represented in this list will be processed
    """

    if os.path.exists(src_path):
        if not os.path.exists(dst_path):
            os.mkdir(dst_path)
        if img_names is not None:
            img_names = sorted(list(set(img_names) & set(os.listdir(src_path))))
        else:
            img_names = sorted(os.listdir(src_path))
        n = len(img_names)
        for i, img_name in enumerate(img_names):
            print(i+1, n, sep=' / ', end=': ')
            if os.path.exists(os.path.join(dst_path, img_name)):
                print(img_name, 'already exists')
                continue
            else:
                print(img_name)
                img = cv2.imread(os.path.join(src_path, img_name))
                processed_img = get_cropped_image(img, size_step=size_step, const_res=const_res, apply_mask=apply_mask,
                                                  solve_hair=solve_hair)
                if presentation_size > 0:
                    pres = np.concatenate((cv2.resize(img, (presentation_size, presentation_size)),
                                           cv2.resize(processed_img, (presentation_size, presentation_size))),
                                          axis=1)
                    cv2.imshow('Original and processed images', pres)
                    cv2.waitKey(1)
                cv2.imwrite(os.path.join(dst_path, img_name), processed_img)
    else:
        print(f'Path {src_path} does not exist')


def create_augmented_dataset(src: str, dst: str = None, rotations: int = 4, flip: bool = True):
    """
    Saves rotated instances of image based on aug_numbers
    :param src: path to a dataset
    :param dst: path to a save folder (==src by default)
    :param rotations: final number of rotated instances of image
    :param flip: whether to create flipped image copies or not
    """

    dst = dst or src
    images = list(set(os.listdir(src)))

    if not os.path.exists(dst):
        os.mkdir(dst)
        print(f'Created folder at {dst}')

    # discard augmented images
    appr_images = []
    for image in images:
        if 'flip' not in image and 'rot' not in image:
            appr_images.append(image)
    print(images)

    # augmentation
    l = len(appr_images)
    for n, image in enumerate(appr_images, 1):
        src_img_path = os.path.join(src, image)
        dst_img_path = os.path.join(dst, image)
        with Image.open(src_img_path) as img:
            if not os.path.exists(dst_img_path):
                img.save(dst_img_path)
            if flip:
                img = img.transpose(Image.FLIP_LEFT_RIGHT)
                if not os.path.exists(dst_img_path.replace('.jpg', '-flip.jpg')):
                    img.save(dst_img_path.replace('.jpg', '-flip.jpg'))
            for i in range(1, rotations):
                img = img.transpose(Image.ROTATE_90)
                save_path = dst_img_path.replace(f'_rot{i-1}', '').replace('.jpg', f'_rot{i}.jpg')
                if not os.path.exists(save_path):
                    img.save(save_path)

                if flip:
                    save_path = save_path.replace('.jpg', f'-flip.jpg')
                    if not os.path.exists(save_path):
                        img.transpose(Image.FLIP_LEFT_RIGHT).save(save_path)
        print(f'{n} / {l}: {image} augmentations have been saved')

    print('Augmentation complete')


def get_augmented_df(df: pd.DataFrame, n: int = 4, flip: bool = False, name: str = 'name', label: str = 'label')\
        ->pd.DataFrame:
    """Augments given df: creates new entries by adding postfixes (_rot1, _rot2, rot_3 and -flip)\n
    Example: image.jpg -> image_rot2-flip.jpg
    :param df: DataFrame to be augmented
    :param n: number of rotations
    :param name: name of column with names of images
    :param label: name of column with labels of images
    :return augmented df"""

    res_df = df.copy()
    # add 3 rotations
    for i in range(1, n):
        add_df = df.copy()
        add_df[name] = df[name].apply(lambda x: x.replace('.jpg', f'_rot{i}.jpg'))
        res_df = pd.concat([res_df, add_df])

    # add 4 flips
    if flip:
        add_df = res_df.copy()
        add_df[name] = df[name].apply(lambda x: x.replace('.jpg', '-flip.jpg'))
        res_df = pd.concat([res_df, add_df])

    return res_df


def get_image_mask(img_path: str) -> np.ndarray:
    """
    Parse https://by-alot.me/, gains mask
    :param img_path: path to the image
    :return: masked image
    """

    img = cv2.imread(img_path)
    shape = img.shape[:2][::-1]

    url = "https://by-alot.me/get_full_mask"
    with open(img_path, "rb") as image_file:
        response = requests.post(url, files={"image": image_file})

    if response.status_code == 200:
        result = response.json()

    data_seg = result['predictions'][0]['segmentation_mask']
    mask_bytes = base64.b64decode(data_seg)
    mask_array = np.frombuffer(mask_bytes, dtype=np.uint8)

    mask_image = cv2.imdecode(mask_array, cv2.IMREAD_COLOR)
    mask_image[mask_image == 1] = 255

    flat_mask_image = mask_image.reshape(-1, 3)
    unique_colors = np.unique(flat_mask_image, axis=0)

    res_mask = cv2.resize(mask_image, (2560, 1920), interpolation=cv2.INTER_AREA)
    res_mask = res_mask.astype('uint8')
    res_mask = cv2.cvtColor(res_mask, cv2.COLOR_BGR2GRAY)
    res_mask = cv2.resize(res_mask, shape)

    return res_mask


def create_segmented_dataset(src_path: str, dst_path: str, img_names: list[str] = None):
    """
    Saves segmented images from src directory to dst directory.
    :param src_path: path to src
    :param dst_path: path to dst
    :param img_names: only images represented in this list will be processed
    """

    if os.path.exists(src_path):
        if not os.path.exists(dst_path):
            os.mkdir(dst_path)
        if img_names is not None:
            img_names = sorted(list(set(img_names) & set(os.listdir(src_path))))
        else:
            img_names = os.listdir(src_path)
        n = len(img_names)
        for i, img_name in enumerate(img_names):
            print(i+1, n, sep=' / ', end=': ')
            if os.path.exists(os.path.join(dst_path, img_name)):
                print(img_name, 'already exists')
                continue
            else:
                print(img_name)
                img = cv2.imread(os.path.join(src_path, img_name))
                mask = get_image_mask(os.path.join(src_path, img_name))
                seg_img = cv2.bitwise_and(img, img, mask=mask)
                cv2.imwrite(os.path.join(dst_path, img_name), seg_img)
    else:
        print(f'Path {src_path} does not exist')


def train_test_split(df: pd.DataFrame, size: int = 15, phrase: str = 'ISIC') -> tuple:
    """
    Split df into random train and test datasets. Test dataset doesn't contain any images with specified phrase.
    :param df: original df
    :param size: number of images for each class which must be represented in test (or all images of class)
    :param phrase: images with this phrase in name will be rejected from test dataset
    """
    df.reset_index(inplace=True, drop=True)
    labels = df['label'].unique()

    cond = (df['name'].str.contains(phrase) |
            (df['name'].str.contains('new') & (df['label']!='BCC') & (df['name'].apply(lambda x:int(x.replace('new_', '').replace('ISIC_00', '').replace('.jpg', ''))) > 1285)) |
            (df['name'].str.contains('new') & (df['label']=='BCC') & (df['name'].apply(
        lambda x: int(x.replace('new_', '').replace('ISIC_00', '').replace('.jpg', ''))) > 6149)))
    clean = df[~cond].reset_index(drop=True)
    dirty = df[cond].reset_index(drop=True)

    selected = []
    for label in labels:
        current = clean[clean['label'] == label]
        cur_size = min(len(current), size)
        selected.append(current.sample(cur_size))

    test = pd.concat(selected)
    idxs = np.ones(len(clean), dtype=bool)
    idxs[test.index] = False
    train = pd.concat([clean.iloc[idxs], dirty], ignore_index=True)
    test.reset_index(inplace=True, drop=True)
    train.reset_index(inplace=True, drop=True)

    return train, test


if __name__ == "__main__":
    pass
