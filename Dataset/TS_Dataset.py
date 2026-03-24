import numpy as np
import torch
from torch.utils.data.dataset import Dataset
import os
import glob
import torchio as tio
from torch.utils.data import DataLoader
from sdf import compute_sdf
import nibabel as nib

def nibabel_reader(path):
    img = nib.load(path)
    data = img.get_fdata(dtype=np.float32)
    tensor = torch.from_numpy(data).unsqueeze(0)
    affine = torch.from_numpy(img.affine).float()
    return tensor, affine

PREPROCESSING_TRANSORMS = tio.Compose([
    tio.Clamp(out_min=-250, out_max=450),
    tio.RescaleIntensity(in_min_max=(-250, 450),
                         out_min_max=(-1.0, 1.0)),
    tio.CropOrPad(target_shape=(64, 64, 64))
])

PREPROCESSING_MASK_TRANSORMS = tio.Compose([
    tio.CropOrPad(target_shape=(64, 64, 64))
])

TRAIN_TRANSFORMS = tio.Compose([
    tio.RandomFlip(axes=(1), flip_probability=0.5),
])


class TS_Dataset(Dataset):
    def __init__(self, root_dir='', mode = ''):
        self.root_dir = root_dir
        self.file_names = self.get_file_names()
        self.preprocessing_img = PREPROCESSING_TRANSORMS
        self.preprocessing_mask = PREPROCESSING_MASK_TRANSORMS
        self.mode = mode

    def train_transform(self, image, label, p):
        TRAIN_TRANSFORMS = tio.Compose([
            tio.RandomFlip(axes=(1), flip_probability=p),
        ])
        image = TRAIN_TRANSFORMS(image)
        label = TRAIN_TRANSFORMS(label)
        return image, label

    def get_file_names(self):
        all_img_names = glob.glob(os.path.join(self.root_dir, '**/*image.nii.gz'), recursive=True)
        return all_img_names

    def __len__(self):
        return len(self.file_names)

    def __getitem__(self, index):
        img_path = self.file_names[index]
        mask_path = img_path.replace("image.nii.gz", "label.nii.gz")

        img = tio.ScalarImage(img_path, reader=nibabel_reader)
        mask = tio.LabelMap(mask_path, reader=nibabel_reader)
        name = img_path.split('/')[-1]
        name = name.split('.nii')[0]

        img = self.preprocessing_img(img)
        mask = self.preprocessing_mask(mask)

        p = np.random.choice([0, 1])

        if self.mode == 'train':
            img, mask = self.train_transform(img, mask, p)
        
        affine = img.affine
        mask = mask.data
        img = img.data.float()

        label_1 = (mask == 1).float()
        label_1_sdf = compute_sdf(label_1)
        label_2 = (mask == 2).float()
        label_2_sdf = compute_sdf(label_2)
        label_3 = (mask == 3).float()
        label_3_sdf = compute_sdf(label_3)
        label_4 = (mask == 4).float()
        label_4_sdf = compute_sdf(label_4)
        label_5 = (mask == 5).float()
        label_5_sdf = compute_sdf(label_5)
        label = torch.cat((label_1, label_2, label_3, label_4, label_5), dim=0)
        label_sdf = torch.cat((
            torch.tensor(label_1_sdf, dtype=torch.float32),
            torch.tensor(label_2_sdf, dtype=torch.float32),
            torch.tensor(label_3_sdf, dtype=torch.float32),
            torch.tensor(label_4_sdf, dtype=torch.float32),
            torch.tensor(label_5_sdf, dtype=torch.float32),
        ), dim=0)

        return {
            'name': name,
            'img': img,
            'mask_sdf': label_sdf,
            'mask': label,
            'affine': affine
        }

def get_TS_dataloader(root_dir, mode, batch_size=1, drop_last=False, num_workers=None):
    if num_workers is None:
        num_workers = int(os.environ.get("DIFFATLAS_NUM_WORKERS", "20"))
    dataset = TS_Dataset(root_dir=root_dir, mode=mode)
    if mode == 'train':
        shuffle = True
        return dataset
    elif mode == 'test':
        shuffle = False
    else:
        raise ValueError('NO SUCH MODE')
    loader = DataLoader(
        dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers, pin_memory=True, drop_last=drop_last
    )
    return loader
