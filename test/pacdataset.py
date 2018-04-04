import torch
from torch.utils.data import  Dataset, DataLoader
import nibabel as nib
from helperFunctions2 import get_experiment_data
import pandas as pd


class PacDataset(Dataset):
    """Pac  dataset. - taking one slice out of each image """
    @staticmethod
    def _jload(file_id):
        
        full_path = PacDataset.file_template % file_id
        img = nib.load(full_path)
        img_data = img.get_data()
        return img_data

    def __init__(self, root_dir="./Pac Data/pac2018/", train=True):
        """
        Args:
            root_dir (string): Directory with all the images.
        """
        exp0_a = get_experiment_data(0)
        if train:
            train_sel = 0
        else:
            train_sel = 1
        self.train0_df = pd.DataFrame(exp0_a[train_sel], 
                                columns=['file_id','cond','age','gender','vol','site'])
        PacDataset.file_template = root_dir + "%s.nii"
        

    def __len__(self):
        return self.train0_df.shape[0]

    def __getitem__(self, idx):
        z_cut = 60
        file_id = self.train0_df.iloc[idx]['file_id']
        img_data = PacDataset._jload(file_id)
        
        inputs = torch.Tensor(img_data[:, :, z_cut].flatten())
        return inputs