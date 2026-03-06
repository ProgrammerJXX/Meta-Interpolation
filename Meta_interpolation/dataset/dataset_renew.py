from torch.utils.data import Dataset, DataLoader

class Dataset_renew(Dataset):
    def __init__(self, patch_Hs, outputs, patch_names, patch_masks, ratios):
        self.patch_Hs = patch_Hs
        self.outputs = outputs
        self.patch_names = patch_names  # list of str
        self.patch_masks = patch_masks
        self.ratios = ratios            # list of float

    def __len__(self):
        return len(self.patch_Hs)

    def __getitem__(self, idx):
        return (self.patch_Hs[idx],
                self.outputs[idx],
                self.patch_names[idx],   # str
                self.patch_masks[idx],
                self.ratios[idx])        # float
