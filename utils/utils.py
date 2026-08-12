from torch.utils.data import Dataset, DataLoader
import os
from PIL import Image
from torchvision import transforms

class ImageFolderDataset(Dataset):
    def __init__(self, root, transform = None):
        super(ImageFolderDataset, self).__init__()
        self.root = root
        self.transform = transform
        self.files = [p for p in list(os.listdir(root)) if p.endswith(('.jpg', '.jpeg', '.png'))]


    def __len__(self):
        return len(self.files)

    def __getitem__(self, index):
        img_path = os.path.join(self.root, self.files[index])

        if self.transform:
            image = self.transform(Image.open(img_path))
        try:
            return image
        except Exception as e:
            raise RuntimeError(f"Failed to load image: {img_path}") from e


def get_transform(size, final_size, crop):
    transform_list = []

    if size > 0: transform_list.append(transforms.Resize(size))
    if crop: 
        transform_list.append(transforms.RandomCrop(final_size)) 
    else: 
        transform_list.append(transforms.Resize(final_size))

    transform_list.append(transforms.ToTensor())
    return transforms.Compose(transform_list)

def adaptive_instance_normalization(content_feats, style_feats):
    pass

def calc_mean_std(features, eps=1e-5):
    pass