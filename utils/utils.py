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

        try:
            if self.transform:
                image = self.transform(Image.open(img_path).convert('RGB'))
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
    # i/p => [batch_size, channels, h, w]
    size = content_feats.size()
    style_mean, style_std = calc_mean_std(style_feats)
    content_mean, content_std = calc_mean_std(content_feats)
    norm_content_feats = (content_feats - content_mean.expand(size)) / content_std.expand(size)

    return norm_content_feats * style_std.expand(size) + style_mean.expand(size)

def calc_mean_std(features, eps=1e-5):
    # i/p => [batch_size, channels, h, w]
    # both mean nd std calculate along channel dim
    size = features.size()
    assert (len(size) == 4)
    batch_size, channels = size[: 2]
    feat_mean = features.view(batch_size, channels, -1).mean(dim=2).view(batch_size, channels, 1, 1)
    feat_var = features.view(batch_size, channels, -1).var(dim=2, unbiased=False) + eps
    feat_std = feat_var.sqrt().view(batch_size, channels, 1, 1)

    return feat_mean, feat_std