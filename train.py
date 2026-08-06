import argparse
import torch
from pathlib import Path
from utils.utils import *


def parse_arguments():
    parser = argparse.ArgumentParser()

    parser.add_argument('--content_dir', type=str, default=r'C:\users\rits\Documents\NST-Neural-Style-Transfer\content_data', help='Location of content dataset')
    parser.add_argument('--style_dir', type=str, default=r'C:\users\rits\Documents\NST-Neural-Style-Transfer\style_data', help='Location of style dataset')
    parser.add_argument('--vgg', type=str, default=r'C:\users\rits\Documents\NST-Neural-Style-Transfer\vgg_normalised.pth', help='Location of pre-trained VGG')
    parser.add_argument('--experiment', type=str, default='experiment1', help='Name of experiment')
    parser.add_argument('--final_size', type=int, default=512, help='Size of final image')
    parser.add_argument('--content_size', type=int, default=256, help='Size of content image')
    parser.add_argument('--style_size', type=int, default=256, help='Size of style image')
    parser.add_argument('--crop', action='store_true', default=True, help='Crop Image')

    return parser.parse_args()

def main():
    args = parse_arguments()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    save_dir = Path('experiment') / args.experiment
    save_dir.mkdir(exist_ok=True, parents=True)

    print(type(args))
    
    # Save args value's
    with open(save_dir/'args.txt', 'w') as args_file:
        for key, val in vars(args).items():
            args_file.write(f"{key}: {val}\n")

    content_transform = get_transform(args.content_size, args.final_size, args.crop)
    style_transform = get_transform(args.style_size, args.final_size, args.crop)

    content_dataset = ImageFolderDataset(args.content_dir, content_transform)
    style_dataset = ImageFolderDataset(args.style_dir, style_transform)

    content_dataloader = DataLoader(content_dataset, batch_size=args.batch_size, shuffle=True, pin_memory=True, drop_last=True)
    style_dataloader = DataLoader(style_dataset, batch_size=args.batch_size, shuffle=True, pin_memory=True, drop_last=True)

    

if __name__ == "__main__":
    main()