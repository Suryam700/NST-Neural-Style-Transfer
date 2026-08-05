import argparse
import torch
from pathlib import Path

def parse_arguments():
    parser = argparse.ArgumentParser()

    parser.add_argument('--content_dir', type=str, default=r'C:\users\rits\Documents\NST-Neural-Style-Transfer\content_data', help='Location of content dataset')
    parser.add_argument('--style_dir', type=str, default=r'C:\users\rits\Documents\NST-Neural-Style-Transfer\style_data', help='Location of style dataset')
    parser.add_argument('--vgg', type=str, default=r'C:\users\rits\Documents\NST-Neural-Style-Transfer\vgg_normalised.pth', help='Location of pre-trained VGG')
    parser.add_argument('--experiment', type=str, default='experiment1', help='Name of experiment')


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
    

if __name__ == "__main__":
    main()