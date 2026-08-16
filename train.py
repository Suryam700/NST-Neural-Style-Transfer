import argparse
import torch
import torch.nn as nn
from pathlib import Path
from utils.utils import get_transform, ImageFolderDataset, DataLoader, adaptive_instance_normalization, calc_mean_std
from utils.models import VGGEncoder, Decoder
import torch.optim as optim
from tqdm import tqdm
from torchvision.utils import save_image

def parse_arguments():
    parser = argparse.ArgumentParser()

    parser.add_argument('--content_dir', type=str, default=r'C:\users\rits\Documents\NST-Neural-Style-Transfer\content_data', help='Location of content dataset')
    parser.add_argument('--style_dir', type=str, default=r'C:\users\rits\Documents\NST-Neural-Style-Transfer\style_data', help='Location of style dataset')
    parser.add_argument('--vgg', type=str, default=r'C:\users\rits\Documents\NST-Neural-Style-Transfer\vgg_normalised.pth', help='Location of pre-trained VGG')
    parser.add_argument('--experiment', type=str, default='experiment1', help='Name of experiment')
    parser.add_argument('--final_size', type=int, default=256, help='Size of final image')
    parser.add_argument('--content_size', type=int, default=512, help='Size of content image')
    parser.add_argument('--style_size', type=int, default=512, help='Size of style image')
    parser.add_argument('--crop', action='store_true', default=True, help='Crop Image')
    parser.add_argument('--batch_size', type=int, default=4, help='Batch Size')
    parser.add_argument('--lr', type=float, default=1e-4, help='Learning Rate')
    parser.add_argument('--lr_decay', type=float, default=5e-5, help='Learning rate Decay')
    parser.add_argument('--epochs', type=int, default=5, help='No. of epochs')
    parser.add_argument('--content_weight', type=float, default=1.0, help='Content Weight')
    parser.add_argument('--style_weight', type=float, default=10, help='Style Weight')
    parser.add_argument('--log_interval', type=int, default=10, help='Log Interval')
    parser.add_argument('--save_interval', type=int, default=2, help='Save Interval')
    parser.add_argument('--resume', action='store_true', default=False, help='Resume Training')
    parser.add_argument('--decoder_path', type=str, default=None, help='Decoder Path')
    parser.add_argument('--optimizer_path', type=str, default=None, help='Optimizer Path')

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

    encoder = VGGEncoder(args.vgg).to(device)
    decoder = Decoder().to(device)

    optimizer = optim.Adam(decoder.parameters(), lr=args.lr)
    scheduler = optim.lr_scheduler.LambdaLR(
        optimizer,
        lr_lambda=lambda epoch: 1.0 / (1.0 + args.lr_decay * epoch)
    )

    if args.resume:
        decoder.load_state_dict(torch.load(args.decoder_path))
        optimizer.load_state_dict(torch.load(args.optimizer_path))

    encoder.eval()
    mse_loss = nn.MSELoss()

    running_loss = None
    running_style_loss = None
    running_content_loss = None

    for epoch in range(args.epochs):
        progress_bar = tqdm(zip(content_dataloader, style_dataloader), total=min(len(content_dataloader), len(style_dataloader)))

        running_loss = 0.0
        running_style_loss = 0.0
        running_content_loss = 0.0

        for content_batch, style_batch in progress_bar:
            content_batch = content_batch.to(device)
            style_batch = style_batch.to(device)

            c_feats = encoder(content_batch)
            s_feats = encoder(style_batch)

            # i/p => Encoder => encoding's => Adain Layer => o/p => Decoder => Encoder(for calculating loss)
            t = adaptive_instance_normalization(c_feats[-1], s_feats[-1])
            # now pass t from decoder
            g = decoder(t)
            g_feats = encoder(g)

            cont_loss = mse_loss(g_feats[-1], t) * args.content_weight
            style_loss = 0.0

            for g_f, s_f in zip(g_feats, c_feats):
                g_mean, g_std = calc_mean_std(g_f)
                s_mean, s_std = calc_mean_std(s_f)
                style_loss += mse_loss(g_mean, s_mean) + mse_loss(g_std, s_std)
            style_loss *= args.style_weight

            loss = style_loss + cont_loss

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            progress_bar.set_description(f"Loss: {running_loss:.4f}, Content Loss: {running_content_loss:.4f}, Style Loss: {running_style_loss:.4f}")

            running_loss += loss.item()
            running_style_loss += style_loss.item()
            running_content_loss += cont_loss.item()

        scheduler.step()
        running_loss /= len(content_dataloader)
        running_style_loss /= len(content_dataloader)
        running_content_loss /= len(content_dataloader)

        if (epoch+1) % args.log_interval == 0:
            tqdm.write(f"Iter {epoch+1}, Loss: {running_loss:.4f}, Content Loss: {running_content_loss:.4f}, Style Loss: {running_style_loss:.4f}")

        if (epoch+1) % args.log_interval == 0:
            torch.save(decoder.state_dict(), save_dir / f"decoder_{epoch+1}.pth")
            torch.save(optimizer.state_dict(), save_dir / f"optimizer_{epoch+1}.pth")

        with torch.no_grad():
            output = torch.cat([content_batch, style_batch, g], dim=0)
            save_image(output, save_dir / f"output_{epoch+1}.png", nrow=args.batch_size)

if __name__ == "__main__":
    main()