import torch
import torch.nn as nn
from flowers import Flowers102
from torch.utils.data import DataLoader


model_path = 'flowerschkpt/generator-epoch200.pt'
data_dir = 'data/102flowers/jpg'
use_cuda = torch.cuda.is_available()
device = torch.device('cuda' if use_cuda else 'cpu')


def NRMSE(test_loader, model, device):
    model.eval()
    criterion = nn.MSELoss()
    all_y = []
    sum_rmse = []
    with torch.no_grad():
        for i, (data, target) in enumerate(test_loader):
            data, target = data.to(device), target.to(device)
            pred = model(data)
            all_y.append(target)
            rmse = torch.sqrt(criterion(pred, target))
            sum_rmse.append(rmse)

    mean_rmse = torch.mean(torch.stack(sum_rmse))
    y_bar = torch.mean(torch.stack(all_y))
    return mean_rmse/y_bar

def PSNR(test_loader, model, device):
    model.eval()
    criterion = nn.MSELoss()
    all_psnr = []
    with torch.no_grad():
        for i, (data, target) in enumerate(test_loader):
            data, target = data.to(device), target.to(device)
            pred = model(data)
            mse = criterion(pred, target)
            psnr = 10*torch.log10(1**2/mse)
            all_psnr.append(psnr)
    mean_psnr = torch.mean(torch.stack(all_psnr))
    return mean_psnr

if __name__ == '__main__':
    test_data = Flowers102(data_dir, 'test')
    test_loader = DataLoader(test_data, batch_size=1, shuffle=False)

    model = torch.load(model_path)
    model = model.to(device)

    nrmse = NRMSE(test_loader, model, device)
    psnr = PSNR(test_loader, model, device)
    print(f'Normalized Root Mean Squared Error: {nrmse}')
    print(f'Peak Signal to Noise Ratio: {psnr}')