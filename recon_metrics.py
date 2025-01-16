from tqdm import tqdm
import torch
from torch.utils.data import DataLoader, Dataset


import lpips
from ssim import ssim


class SubsetDataset(Dataset):
    def __init__(self, dataset, size):
        assert len(dataset) >= size
        self.dataset = dataset
        self.size = size

    def __len__(self):
        return self.size

    def __getitem__(self, index):
        assert index < self.size
        return self.dataset[index]


def psnr(img1, img2):
    """
    Args:
        img1: (n, c, h, w)
    """
    v_max = 1.
    # (n,)
    mse = torch.mean((img1 - img2)**2, dim=[1, 2, 3])
    return 20 * torch.log10(v_max / torch.sqrt(mse))


def make_subset_loader(eval_num_images: int,
                       dataset: Dataset,
                       batch_size: int,
                       shuffle: bool,
                       num_workers: int,
                       drop_last=True):
    dataset = SubsetDataset(dataset, size=eval_num_images)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=drop_last,
    )


def evaluate_lpips(
        sampler,  # Diffusion sampler,
        model,  # Diffusion model (with encoder)
        batch_size_eval, # conf that stores hyperparams
        device,
        eval_num_images: int,
        num_workers: int,
        val_data: Dataset,
        img_size,  # tuple of ints
):
    """
    compare the generated images from autoencoder on validation dataset

    Args:
        use_inversed_noise: the noise is also inverted from DDIM
    """
    lpips_fn = lpips.LPIPS(net='alex').to(device)
    val_loader = make_subset_loader(eval_num_images=eval_num_images,
                                    dataset=val_data,
                                    batch_size=batch_size_eval,
                                    shuffle=False,
                                    num_workers=num_workers)
    model.eval()
    with torch.no_grad():
        scores = {
            'lpips': [],
            'mse': [],
            'ssim': [],
            'psnr': [],
        }
        for batch in tqdm(val_loader, desc='lpips'):
            imgs = batch['img'].to(device)

            x_T = torch.randn((len(imgs), 3, img_size, img_size), device=device)

            # TODO: @yingheng,
            #  this is their code for generating images, you can adapt to your codebase
            pred_imgs = render_condition(model=model,
                                         x_T=x_T,
                                         x_start=imgs,
                                         cond=None,
                                         sampler=sampler)
            # (n, 1, 1, 1) => (n, )
            scores['lpips'].append(lpips_fn.forward(imgs, pred_imgs).view(-1))

            # need to normalize into [0, 1]
            norm_imgs = (imgs + 1) / 2
            norm_pred_imgs = (pred_imgs + 1) / 2
            # (n, )
            scores['ssim'].append(
                ssim(norm_imgs, norm_pred_imgs, size_average=False))
            # (n, )
            scores['mse'].append(
                (norm_imgs - norm_pred_imgs).pow(2).mean(dim=[1, 2, 3]))
            # (n, )
            scores['psnr'].append(psnr(norm_imgs, norm_pred_imgs))
        # (N, )
        for key in scores.keys():
            scores[key] = torch.cat(scores[key]).float()

    # final scores
    for key in scores.keys():
        scores[key] = torch.cat(scores[key]).mean().item()

    # {'lpips', 'mse', 'ssim'}
    return scores
