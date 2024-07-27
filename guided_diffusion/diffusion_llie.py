import os
import numpy as np
import tqdm
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data as data
import torch.fft
from datasets import get_dataset
from functions.ckpt_util import download
import torchvision.utils as tvu
from guided_diffusion.script_util import create_model
from numpy import arange


def check_image_size(x, down_factor):
    _, _, h, w = x.size()
    mod_pad_h = (down_factor - h % down_factor) % down_factor
    mod_pad_w = (down_factor - w % down_factor) % down_factor
    x = F.pad(x, (0, mod_pad_w, 0, mod_pad_h), 'reflect')
    return x

class mix_x0that(nn.Module):
    def __init__(self):
        super(mix_x0that, self).__init__()
        self.level = nn.Parameter(torch.tensor([1.]), requires_grad=True)
    def forward(self, y_frequency, x0_t_frequency):
        return y_frequency + self.level * x0_t_frequency


class L_bri(nn.Module):
    def __init__(self, patch_size=16, mean_val=0.5):
        super(L_bri, self).__init__()
        self.pool = nn.AvgPool2d(patch_size)
        self.mean_val = mean_val
    def forward(self, x):
        x = torch.mean(x, 1, keepdim=True)
        mean = self.pool(x)
        d = torch.mean(torch.pow(mean - torch.FloatTensor([self.mean_val]).cuda(), 2))
        return d


def get_gaussian_noisy_img(img, noise_level):
    return img + torch.randn_like(img).cuda() * noise_level


def get_beta_schedule(beta_schedule, *, beta_start, beta_end, num_diffusion_timesteps):
    def sigmoid(x):
        return 1 / (np.exp(-x) + 1)

    if beta_schedule == "quad":
        betas = (
            np.linspace(
                beta_start ** 0.5,
                beta_end ** 0.5,
                num_diffusion_timesteps,
                dtype=np.float64,
            )
            ** 2
        )
    elif beta_schedule == "linear":
        betas = np.linspace(
            beta_start, beta_end, num_diffusion_timesteps, dtype=np.float64
        )
    elif beta_schedule == "const":
        betas = beta_end * np.ones(num_diffusion_timesteps, dtype=np.float64)
    elif beta_schedule == "jsd":  
        betas = 1.0 / np.linspace(
            num_diffusion_timesteps, 1, num_diffusion_timesteps, dtype=np.float64
        )
    elif beta_schedule == "sigmoid":
        betas = np.linspace(-6, 6, num_diffusion_timesteps)
        betas = sigmoid(betas) * (beta_end - beta_start) + beta_start
    else:
        raise NotImplementedError(beta_schedule)
    assert betas.shape == (num_diffusion_timesteps,)
    return betas

# Code based on DDNM
class Diffusion(object):
    def __init__(self, args, config, device=None):
        self.args = args
        self.config = config
        if device is None:
            device = (
                torch.device("cuda")
                if torch.cuda.is_available()
                else torch.device("cpu")
            )
        self.device = device

        self.model_var_type = config.model.var_type
        betas = get_beta_schedule(
            beta_schedule=config.diffusion.beta_schedule,
            beta_start=config.diffusion.beta_start,
            beta_end=config.diffusion.beta_end,
            num_diffusion_timesteps=config.diffusion.num_diffusion_timesteps,
        )
        betas = self.betas = torch.from_numpy(betas).float().to(self.device)
        self.num_timesteps = betas.shape[0]

        alphas = 1.0 - betas
        alphas_cumprod = alphas.cumprod(dim=0)
        alphas_cumprod_prev = torch.cat(
            [torch.ones(1).to(device), alphas_cumprod[:-1]], dim=0
        )
        self.alphas_cumprod_prev = alphas_cumprod_prev
        posterior_variance = (
            betas * (1.0 - alphas_cumprod_prev) / (1.0 - alphas_cumprod)
        )
        if self.model_var_type == "fixedlarge":
            self.logvar = betas.log()
        elif self.model_var_type == "fixedsmall":
            self.logvar = posterior_variance.clamp(min=1e-20).log()

    def sample(self):
        cls_fn = None
        config_dict = vars(self.config.model)
        model = create_model(**config_dict)
        if self.config.model.use_fp16:
            model.convert_to_fp16()
        if self.config.model.class_cond:
            ckpt = os.path.join(self.args.exp, 'logs/imagenet/%dx%d_diffusion.pt' % (
            self.config.data.image_size, self.config.data.image_size))
            if not os.path.exists(ckpt):
                download(
                    'https://openaipublic.blob.core.windows.net/diffusion/jul-2021/%dx%d_diffusion_uncond.pt' % (
                    self.config.data.image_size, self.config.data.image_size), ckpt)
        else:
            ckpt = os.path.join(self.args.exp, "logs/imagenet/256x256_diffusion_uncond.pt")
            if not os.path.exists(ckpt):
                download(
                    'https://openaipublic.blob.core.windows.net/diffusion/jul-2021/256x256_diffusion_uncond.pt',
                    ckpt)

        model.load_state_dict(torch.load(ckpt, map_location=self.device),strict=False)
        model.to(self.device)
        model.eval()

        print(
              f'{self.config.time_travel.T_sampling} sampling steps.',
              f'travel_length = {self.config.time_travel.travel_length},',
              f'travel_repeat = {self.config.time_travel.travel_repeat}.'
             )
        self.FourierDiff(model,cls_fn)

    def FourierDiff(self, model,cls_fn):
        args, config = self.args, self.config

        dataset, test_dataset = get_dataset(args, config)

        if args.subset_start >= 0 and args.subset_end > 0:
            assert args.subset_end > args.subset_start
            test_dataset = torch.utils.data.Subset(test_dataset, range(args.subset_start, args.subset_end))
        else:
            args.subset_start = 0
            args.subset_end = len(test_dataset)

        print(f'Dataset has size {len(test_dataset)}')

        g = torch.Generator()
        g.manual_seed(args.seed)
        val_loader = data.DataLoader(
            test_dataset,
            batch_size=config.sampling.batch_size,
            shuffle=True,
            num_workers=config.data.num_workers,
            generator=g,
        )

        args.sigma_y = 2 * args.sigma_y #to account for scaling to [-1,1]
        sigma_y = args.sigma_y
        
        print(f'Start from {args.subset_start}')

        pbar = tqdm.tqdm(val_loader)
        for (input_y, classes),name in pbar:
            mean=torch.mean(input_y)
            print(mean)
            if mean<0.1:
                print("warm-start!")
                y=input_y
                # warm-start (code from PEC)
                indicator = config.data.INDICATOR
                outer_iterN = config.data.OUTER_ITERN
                inner_iterN = config.data.INNER_ITERN
                fy = y * (1 - y)
                x = y + indicator * fy
                Isinner = config.data.ISINNER
                for outer_iter in arange(outer_iterN):
                    x0 = x
                    if Isinner == 1:
                        for inner_iter in arange(inner_iterN[outer_iter]):
                            fx = x * (1 - x)
                            x = x0 + indicator * fx
                    else:
                        x = x0
                input_y=x

            H, W = input_y.shape[2:]
            input_y = check_image_size(input_y, 32)
            H_, W_ = input_y.shape[2:]
            name = name[0].split('/')[-1]
            input_y = input_y.to(self.device)
            # FFT
            y_frequency = torch.fft.fft2(input_y, dim=(2, 3))
            # Amplitude and Phase
            y_m = torch.abs(y_frequency)
            y_p = torch.angle(y_frequency)

            if config.sampling.batch_size!=1:
                raise ValueError("please change the config file to set batch size as 1")

            # init x_T
            x = torch.randn(
                input_y.shape[0],
                config.data.channels,
                H_,
                W_,
                device=self.device,
            )

            with torch.no_grad():
                skip = config.diffusion.num_diffusion_timesteps//config.time_travel.T_sampling
                n = x.size(0)
                x0_preds = []
                xs = [x]
                
                times = get_schedule_jump(config.time_travel.T_sampling, 
                                               config.time_travel.travel_length, 
                                               config.time_travel.travel_repeat,
                                              )
                time_pairs = list(zip(times[:-1], times[1:]))
                
                
                # reverse diffusion sampling
                for i, j in tqdm.tqdm(time_pairs):
                    i, j = i*skip, j*skip
                    if j<0: j=-1 

                    if j < i: # normal sampling 
                        t = (torch.ones(n) * i).to(x.device)
                        next_t = (torch.ones(n) * j).to(x.device)
                        at = compute_alpha(self.betas, t.long())
                        at_next = compute_alpha(self.betas, next_t.long())
                        sigma_t = (1 - at_next**2).sqrt()
                        xt = xs[-1].to('cuda')

                        et = model(xt, t)


                        if et.size(1) == 6:
                            et = et[:, :3]

                        x0_t = (xt - et * (1 - at).sqrt()) / at.sqrt()
                        x0_t = (x0_t - x0_t.min()) / (x0_t.max() - x0_t.min())
                        x0_t_frequency = torch.fft.fft2(x0_t, dim=(2, 3))
                        x0_t_m = torch.abs(x0_t_frequency)
                        x0_t_p = torch.angle(x0_t_frequency)

                        mix_model = mix_x0that().cuda()
                        mix_model.train()
                        certerion = L_bri()
                        optimizer_mix = torch.optim.Adam(mix_model.parameters(), lr=1e-2)

                        if i<100:
                            with torch.enable_grad():
                                for epoch in range(50):
                                    y_plus_x0_t_frequency = mix_model(y_frequency, x0_t_frequency)
                                    y_plus_x0_t_m = torch.abs(y_plus_x0_t_frequency)
                                    y_plus_x0_t_p = torch.angle(y_plus_x0_t_frequency)
                                    x0_t_hat = (y_plus_x0_t_m) * np.e ** (1j * (y_p))
                                    x0_t_hat = torch.abs(torch.fft.ifft2(x0_t_hat, dim=(2, 3)))
                                    optimizer_mix.zero_grad()
                                    loss = certerion(x0_t_hat)
                                    loss.backward()
                                    optimizer_mix.step()
                        else:
                            y_plus_x0_t_frequency = y_frequency + x0_t_frequency
                            y_plus_x0_t_m = torch.abs(y_plus_x0_t_frequency)
                            y_plus_x0_t_p = torch.angle(y_plus_x0_t_frequency)
                            x0_t_hat = (y_plus_x0_t_m) * np.e ** (1j * (y_p))
                            x0_t_hat = torch.abs(torch.fft.ifft2(x0_t_hat, dim=(2, 3)))

                        if sigma_t >= at_next*sigma_y:
                            lambda_t = 1.
                            gamma_t = (sigma_t**2 - (at_next*sigma_y)**2).sqrt()
                        else:
                            lambda_t = (sigma_t)/(at_next*sigma_y)
                            gamma_t = 0.

                        eta = self.args.eta

                        c1 = (1 - at_next).sqrt() * eta
                        c2 = (1 - at_next).sqrt() * ((1 - eta ** 2) ** 0.5)

                        # Code form DDNM
                        xt_next = at_next.sqrt() * x0_t_hat + gamma_t * (c1 * torch.randn_like(x0_t) + c2 * et)

                        x0_preds.append(x0_t.to('cpu'))
                        xs.append(xt_next.to('cpu'))    
                    else: # time-travel back
                        next_t = (torch.ones(n) * j).to(x.device)
                        at_next = compute_alpha(self.betas, next_t.long())
                        x0_t = x0_preds[-1].to('cuda')

                        xt_next = at_next.sqrt() * x0_t + torch.randn_like(x0_t) * (1 - at_next).sqrt()

                        xs.append(xt_next.to('cpu'))

                x = xs[-1]
            x = torch.clamp(x, 0.0, 1.0)
            x = x[:, :, :H, :W]

            tvu.save_image(
                x[0], os.path.join(self.args.image_folder, f"{name}")
            )


# Code form RePaint   
def get_schedule_jump(T_sampling, travel_length, travel_repeat):
    jumps = {}
    for j in range(0, T_sampling - travel_length, travel_length):
        jumps[j] = travel_repeat - 1

    t = T_sampling
    ts = []

    while t >= 1:
        t = t-1
        ts.append(t)

        if jumps.get(t, 0) > 0:
            jumps[t] = jumps[t] - 1
            for _ in range(travel_length):
                t = t + 1
                ts.append(t)

    ts.append(-1)

    _check_times(ts, -1, T_sampling)
    return ts

def _check_times(times, t_0, T_sampling):
    # Check end
    assert times[0] > times[1], (times[0], times[1])

    # Check beginning
    assert times[-1] == -1, times[-1]

    # Steplength = 1
    for t_last, t_cur in zip(times[:-1], times[1:]):
        assert abs(t_last - t_cur) == 1, (t_last, t_cur)

    # Value range
    for t in times:
        assert t >= t_0, (t, t_0)
        assert t <= T_sampling, (t, T_sampling)
        
def compute_alpha(beta, t):
    beta = torch.cat([torch.zeros(1).to(beta.device), beta], dim=0)
    a = (1 - beta).cumprod(dim=0).index_select(0, t + 1).view(-1, 1, 1, 1)
    return a
