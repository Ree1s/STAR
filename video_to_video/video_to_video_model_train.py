import os
import os.path as osp
import random
from typing import Any, Dict

import torch
import torch.cuda.amp as amp
import torch.nn.functional as F

from video_to_video.modules import *
from video_to_video.utils.config import cfg
from video_to_video.diffusion.diffusion_ddim import DiffusionDDIM
from video_to_video.diffusion.schedules_sdedit import noise_schedule
from video_to_video.utils.logger import get_logger
from modelscope.models import TorchModel
from diffusers import AutoencoderKLTemporalDecoder

logger = get_logger()
def get_low_pass_mask(height: int, width: int, cutoff_ratio: float = 0.1) -> torch.Tensor:
    """
    Create a 2D low-pass filter mask in the frequency domain.
    
    Args:
        height (int): Height of the feature map.
        width (int): Width of the feature map.
        cutoff_ratio (float): Ratio to determine the radius of low-frequency region.
    
    Returns:
        torch.Tensor: The low-pass mask with shape [1, 1, H, W] for broadcasting.
    """
    y = torch.linspace(-0.5, 0.5, height, device='cuda')
    x = torch.linspace(-0.5, 0.5, width, device='cuda')
    yy, xx = torch.meshgrid(y, x, indexing="ij")
    dist = torch.sqrt(xx**2 + yy**2)
    cutoff = cutoff_ratio / 2.0
    mask = (dist <= cutoff).float()
    return mask.unsqueeze(0).unsqueeze(0)  # shape: [1, 1, H, W]

def compute_df_loss(
    model_pred_cond: torch.Tensor,
    noise: torch.Tensor,
    t: torch.Tensor,
    diffusion,
    vae,
    ground_truth: torch.Tensor,
    df_alpha: float,
    chunk_size: int = 3,
    cutoff_ratio: float = 0.1,
    t_max: float = 999.0
) -> torch.Tensor:
    """
    Compute the Dynamic Frequency (DF) loss on the conditional branch.
    
    The process is as follows:
    1. Invert the predicted noise to estimate the high-res latent using the scaling factors αₜ and σₜ.
    2. Decode the latent to pixel space.
    3. Compute the 2D FFT of both the predicted and ground truth high-res videos.
    4. Use a predefined low-pass filter to separate low- and high-frequency components.
    5. Compute an MSE loss on both frequency bands, then weight them using a function of the timestep.

    Args:
        model_pred_cond (torch.Tensor): The generator output for the conditional branch (shape: [B, ...]).
        noise (torch.Tensor): The noise tensor used in the diffusion process.
        t (torch.Tensor): The timestep tensor (shape: [B]).
        diffusion: An object providing diffusion-related functions (e.g. diffuse, get_scalings).
        vae: The VAE model (or its decoder method) to decode latents back to pixel space.
        ground_truth (torch.Tensor): The high-resolution ground truth video (shape: [B, F, C, H, W]).
        df_alpha (float): Hyperparameter controlling the weighting function c(t).
        chunk_size (int): Chunk size used in decoding (if your implementation uses chunking).
        cutoff_ratio (float): Cutoff ratio for creating the low-pass filter.
        t_max (float): Maximum diffusion timestep (used in weighting functions).

    Returns:
        torch.Tensor: The computed dynamic frequency loss.
    """
    # Obtain scaling factors α and σ for the timesteps.
    sigma = diffusion._t_to_sigma(t).repeat(noise.shape[0], 1, 1, 1, 1)

    # Invert predicted noise (or velocity) to get estimated high-res latent.
    hat_Z_H = (alpha * noise - model_pred_cond) / sigma
    # Decode the latent back to pixel space.
    hat_X_H = vae.vae_decode_chunk(hat_Z_H, chunk_size=chunk_size)  # shape: [B, C, F, H, W]

    # Reshape predicted and ground truth videos for FFT computation.
    B, C, F, H, W = hat_X_H.shape
    hat_X_H_reshaped = hat_X_H.permute(0, 2, 1, 3, 4).reshape(B * F, C, H, W)
    gt_reshaped = ground_truth.reshape(B * ground_truth.shape[1],
                                        ground_truth.shape[2],
                                        ground_truth.shape[3],
                                        ground_truth.shape[4])
    
    # Compute 2D FFT (using orthonormal normalization).
    fft_hat = torch.fft.fft2(hat_X_H_reshaped, norm="ortho")
    fft_gt = torch.fft.fft2(gt_reshaped, norm="ortho")
    # Compare magnitudes.
    mag_hat = torch.abs(fft_hat)
    mag_gt = torch.abs(fft_gt)
    
    # Create a low-pass filter mask.
    psi = get_low_pass_mask(H, W, cutoff_ratio)  # shape: [1, 1, H, W]
    low_hat = mag_hat * psi
    high_hat = mag_hat * (1 - psi)
    low_gt = mag_gt * psi
    high_gt = mag_gt * (1 - psi)
    
    # Compute low-frequency and high-frequency losses.
    L_LF = F.mse_loss(low_hat, low_gt)
    L_HF = F.mse_loss(high_hat, high_gt)
    
    # Compute weighting: c(t) = (t/t_max)^df_alpha and b(t) = 1 - t/t_max.
    t_norm = t.float() / t_max  # shape: [B]
    c_weight = (t_norm ** df_alpha).mean()
    b_weight = (1 - t_norm).mean()
    
    # Combine losses.
    L_DF = c_weight * L_LF + (1 - c_weight) * L_HF
    # Optionally, you might multiply L_DF by b_weight to adjust its overall contribution.
    return b_weight * L_DF
class VideoToVideo_sr(TorchModel):
    def __init__(self, opt=None, device=torch.device(f'cuda:0')):
        super().__init__()
        self.opt = opt
        self.device = device # torch.device(f'cuda:0')

        # text_encoder
        text_encoder = FrozenOpenCLIPEmbedder(device=self.device, pretrained="laion2b_s32b_b79k")
        text_encoder.model.to(self.device)
        self.text_encoder = text_encoder
        logger.info(f'Build encoder with FrozenOpenCLIPEmbedder')

        # U-Net with ControlNet
        generator = ControlledV2VUNet()
        generator = generator.to(self.device)
        # generator.eval()

        cfg.model_path = opt.model_path
        load_dict = torch.load(cfg.model_path, map_location='cpu')
        if 'state_dict' in load_dict:
            load_dict = load_dict['state_dict']
        ret = generator.load_state_dict(load_dict, strict=True)
        
        # self.generator = generator
        self.generator = generator
        logger.info('Load model path {}, with local status {}'.format(cfg.model_path, ret))

        # Noise scheduler
        # self.sigmas = noise_schedule(
        #     schedule='logsnr_cosine_interp',
        #     n=1000,
        #     zero_terminal_snr=True,
        #     scale_min=2.0,
        #     scale_max=4.0)

        diffusion = DiffusionDDIM(schedule='cosine', schedule_param={
            'num_timesteps': 1000,
            'cosine_s': 0.008,
            'zero_terminal_snr': True,
        },
        mean_type='v',
        loss_type='mse',
        var_type='fixed_small',
        rescale_timesteps=False,
        noise_strength=0.1)
        self.diffusion = diffusion
        logger.info('Build diffusion with DiffusionDDIM')

        # Temporal VAE

        vae = AutoencoderKLTemporalDecoder.from_pretrained(
            "stabilityai/stable-video-diffusion-img2vid", cache_dir="/group/ossdphi_algo_scratch_14/sichegao/checkpoints", subfolder="vae", variant="fp16"
        )
        vae.eval()
        vae.requires_grad_(False)
        vae.to(self.device)
        self.vae = vae
        logger.info('Build Temporal VAE')

        torch.cuda.empty_cache()

        self.negative_prompt = cfg.negative_prompt
        self.positive_prompt = cfg.positive_prompt


        exceptions = ["VideoControlNet", "local1", "local2"]
        self.freeze_parameters_except(self.generator, exceptions)
        negative_y = text_encoder(self.negative_prompt).detach()
        self.negative_y = negative_y




    def train_losses(self, x, y, text, model_kwargs=None, noise=None):
        B, T, C, H, W = x.shape
        x = x.view(B * T, C, H, W)
        x = F.interpolate(x, scale_factor=4, mode='bilinear')
        x = x.view(B, T, C, x.shape[2], x.shape[3])
        with torch.no_grad():
            x = self.vae_encode(x)
            y = self.vae_encode(y)
            # print("x_encoded:", x.mean().item(), x.std().item())
            # print("y_encoded:", y.mean().item(), y.std().item())
            text = self.text_encoder(text)
            # text_0 = text.clone()
            try:
                text[torch.rand(text.size(0)) < 0.0, :] =self.negative_y
            except:
                pass
        model_kwargs = {'y': text, 'hint': x}
        # if noise is None:
        #     noise = torch.randn_like(y)
        bs = y.shape[0]
        # # Sample a random timestep for each video
        t = torch.randint(
            0,
            self.diffusion.num_timesteps,
            (bs,),
            dtype=torch.long,
            device=y.device
        )
        loss = self.diffusion.loss(x0=y, t=t, model=self.generator, model_kwargs=model_kwargs)
        loss = loss.mean()
        return loss
    def test(self, input: Dict[str, Any], total_noise_levels=1000, \
                 steps=50, solver_mode='fast', guide_scale=7.5, max_chunk_len=32):
        video_data = input['video_data']
        y = input['y']
        (target_h, target_w) = input['target_res']

        video_data = F.interpolate(video_data, [target_h,target_w], mode='bilinear')

        logger.info(f'video_data shape: {video_data.shape}')
        frames_num, _, h, w = video_data.shape

        padding = pad_to_fit(h, w)
        video_data = F.pad(video_data, padding, 'constant', 1)

        video_data = video_data.unsqueeze(0)
        bs = 1
        video_data = video_data.to(self.device)

        video_data_feature = self.vae_encode(video_data)
        # torch.save(video_data_feature, "latents.pt")
        torch.cuda.empty_cache()

        y = self.text_encoder(y).detach()

        with amp.autocast(enabled=True):

            t = torch.LongTensor([total_noise_levels-1]).to(self.device)
            noised_lr = self.diffusion.diffuse(video_data_feature, t)

            model_kwargs = [{'y': y}, {'y': self.negative_y}]
            model_kwargs.append({'hint': video_data_feature})

            torch.cuda.empty_cache()
            chunk_inds = make_chunks(frames_num, interp_f_num=0, max_chunk_len=max_chunk_len) if frames_num > max_chunk_len else None

            solver = 'dpmpp_2m_sde' # 'heun' | 'dpmpp_2m_sde' 
            gen_vid = self.diffusion.sample_sr(
                noise=noised_lr,
                model=self.generator,
                model_kwargs=model_kwargs,
                guide_scale=guide_scale,
                guide_rescale=0.2,
                solver=solver,
                solver_mode=solver_mode,
                return_intermediate=None,
                steps=steps,
                t_max=total_noise_levels - 1,
                t_min=0,
                discretization='trailing',
                chunk_inds=chunk_inds,)
            torch.cuda.empty_cache()

            logger.info(f'sampling, finished.')
            vid_tensor_gen = self.vae_decode_chunk(gen_vid, chunk_size=3)

            logger.info(f'temporal vae decoding, finished.')

        w1, w2, h1, h2 = padding
        vid_tensor_gen = vid_tensor_gen[:,:,h1:h+h1,w1:w+w1]

        gen_video = rearrange(
            vid_tensor_gen, '(b f) c h w -> b c f h w', b=bs)

        torch.cuda.empty_cache()
        
        return gen_video.type(torch.float32).cpu()

    def temporal_vae_decode(self, z, num_f):
        return self.vae.decode(z/self.vae.config.scaling_factor, num_frames=num_f).sample

    def vae_decode_chunk(self, z, chunk_size=3):
        z = rearrange(z, "b c f h w -> (b f) c h w")
        video = []
        for ind in range(0, z.shape[0], chunk_size):
            num_f = z[ind:ind+chunk_size].shape[0]
            video.append(self.temporal_vae_decode(z[ind:ind+chunk_size],num_f))
        video = torch.cat(video)
        return video

    def vae_encode(self, t, chunk_size=1):
        num_f = t.shape[1]
        t = rearrange(t, "b f c h w -> (b f) c h w")
        z_list = []
        for ind in range(0,t.shape[0],chunk_size):
            z_list.append(self.vae.encode(t[ind:ind+chunk_size]).latent_dist.sample())
        z = torch.cat(z_list, dim=0)
        z = rearrange(z, "(b f) c h w -> b c f h w", f=num_f)
        return z * self.vae.config.scaling_factor
    
    def freeze_parameters_except(self, module: torch.nn.Module, exceptions: list):
        """
        Freeze all parameters in `module` except those whose name contains one of the substrings in `exceptions`.
        """
        for name, param in module.named_parameters():
            if any(exc in name for exc in exceptions):
                param.requires_grad = True
                # logger.info(f"Keeping parameter trainable: {name}")
            else:
                param.requires_grad = False
                # logger.info(f"Freezing parameter: {name}")
def pad_to_fit(h, w):
    BEST_H, BEST_W = 720, 1280

    if h < BEST_H:
        h1, h2 = _create_pad(h, BEST_H)
    elif h == BEST_H:
        h1 = h2 = 0
    else: 
        h1 = 0
        h2 = int((h + 48) // 64 * 64) + 64 - 48 - h

    if w < BEST_W:
        w1, w2 = _create_pad(w, BEST_W)
    elif w == BEST_W:
        w1 = w2 = 0
    else:
        w1 = 0
        w2 = int(w // 64 * 64) + 64 - w
    return (w1, w2, h1, h2)

def _create_pad(h, max_len):
    h1 = int((max_len - h) // 2)
    h2 = max_len - h1 - h
    return h1, h2


def make_chunks(f_num, interp_f_num, max_chunk_len, chunk_overlap_ratio=0.5):
    MAX_CHUNK_LEN = max_chunk_len
    MAX_O_LEN = MAX_CHUNK_LEN * chunk_overlap_ratio
    chunk_len = int((MAX_CHUNK_LEN-1)//(1+interp_f_num)*(interp_f_num+1)+1)
    o_len = int((MAX_O_LEN-1)//(1+interp_f_num)*(interp_f_num+1)+1)
    chunk_inds = sliding_windows_1d(f_num, chunk_len, o_len)
    return chunk_inds


def sliding_windows_1d(length, window_size, overlap_size):
    stride = window_size - overlap_size
    ind = 0
    coords = []
    while ind<length:
        if ind+window_size*1.25>=length:
            coords.append((ind,length))
            break
        else:
            coords.append((ind,ind+window_size))
            ind += stride  
    return coords
