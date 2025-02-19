import math
import time
from typing import Type, Dict, Any, Tuple, Callable

import numpy as np
from einops import rearrange
import torch
import torch.nn.functional as F
import einops
from . import merge
from .utils import isinstance_str, init_generator, join_frame, split_frame, func_warper, join_warper, split_warper

def compute_merge(module: torch.nn.Module, x: torch.Tensor, tome_info: Dict[str, Any]) -> Tuple[Callable, ...]:
    original_h, original_w = tome_info["size"]
    original_tokens = original_h * original_w
    downsample = int(math.ceil(math.sqrt(original_tokens // x.shape[1])))

    args = tome_info["args"]
    generator = module.generator

    # Frame Number and Token Number
    fsize = x.shape[0] // args["batch_size"]
    tsize = x.shape[1]

    # Merge tokens in high resolution layers
    if downsample <= args["max_downsample"]:

        if args["generator"] is None:
            args["generator"] = init_generator(x.device)
            # module.generator = module.generator.manual_seed(123)
        elif args["generator"].device != x.device:
            args["generator"] = init_generator(x.device, fallback=args["generator"])

        # Local Token Merging!

        local_tokens = join_frame(x, fsize)
        m_ls = [join_warper(fsize)]
        u_ls = [split_warper(fsize)]
        unm = 0
        curF = fsize

        # Recursive merge multi-frame tokens into one set. Such as 4->1 for 4 frames and 8->2->1 for 8 frames when target stride is 4.
        while curF > 1:
            m, u, ret_dict = merge.bipartite_soft_matching_randframe(
                local_tokens, curF, args["local_merge_ratio"], unm, generator, args["target_stride"], args["align_batch"])
            unm += ret_dict["unm_num"]
            m_ls.append(m)
            u_ls.append(u)
            local_tokens = m(local_tokens)

            # assert (x.shape[1] - unm) % tsize == 0
            # Total token number = current frame number * per-frame token number + unmerged token number
            curF = (local_tokens.shape[1] - unm) // tsize

        merged_tokens = local_tokens
        
        # Global Token Merging!
        if args["merge_global"]:
            if hasattr(module, "global_tokens") and module.global_tokens is not None:
                # Merge local tokens with global tokens. Randomly determine merging destination.
                if torch.rand(1, generator=generator, device=generator.device) > args["global_rand"]:
                    src_len = local_tokens.shape[1]
                    tokens = torch.cat(
                        [local_tokens, module.global_tokens.to(local_tokens)], dim=1)
                    local_chunk = 0
                else:
                    src_len = module.global_tokens.shape[1]
                    tokens = torch.cat(
                        [module.global_tokens.to(local_tokens), local_tokens], dim=1)
                    local_chunk = 1

                m, u, _ = merge.bipartite_soft_matching_2s(
                    tokens, src_len, args["global_merge_ratio"], args["align_batch"], unmerge_chunk=local_chunk)
                merged_tokens = m(tokens)
                m_ls.append(m)
                u_ls.append(u)

                # Update global tokens with unmerged local tokens. There should be a better way to do this.
                module.global_tokens = u(merged_tokens).detach().clone().cpu()
            else:
                module.global_tokens = local_tokens.detach().clone().cpu()

        m = func_warper(m_ls)
        u = func_warper(u_ls[::-1])
    else:
        m, u = (merge.do_nothing, merge.do_nothing)
        merged_tokens = x

    # Return merge op, unmerge op, and merged tokens.
    return m, u, merged_tokens

def compute_merge_local_temporal(module: torch.nn.Module, 
                                 x: torch.Tensor, 
                                 tome_info: Dict[str, Any],
                                 h: int,
                                 w: int) -> Tuple[Callable, Callable, torch.Tensor]:
    original_h, original_w = tome_info["size"]
    original_tokens = original_h * original_w
    downsample = int(math.ceil(math.sqrt(original_tokens // x.shape[1])))

    args = tome_info["args"]
    generator = module.generator

    # Frame Number and Token Number
    fsize = x.shape[0] // args["batch_size"]
    tsize = x.shape[1]

    # Merge tokens in high resolution layers
    if downsample <= args["max_downsample"]:

        if args["generator"] is None:
            args["generator"] = init_generator(x.device)
            # module.generator = module.generator.manual_seed(123)
        elif args["generator"].device != x.device:
            args["generator"] = init_generator(x.device, fallback=args["generator"])

        # Local Token Merging!

        local_tokens = join_frame(x, fsize)
        m_ls = [join_warper(fsize)]
        u_ls = [split_warper(fsize)]
        unm = 0
        curF = fsize

        # Recursive merge multi-frame tokens into one set. Such as 4->1 for 4 frames and 8->2->1 for 8 frames when target stride is 4.
        while curF > 1:
            m, u, ret_dict = merge.bipartite_soft_matching_randframe(
                local_tokens, curF, args["local_merge_ratio"], unm, generator, args["target_stride"], args["align_batch"])
            unm += ret_dict["unm_num"]
            m_ls.append(m)
            u_ls.append(u)
            local_tokens = m(local_tokens)

            # assert (x.shape[1] - unm) % tsize == 0
            # Total token number = current frame number * per-frame token number + unmerged token number
            curF = (local_tokens.shape[1] - unm) // tsize

        merged_tokens = local_tokens
        m = func_warper(m_ls)
        u = func_warper(u_ls[::-1])
    else:
        m, u = (merge.do_nothing, merge.do_nothing)
        merged_tokens = x

    # Return merge op, unmerge op, and merged tokens.
    return m, u, merged_tokens
def compute_merge_local_spatial(module: torch.nn.Module, 
                                x: torch.Tensor, 
                                tome_info: Dict[str, Any]) -> Tuple[Callable, Callable, torch.Tensor]:
    """
    Merges tokens along the spatial dimension.
    Assumes x represents spatial tokens (e.g. a flattened HxW grid per frame).
    """

    original_h, original_w = tome_info["size"]
    original_tokens = original_h * original_w
    downsample = int(math.ceil(math.sqrt(original_tokens // x.shape[1])))
    generator = module.generator
    args = tome_info["args"]
    if downsample <= args["max_downsample"]:
        w = int(math.ceil(original_w / downsample))
        h = int(math.ceil(original_h / downsample)) if downsample == 1 else int(math.ceil(original_h / downsample)) + 1
        r = int(x.shape[1] * args["local_merge_ratio"])
    # For spatial merging, you can use a 2D partition algorithm.
        use_rand = False
        # use_rand = False if x.shape[0] % 2 == 1 else args["use_rand"]
        m, u = merge.bipartite_soft_matching_random2d(
            x, w, h, r=r,# Since spatial merging is within one frame
            sx=2, sy=2, no_rand=not use_rand,
            generator=module.generator,
           
           )
    else:
        m, u = (merge.do_nothing, merge.do_nothing)
    merged_tokens = m(x)
    return m, u, merged_tokens
def compute_merge_global(module: torch.nn.Module, 
                         x: torch.Tensor, 
                         tome_info: Dict[str, Any]) -> Tuple[Callable, Callable, torch.Tensor]:
    """
    Merges tokens globally across the entire set.
    Assumes a fixed split: for example, the first src_len tokens are used as source.
    """

    args = tome_info["args"]
    generator = module.generator
    # Define a split point (could be based on a fixed ratio or a parameter)
    src_len = args.get("global_src_len", x.shape[1] // 2)
    
    m, u, ret_dict = merge.bipartite_soft_matching_2s(
        x, src_len, args["global_merge_ratio"], args["align_batch"], 
        merge_mode="replace", unmerge_chunk=0
    )
    merged_tokens = m(x)
    return m, u, merged_tokens

import torch
from torch import nn
from einops import rearrange

def make_basictransformerblock_tome_block(block_class: type) -> type:
    """
    Returns a patched version of your BasicTransformerBlock that integrates token merging (ToMe).
    
    This new block overrides the forward method to:
      - Normalize the input tokens.
      - Merge tokens via `compute_merge` (which should return a tuple (m_a, u_a, merged_tokens)).
      - Apply the first attention layer on the merged tokens.
      - Unmerge the tokens before adding the residual connection.
    
    It handles both the 'space' and 'temp' (temporal) local attention branches, as well as a default case.
    Assumes that:
      - `compute_merge(self, tokens, self._tome_info)` is defined elsewhere.
      - An attribute `_tome_info` is present on the block (e.g. registered via an external patch).
    """
    class ToMeBasicTransformerBlock(block_class):
        # Save a reference to the original class for unpatching if needed.
        _parent = block_class

        def forward(self, x, context=None, h=None, w=None):
            # If no token merging info is set, fallback to the original forward.
            n, f, c = x.shape
            if not hasattr(self, "_tome_info") or self._tome_info is None:
                return super().forward(x, context, h, w)

            # --- Branch for spatial local attention ---
            if self.local_type == 'space' and self.is_ctrl:
                # Rearrange input for spatial processing.
                x_local = rearrange(x, 'b (h w) c -> b c h w', h=h)
                x_local = self.local1(x_local)
                x_local = rearrange(x_local, 'b c h w -> b (h w) c')

                # Normalize and perform token merging.
                norm_tokens = self.norm1(x_local)
                m_a, u_a, merged_tokens = compute_merge_local_spatial(self, norm_tokens, self._tome_info)
                # Use merged tokens as input to the first attention layer.
                attn_out = self.attn1(merged_tokens, context=context if self.disable_self_attn else None)
                # Unmerge the output tokens.
                attn_out = u_a(attn_out)
                # if u_a.__name__ != 'do_nothing':
                    # attn_out = rearrange(attn_out, "(b h w) f c -> f (b h w) c", h=h, w=w)
                # Residual connection.
                x = attn_out + x
                norm_tokens = self.norm2(x)
                merged_tokens = m_a(norm_tokens)
                # m_a, u_a, merged_tokens = compute_merge_local_spatial(self, norm_tokens, self._tome_info)
                # Use merged tokens as input to the first attention layer.
                attn_out = self.attn2(merged_tokens, context=context)
                # Unmerge the output tokens.
                attn_out = u_a(attn_out)
                x = attn_out + x
                norm_tokens = self.norm3(x)
                merged_tokens = m_a(norm_tokens)

                # m_a, u_a, merged_tokens = compute_merge_local_spatial(self, norm_tokens, self._tome_info)
                # Use merged tokens as input to the first attention layer.
                attn_out = self.ff(merged_tokens)
                # Unmerge the output tokens.
                attn_out = u_a(attn_out)
                x = attn_out + x
                # Continue with the remaining attention and feed-forward layers.
                # x = self.attn2(self.norm2(x), context=context) + x
                # x = self.ff(self.norm3(x)) + x
                return x

            # --- Branch for temporal local attention ---
            elif self.local_type == 'temp' and self.is_ctrl:
                # Process with the first local (temporal) attention module.
                x_local = self.local1(x)
                norm_tokens = self.norm1(x_local)
                norm_tokens = rearrange(norm_tokens, "(b h w) f c -> (b f) (h w) c", h=h, w=w)
                m_a, u_a, merged_tokens = compute_merge_local_spatial(self, norm_tokens, self._tome_info)
                # m_a, u_a, merged_tokens = compute_merge_local_temporal(self, norm_tokens, self._tome_info, h, w)
                # norm_tokens = merged_tokens
                # if u_a.__name__ != 'do_nothing':
                #     norm_tokens = rearrange(norm_tokens, "b (f n) c -> (b n) f c", b=self._tome_info['args']['batch_size'], f=x.shape[1])
                #     attn_out = self.attn1(norm_tokens, context=context if self.disable_self_attn else None)
                #     attn_out = rearrange(attn_out, "(b n) f c -> b (f n) c", b=self._tome_info['args']['batch_size'], f=x.shape[1])
                # else:
                attn_out = self.attn1(merged_tokens, context=context if self.disable_self_attn else None)
                attn_out = u_a(attn_out)
                # attn_out = u_a(attn_out)
                # if u_a.__name__ == 'do_nothing':
                attn_out = rearrange(attn_out, "f (b h w) c -> (b h w) f c", h=h, w=w)
                x = attn_out + x

                # Process with the second local module and cross-attention.
                # x_local = self.local2(x)
                norm_tokens = self.local2(x)
                norm_tokens = self.norm3(norm_tokens)

                merged_tokens = m_a(norm_tokens)

                # norm_tokens = rearrange(merged_tokens, "(b h w) f c -> (b f) (h w) c", h=h//2, w=w//2)
                # m_a, u_a, merged_tokens = compute_merge_local_spatial(self, norm_tokens, self._tome_info)
                attn_out = self.attn2(merged_tokens, context=context)
                attn_out = u_a(attn_out)
                if u_a.__name__ != 'do_nothing':
                    attn_out = rearrange(attn_out, "f (b h w) c -> (b h w) f c", h=h, w=w)

                x = attn_out + x
                norm_tokens = self.norm3(x)
                # norm_tokens = rearrange(norm_tokens, "(b h w) f c -> (b f) (h w) c", h=h//2, w=w//2)
                # m_a, u_a, merged_tokens = compute_merge_local_spatial(self, norm_tokens, self._tome_info)
                merged_tokens = m_a(norm_tokens)

                attn_out = self.ff(merged_tokens)
                attn_out = u_a(attn_out)
                if u_a.__name__ != 'do_nothing':
                    attn_out = rearrange(attn_out, "f (b h w) c -> (b h w) f c", h=h, w=w)
                x = attn_out + x
                # x = self.ff(self.norm3(x)) + x
                return x

            # --- Default branch (if no special local attention is set) ---
            else:
                norm_tokens = self.norm1(x)
                m_a, u_a, merged_tokens = compute_merge(self, norm_tokens, self._tome_info)
                attn_out = self.attn1(merged_tokens, context=context if self.disable_self_attn else None)
                attn_out = u_a(attn_out)
                x = attn_out + x
                x = self.attn2(self.norm2(x), context=context) + x
                x = self.ff(self.norm3(x)) + x
                return x

    return ToMeBasicTransformerBlock



def hook_tome_model(model: torch.nn.Module):
    """ Adds a forward pre hook to get the image size. This hook can be removed with remove_patch. """
    def hook(module, args):
        module._tome_info["size"] = (args[0].shape[3], args[0].shape[4])
        return None

    model._tome_info["hooks"].append(model.register_forward_pre_hook(hook))


def hook_tome_module(module: torch.nn.Module):
    """ Adds a forward pre hook to initialize random number generator.
        All modules share the same generator state to keep their randomness in VidToMe consistent in one pass.
        This hook can be removed with remove_patch. """
    def hook(module, args):
        if not hasattr(module, "generator"):
            module.generator = init_generator(args[0].device)
        elif module.generator.device != args[0].device:
            module.generator = init_generator(
                args[0].device, fallback=module.generator)
        else:
            return None

        # module.generator = module.generator.manual_seed(module._tome_info["args"]["seed"])
        return None

    module._tome_info["hooks"].append(module.register_forward_pre_hook(hook))


def apply_patch(
        model: torch.nn.Module,
        local_merge_ratio: float = 0.9,
        merge_global: bool = False,
        global_merge_ratio=0.8,
        max_downsample: int = 2,
        seed: int = 123,
        batch_size: int = 2,
        include_control: bool = False,
        align_batch: bool = False,
        target_stride: int = 4,
        global_rand=0.5):
    """
    Patches a stable diffusion model with VidToMe.
    Apply this to the highest level stable diffusion object (i.e., it should have a .model.diffusion_model).

    Important Args:
     - model: A top level Stable Diffusion module to patch in place. Should have a ".model.diffusion_model"
     - local_merge_ratio: The ratio of tokens to merge locally. I.e., 0.9 would merge 90% src tokens.
              If there are 4 frames in a chunk (3 src, 1 dst), the compression ratio will be 1.3 / 4.0.
              And the largest compression ratio is 0.25 (when local_merge_ratio = 1.0).
              Higher values result in more consistency, but with more visual quality loss.
     - merge_global: Whether or not to include global token merging.
     - global_merge_ratio: The ratio of tokens to merge locally. I.e., 0.8 would merge 80% src tokens.
                           When find significant degradation in video quality. Try to lower the value.

    Args to tinker with if you want:
     - max_downsample [1, 2, 4, or 8]: Apply VidToMe to layers with at most this amount of downsampling.
                                       E.g., 1 only applies to layers with no downsampling (4/15) while
                                       8 applies to all layers (15/15). I recommend a value of 1 or 2.
     - seed: Manual random seed. 
     - batch_size: Video batch size. Number of video chunks in one pass. When processing one video, it 
                   should be 2 (cond + uncond) or 3 (when using PnP, source + cond + uncond).
     - include_control: Whether or not to patch ControlNet model.
     - align_batch: Whether or not to align similarity matching maps of samples in the batch. It should
                    be True when using PnP as control.
     - target_stride: Stride between target frames. I.e., when target_stride = 4, there is 1 target frame
                      in any 4 consecutive frames. 
     - global_rand: Probability in global token merging src/dst split. Global tokens are always src when
                    global_rand = 1.0 and always dst when global_rand = 0.0 .
    """

    # Make sure the module is not currently patched
    remove_patch(model)

    # is_diffusers = isinstance_str(
    #     model, "DiffusionPipeline") or isinstance_str(model, "ModelMixin")

    # if not is_diffusers:
    #     if not hasattr(model, "model") or not hasattr(model.model, "diffusion_model"):
    #         # Provided model not supported
    #         raise RuntimeError(
    #             "Provided model was not a Stable Diffusion / Latent Diffusion model, as expected.")
    #     diffusion_model = model.model.diffusion_model
    # else:
    #     # Supports "pipe.unet" and "unet"
    diffusion_model = model.generator if hasattr(model, "generator") else model

    # if isinstance_str(model, "StableDiffusionControlNetPipeline") and include_control:
    #     diffusion_models = [diffusion_model, model.controlnet]
    # else:
    #     diffusion_models = [diffusion_model]

    # for diffusion_model in diffusion_models:
    diffusion_model._tome_info = {
        "size": None,
        "hooks": [],
        "args": {
            "max_downsample": max_downsample,
            "generator": None,
            "seed": seed,
            "batch_size": batch_size,
            "align_batch": align_batch,
            "merge_global": merge_global,
            "global_merge_ratio": global_merge_ratio,
            "local_merge_ratio": local_merge_ratio,
            "global_rand": global_rand,
            "target_stride": target_stride
        }
    }
    hook_tome_model(diffusion_model)

    for name, module in diffusion_model.named_modules():
        # If for some reason this has a different name, create an issue and I'll fix it
        # if isinstance_str(module, "BasicTransformerBlock") and "down_blocks" not in name:
        if isinstance_str(module, "BasicTransformerBlock") and module.local_type is not 'temp':
        # if isinstance_str(module, "BasicTransformerBlock"):
            make_tome_block_fn = make_basictransformerblock_tome_block
            module.__class__ = make_tome_block_fn(module.__class__)
            module._tome_info = diffusion_model._tome_info
            hook_tome_module(module)
            # module.set_metrics_log()
            # hook_tome_mse(module)
            # Something introduced in SD 2.0 (LDM only)
            if not hasattr(module, "disable_self_attn"):
                module.disable_self_attn = False

            # Something needed for older versions of diffusers
            if not hasattr(module, "use_ada_layer_norm_zero"):
                module.use_ada_layer_norm = False
                module.use_ada_layer_norm_zero = False
        # register_tome_metric_hooks(diffusion_model)

    return model


def remove_patch(model: torch.nn.Module):
    """ Removes a patch from a ToMe Diffusion module if it was already patched. """
    # For diffusers

    model = model.generator if hasattr(model, "generator") else model
    model_ls = [model]

    for model in model_ls:
        for _, module in model.named_modules():
            if hasattr(module, "_tome_info"):
                for hook in module._tome_info["hooks"]:
                    hook.remove()
                module._tome_info["hooks"].clear()

            if module.__class__.__name__ == "ToMeBlock":
                module.__class__ = module._parent

    return model


def update_patch(model: torch.nn.Module, **kwargs):
    """ Update arguments in patched modules """
    # For diffusers
    model0 = model.unet if hasattr(model, "unet") else model
    model_ls = [model0]
    if hasattr(model, "controlnet"):
        model_ls.append(model.controlnet)
    for model in model_ls:
        for _, module in model.named_modules():
            if hasattr(module, "_tome_info"):
                for k, v in kwargs.items():
                    setattr(module, k, v)
    return model


def collect_from_patch(model: torch.nn.Module, attr="tome"):
    """ Collect attributes in patched modules """
    # For diffusers
    model0 = model.unet if hasattr(model, "unet") else model
    model_ls = [model0]
    if hasattr(model, "controlnet"):
        model_ls.append(model.controlnet)
    ret_dict = dict()
    for model in model_ls:
        for name, module in model.named_modules():
            if hasattr(module, attr):
                res = getattr(module, attr)
                ret_dict[name] = res

    return ret_dict
