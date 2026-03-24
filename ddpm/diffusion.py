import math
import copy
import torch
from torch import nn, einsum
import torch.nn.functional as F
from functools import partial
from pathlib import Path
from torch.optim import Adam
from torch.cuda.amp import autocast, GradScaler
from einops import rearrange
from einops_exts import rearrange_many
from rotary_embedding_torch import RotaryEmbedding
from ddpm.guidance import LNCCLoss3D, EdgeLoss3D, build_guidance_scale, clamp_joint_x0, cfg_get
from ddpm.text import tokenize, bert_embed, BERT_MODEL_DIM
from torch.utils.data import DataLoader

def exists(x):
    return x is not None
def noop(*args, **kwargs):
    pass
def is_odd(n):
    return (n % 2) == 1
def default(val, d):
    if exists(val):
        return val
    return d() if callable(d) else d
def cycle(dl):
    while True:
        for data in dl:
            yield data
def num_to_groups(num, divisor):
    groups = num // divisor
    remainder = num % divisor
    arr = [divisor] * groups
    if remainder > 0:
        arr.append(remainder)
    return arr
def prob_mask_like(shape, prob, device):
    if prob == 1:
        return torch.ones(shape, device=device, dtype=torch.bool)
    elif prob == 0:
        return torch.zeros(shape, device=device, dtype=torch.bool)
    else:
        return torch.zeros(shape, device=device).float().uniform_(0, 1) < prob
def is_list_str(x):
    if not isinstance(x, (list, tuple)):
        return False
    return all([type(el) == str for el in x])

class RelativePositionBias(nn.Module):
    def __init__(
        self,
        heads=8,
        num_buckets=32,
        max_distance=128
    ):
        super().__init__()
        self.num_buckets = num_buckets
        self.max_distance = max_distance
        self.relative_attention_bias = nn.Embedding(num_buckets, heads)

    @staticmethod
    def _relative_position_bucket(relative_position, num_buckets=32, max_distance=128):
        ret = 0
        n = -relative_position
        num_buckets //= 2
        ret += (n < 0).long() * num_buckets
        n = torch.abs(n)
        max_exact = num_buckets // 2
        is_small = n < max_exact
        val_if_large = max_exact + (
            torch.log(n.float() / max_exact) / math.log(max_distance /
                                                        max_exact) * (num_buckets - max_exact)
        ).long()
        val_if_large = torch.min(
            val_if_large, torch.full_like(val_if_large, num_buckets - 1))
        ret += torch.where(is_small, n, val_if_large)
        return ret

    def forward(self, n, device):
        q_pos = torch.arange(n, dtype=torch.long, device=device)
        k_pos = torch.arange(n, dtype=torch.long, device=device)
        rel_pos = rearrange(k_pos, 'j -> 1 j') - rearrange(q_pos, 'i -> i 1')
        rp_bucket = self._relative_position_bucket(
            rel_pos, num_buckets=self.num_buckets, max_distance=self.max_distance)
        values = self.relative_attention_bias(rp_bucket)
        return rearrange(values, 'i j h -> h i j')

class EMA():
    def __init__(self, beta):
        super().__init__()
        self.beta = beta

    def update_model_average(self, ma_model, current_model):
        for current_params, ma_params in zip(current_model.parameters(), ma_model.parameters()):
            old_weight, up_weight = ma_params.data, current_params.data
            ma_params.data = self.update_average(old_weight, up_weight)

    def update_average(self, old, new):
        if old is None:
            return new
        return old * self.beta + (1 - self.beta) * new


class Residual(nn.Module):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn

    def forward(self, x, *args, **kwargs):
        return self.fn(x, *args, **kwargs) + x


class SinusoidalPosEmb(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, x):
        device = x.device
        half_dim = self.dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=device) * -emb)
        emb = x[:, None] * emb[None, :]
        emb = torch.cat((emb.sin(), emb.cos()), dim=-1)
        return emb


def Upsample(dim):
    return nn.ConvTranspose3d(dim, dim, (1, 4, 4), (1, 2, 2), (0, 1, 1))


def Downsample(dim):
    return nn.Conv3d(dim, dim, (1, 4, 4), (1, 2, 2), (0, 1, 1))


class LayerNorm(nn.Module):
    def __init__(self, dim, eps=1e-5):
        super().__init__()
        self.eps = eps
        self.gamma = nn.Parameter(torch.ones(1, dim, 1, 1, 1))

    def forward(self, x):
        var = torch.var(x, dim=1, unbiased=False, keepdim=True)
        mean = torch.mean(x, dim=1, keepdim=True)
        return (x - mean) / (var + self.eps).sqrt() * self.gamma


class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.fn = fn
        self.norm = LayerNorm(dim)

    def forward(self, x, **kwargs):
        x = self.norm(x)
        return self.fn(x, **kwargs)



class Block(nn.Module):
    def __init__(self, dim, dim_out, groups=8):
        super().__init__()
        self.proj = nn.Conv3d(dim, dim_out, (1, 3, 3), padding=(0, 1, 1))
        self.norm = nn.GroupNorm(groups, dim_out)
        self.act = nn.SiLU()

    def forward(self, x, scale_shift=None):
        x = self.proj(x)
        x = self.norm(x)
        if exists(scale_shift):
            scale, shift = scale_shift
            x = x * (scale + 1) + shift
        return self.act(x)


class ResnetBlock(nn.Module):
    def __init__(self, dim, dim_out, *, time_emb_dim=None, groups=8):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.SiLU(),
            nn.Linear(time_emb_dim, dim_out * 2)
        ) if exists(time_emb_dim) else None
        self.block1 = Block(dim, dim_out, groups=groups)
        self.block2 = Block(dim_out, dim_out, groups=groups)
        self.res_conv = nn.Conv3d(
            dim, dim_out, 1) if dim != dim_out else nn.Identity()

    def forward(self, x, time_emb=None):
        scale_shift = None
        if exists(self.mlp):
            assert exists(time_emb), 'time emb must be passed in'
            time_emb = self.mlp(time_emb)
            time_emb = rearrange(time_emb, 'b c -> b c 1 1 1')
            scale_shift = time_emb.chunk(2, dim=1)
        h = self.block1(x, scale_shift=scale_shift)
        h = self.block2(h)
        return h + self.res_conv(x)


class SpatialLinearAttention(nn.Module):
    def __init__(self, dim, heads=4, dim_head=32):
        super().__init__()
        self.scale = dim_head ** -0.5
        self.heads = heads
        hidden_dim = dim_head * heads
        self.to_qkv = nn.Conv2d(dim, hidden_dim * 3, 1, bias=False)
        self.to_out = nn.Conv2d(hidden_dim, dim, 1)

    def forward(self, x):
        b, c, f, h, w = x.shape
        x = rearrange(x, 'b c f h w -> (b f) c h w')
        qkv = self.to_qkv(x).chunk(3, dim=1)
        q, k, v = rearrange_many(
            qkv, 'b (h c) x y -> b h c (x y)', h=self.heads)
        q = q.softmax(dim=-2)
        k = k.softmax(dim=-1)
        q = q * self.scale
        context = torch.einsum('b h d n, b h e n -> b h d e', k, v)

        out = torch.einsum('b h d e, b h d n -> b h e n', context, q)
        out = rearrange(out, 'b h c (x y) -> b (h c) x y',
                        h=self.heads, x=h, y=w)
        out = self.to_out(out)
        return rearrange(out, '(b f) c h w -> b c f h w', b=b)


class EinopsToAndFrom(nn.Module):
    def __init__(self, from_einops, to_einops, fn):
        super().__init__()
        self.from_einops = from_einops
        self.to_einops = to_einops
        self.fn = fn

    def forward(self, x, **kwargs):
        shape = x.shape
        reconstitute_kwargs = dict(
            tuple(zip(self.from_einops.split(' '), shape)))
        x = rearrange(x, f'{self.from_einops} -> {self.to_einops}')
        x = self.fn(x, **kwargs)
        x = rearrange(
            x, f'{self.to_einops} -> {self.from_einops}', **reconstitute_kwargs)
        return x


class Attention(nn.Module):
    def __init__(
        self,
        dim,
        heads=4,
        dim_head=32,
        rotary_emb=None
    ):
        super().__init__()
        self.scale = dim_head ** -0.5
        self.heads = heads
        hidden_dim = dim_head * heads
        self.rotary_emb = rotary_emb
        self.to_qkv = nn.Linear(dim, hidden_dim * 3, bias=False)
        self.to_out = nn.Linear(hidden_dim, dim, bias=False)

    def forward(
        self,
        x,
        pos_bias=None,
        focus_present_mask=None
    ):
        n, device = x.shape[-2], x.device
        qkv = self.to_qkv(x).chunk(3, dim=-1)
        if exists(focus_present_mask) and focus_present_mask.all():
            values = qkv[-1]
            return self.to_out(values)
        q, k, v = rearrange_many(qkv, '... n (h d) -> ... h n d', h=self.heads)
        q = q * self.scale
        if exists(self.rotary_emb):
            q = self.rotary_emb.rotate_queries_or_keys(q)
            k = self.rotary_emb.rotate_queries_or_keys(k)
        sim = einsum('... h i d, ... h j d -> ... h i j', q, k)
        if exists(pos_bias):
            sim = sim + pos_bias

        if exists(focus_present_mask) and not (~focus_present_mask).all():
            attend_all_mask = torch.ones(
                (n, n), device=device, dtype=torch.bool)
            attend_self_mask = torch.eye(n, device=device, dtype=torch.bool)

            mask = torch.where(
                rearrange(focus_present_mask, 'b -> b 1 1 1 1'),
                rearrange(attend_self_mask, 'i j -> 1 1 1 i j'),
                rearrange(attend_all_mask, 'i j -> 1 1 1 i j'),
            )
            sim = sim.masked_fill(~mask, -torch.finfo(sim.dtype).max)
        sim = sim - sim.amax(dim=-1, keepdim=True).detach()
        attn = sim.softmax(dim=-1)
        out = einsum('... h i j, ... h j d -> ... h i d', attn, v)
        out = rearrange(out, '... h n d -> ... n (h d)')
        return self.to_out(out)


class Unet3D(nn.Module):
    def __init__(
        self,
        dim,
        cond_dim=None,
        out_dim=None,
        dim_mults=(1, 2, 4, 8),
        channels=3,
        attn_heads=8,
        attn_dim_head=32,
        use_bert_text_cond=False,
        init_dim=None,
        init_kernel_size=7,
        use_sparse_linear_attn=True,
        resnet_groups=8
    ):
        super().__init__()
        self.channels = channels
        rotary_emb = RotaryEmbedding(min(32, attn_dim_head))
        def temporal_attn(dim): return EinopsToAndFrom('b c f h w', 'b (h w) f c', Attention(
            dim, heads=attn_heads, dim_head=attn_dim_head, rotary_emb=rotary_emb))
        self.time_rel_pos_bias = RelativePositionBias(
            heads=attn_heads, max_distance=32)
        init_dim = default(init_dim, dim)
        assert is_odd(init_kernel_size)
        init_padding = init_kernel_size // 2
        self.init_conv = nn.Conv3d(channels, init_dim, (1, init_kernel_size,
                                   init_kernel_size), padding=(0, init_padding, init_padding))
        self.init_temporal_attn = Residual(
            PreNorm(init_dim, temporal_attn(init_dim)))
        dims = [init_dim, *map(lambda m: dim * m, dim_mults)]
        in_out = list(zip(dims[:-1], dims[1:]))
        time_dim = dim * 4
        self.time_mlp = nn.Sequential(
            SinusoidalPosEmb(dim),
            nn.Linear(dim, time_dim),
            nn.GELU(),
            nn.Linear(time_dim, time_dim)
        )
        self.has_cond = exists(cond_dim) or use_bert_text_cond
        cond_dim = BERT_MODEL_DIM if use_bert_text_cond else cond_dim
        self.null_cond_emb = nn.Parameter(
            torch.randn(1, cond_dim)) if self.has_cond else None
        cond_dim = time_dim + int(cond_dim or 0)
        self.downs = nn.ModuleList([])
        self.ups = nn.ModuleList([])
        num_resolutions = len(in_out)
        block_klass = partial(ResnetBlock, groups=resnet_groups)
        block_klass_cond = partial(block_klass, time_emb_dim=cond_dim)
        for ind, (dim_in, dim_out) in enumerate(in_out):
            is_last = ind >= (num_resolutions - 1)
            self.downs.append(nn.ModuleList([
                block_klass_cond(dim_in, dim_out),
                block_klass_cond(dim_out, dim_out),
                Residual(PreNorm(dim_out, SpatialLinearAttention(
                    dim_out, heads=attn_heads))) if use_sparse_linear_attn else nn.Identity(),
                Residual(PreNorm(dim_out, temporal_attn(dim_out))),
                Downsample(dim_out) if not is_last else nn.Identity()
            ]))
        mid_dim = dims[-1]
        self.mid_block1 = block_klass_cond(mid_dim, mid_dim)
        spatial_attn = EinopsToAndFrom(
            'b c f h w', 'b f (h w) c', Attention(mid_dim, heads=attn_heads))
        self.mid_spatial_attn = Residual(PreNorm(mid_dim, spatial_attn))
        self.mid_temporal_attn = Residual(
            PreNorm(mid_dim, temporal_attn(mid_dim)))
        self.mid_block2 = block_klass_cond(mid_dim, mid_dim)
        for ind, (dim_in, dim_out) in enumerate(reversed(in_out)):
            is_last = ind >= (num_resolutions - 1)
            self.ups.append(nn.ModuleList([
                block_klass_cond(dim_out * 2, dim_in),
                block_klass_cond(dim_in, dim_in),
                Residual(PreNorm(dim_in, SpatialLinearAttention(
                    dim_in, heads=attn_heads))) if use_sparse_linear_attn else nn.Identity(),
                Residual(PreNorm(dim_in, temporal_attn(dim_in))),
                Upsample(dim_in) if not is_last else nn.Identity()
            ]))
        out_dim = default(out_dim, channels)
        self.final_conv = nn.Sequential(
            block_klass(dim * 2, dim),
            nn.Conv3d(dim, out_dim, 1)
        )

    def forward_with_cond_scale(
        self,
        *args,
        cond_scale=2.,
        **kwargs
    ): 
        logits = self.forward(*args, null_cond_prob=0., **kwargs)
        if cond_scale == 1 or not self.has_cond:
            return logits
        null_logits = self.forward(*args, null_cond_prob=1., **kwargs)
        return null_logits + (logits - null_logits) * cond_scale

    def forward(
        self,
        x,
        time,
        cond=None,
        null_cond_prob=0.,
        focus_present_mask=None,
        prob_focus_present=0.
    ):
        if cond is None:                    
            cond = torch.zeros((1, 16))
            cond[0, -1] = 1.0
        assert not (self.has_cond and not exists(cond)
                    ), 'cond must be passed in if cond_dim specified'
        batch, device = x.shape[0], x.device
        focus_present_mask = default(focus_present_mask, lambda: prob_mask_like(
            (batch,), prob_focus_present, device=device))
        time_rel_pos_bias = self.time_rel_pos_bias(x.shape[2], device=x.device)
        x = self.init_conv(x)
        r = x.clone()
        x = self.init_temporal_attn(x, pos_bias=time_rel_pos_bias)
        t = self.time_mlp(time) if exists(self.time_mlp) else None
        if self.has_cond:
            batch, device = x.shape[0], x.device
            mask = prob_mask_like((batch,), null_cond_prob, device=device)
            cond = cond.to(device)
            cond = torch.where(rearrange(mask, 'b -> b 1'),
                               self.null_cond_emb, cond)

            t = torch.cat((t, cond), dim=-1)
        h = []
        for block1, block2, spatial_attn, temporal_attn, downsample in self.downs:
            x = block1(x, t)
            x = block2(x, t)
            x = spatial_attn(x)
            x = temporal_attn(x, pos_bias=time_rel_pos_bias,
                              focus_present_mask=focus_present_mask)
            h.append(x)
            x = downsample(x)
        x = self.mid_block1(x, t)
        x = self.mid_spatial_attn(x)
        x = self.mid_temporal_attn(
            x, pos_bias=time_rel_pos_bias, focus_present_mask=focus_present_mask)
        x = self.mid_block2(x, t)
        for block1, block2, spatial_attn, temporal_attn, upsample in self.ups:
            x = torch.cat((x, h.pop()), dim=1)
            x = block1(x, t)
            x = block2(x, t)
            x = spatial_attn(x)
            x = temporal_attn(x, pos_bias=time_rel_pos_bias,
                              focus_present_mask=focus_present_mask)
            x = upsample(x)

        x = torch.cat((x, r), dim=1)
        return self.final_conv(x)



def extract(a, t, x_shape):
    b, *_ = t.shape
    out = a.gather(-1, t)
    return out.reshape(b, *((1,) * (len(x_shape) - 1)))


def cosine_beta_schedule(timesteps, s=0.008):

    steps = timesteps + 1
    x = torch.linspace(0, timesteps, steps, dtype=torch.float64)
    alphas_cumprod = torch.cos(
        ((x / timesteps) + s) / (1 + s) * torch.pi * 0.5) ** 2
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
    return torch.clip(betas, 0, 0.9999)


def make_ddim_timesteps(num_ddim_steps, num_ddpm_steps):
    if num_ddim_steps >= num_ddpm_steps:
        return torch.arange(num_ddpm_steps - 1, -1, -1, dtype=torch.long)
    step_ratio = num_ddpm_steps / num_ddim_steps
    steps = torch.arange(num_ddim_steps, dtype=torch.float32) * step_ratio
    steps = torch.round(steps).long().clamp(max=num_ddpm_steps - 1)
    steps = steps.unique(sorted=True)
    return torch.flip(steps, dims=[0])


class GaussianDiffusion_Nolatent(nn.Module):
    def __init__(
        self,
        denoise_fn,
        *,
        image_size,
        num_frames,
        text_use_bert_cls=False,
        channels=2,
        timesteps=1000,
        loss_type='l1',
        use_dynamic_thres=False, 
        dynamic_thres_percentile=0.9,
        device=None,
        use_guide=True,
    ):
        super().__init__()
        self.channels = channels
        self.image_size = image_size
        self.num_frames = num_frames
        self.denoise_fn = denoise_fn
        self.device=device
        betas = cosine_beta_schedule(timesteps)
        alphas = 1. - betas
        alphas_cumprod = torch.cumprod(alphas, axis=0)
        alphas_cumprod_prev = F.pad(alphas_cumprod[:-1], (1, 0), value=1.)
        timesteps, = betas.shape
        self.num_timesteps = int(timesteps)
        print("timesteps : ", timesteps)
        self.loss_type = loss_type
        if use_guide is None:
            use_guide = False
        elif not isinstance(use_guide, bool):
            raise TypeError('use_guide must be a boolean value')
        self.use_guide = use_guide


        def register_buffer(name, val): return self.register_buffer(
            name, val.to(torch.float32))
        register_buffer('betas', betas)
        register_buffer('alphas_cumprod', alphas_cumprod)
        register_buffer('alphas_cumprod_prev', alphas_cumprod_prev)
        register_buffer('sqrt_alphas_cumprod', torch.sqrt(alphas_cumprod))
        register_buffer('sqrt_one_minus_alphas_cumprod',
                        torch.sqrt(1. - alphas_cumprod))
        register_buffer('log_one_minus_alphas_cumprod',
                        torch.log(1. - alphas_cumprod))
        register_buffer('sqrt_recip_alphas_cumprod',
                        torch.sqrt(1. / alphas_cumprod))
        register_buffer('sqrt_recipm1_alphas_cumprod',
                        torch.sqrt(1. / alphas_cumprod - 1))


        posterior_variance = betas * \
            (1. - alphas_cumprod_prev) / (1. - alphas_cumprod)
        register_buffer('posterior_variance', posterior_variance)

        register_buffer('posterior_log_variance_clipped',
                        torch.log(posterior_variance.clamp(min=1e-20)))
        register_buffer('posterior_mean_coef1', betas *
                        torch.sqrt(alphas_cumprod_prev) / (1. - alphas_cumprod))
        register_buffer('posterior_mean_coef2', (1. - alphas_cumprod_prev)
                        * torch.sqrt(alphas) / (1. - alphas_cumprod))


        self.text_use_bert_cls = text_use_bert_cls


        self.use_dynamic_thres = use_dynamic_thres
        self.dynamic_thres_percentile = dynamic_thres_percentile

    def q_mean_variance(self, x_start, t):
        mean = extract(self.sqrt_alphas_cumprod, t, x_start.shape) * x_start
        variance = extract(1. - self.alphas_cumprod, t, x_start.shape)
        log_variance = extract(
            self.log_one_minus_alphas_cumprod, t, x_start.shape)
        return mean, variance, log_variance
    
    def predict_start_from_noise(self, x_t, t, noise):
        return (
            extract(self.sqrt_recip_alphas_cumprod, t, x_t.shape) * x_t -
            extract(self.sqrt_recipm1_alphas_cumprod, t, x_t.shape) * noise
        )

    def q_posterior(self, x_start, x_t, t):
        posterior_mean = (
            extract(self.posterior_mean_coef1, t, x_t.shape) * x_start +
            extract(self.posterior_mean_coef2, t, x_t.shape) * x_t
        )
        posterior_variance = extract(self.posterior_variance, t, x_t.shape)
        posterior_log_variance_clipped = extract(
            self.posterior_log_variance_clipped, t, x_t.shape)
        return posterior_mean, posterior_variance, posterior_log_variance_clipped

    def _predict_noise(self, x, t, cond=None, cond_scale=1.):
        if isinstance(self.denoise_fn, torch.nn.DataParallel):
            return self.denoise_fn.module.forward_with_cond_scale(x, t, cond=cond, cond_scale=cond_scale)
        return self.denoise_fn.forward_with_cond_scale(x, t, cond=cond, cond_scale=cond_scale)

    def p_mean_variance(self, x, t, clip_denoised: bool, cond=None, cond_scale=1.):
        noise = self._predict_noise(x, t, cond=cond, cond_scale=cond_scale)
        x_recon = self.predict_start_from_noise(x, t=t, noise=noise)
        if clip_denoised:
            x_recon = clamp_joint_x0(x_recon)
        model_mean, posterior_variance, posterior_log_variance = self.q_posterior(
            x_start=x_recon, x_t=x, t=t)
        return model_mean, posterior_variance, posterior_log_variance

    @torch.inference_mode()
    def p_sample(self, x, t, cond=None, cond_scale=1., clip_denoised=True):
        b, *_ = x.shape
        model_mean, _, model_log_variance = self.p_mean_variance(
            x=x, t=t, clip_denoised=clip_denoised, cond=cond, cond_scale=cond_scale)
        noise = torch.randn_like(x)
        nonzero_mask = (1 - (t == 0).float()).reshape(b,
                                                      *((1,) * (len(x.shape) - 1)))
        return model_mean + nonzero_mask * (0.5 * model_log_variance).exp() * noise
    
    def _apply_guidance(self, sample, real_img, t):
        real_noisy_image = self.q_sample(x_start=real_img, t=t)
        sample = sample.clone()
        sample[:, :1, :, :, :] = real_noisy_image[:, :1, :, :, :]
        return sample

    def _init_joint_sample(self, shape_image, shape_mask, device):
        img = torch.randn(shape_image, device=device)
        mask = torch.randn(shape_mask, device=device)
        return torch.cat((img, mask), dim=1)

    def ddim_step(self, x_t, t, eps, t_prev=None, eta=0.0, clip_x0=True):
        alpha_bar_t = extract(self.alphas_cumprod, t, x_t.shape)
        x0_hat = self.predict_start_from_noise(x_t, t=t, noise=eps)
        if clip_x0:
            x0_hat = clamp_joint_x0(x0_hat)
        if t_prev is None:
            return x0_hat, x0_hat

        alpha_bar_prev = extract(self.alphas_cumprod, t_prev, x_t.shape)
        sigma = (
            eta
            * torch.sqrt((1 - alpha_bar_prev) / (1 - alpha_bar_t))
            * torch.sqrt(torch.clamp(1 - alpha_bar_t / alpha_bar_prev, min=0.0))
        )
        noise = torch.randn_like(x_t) if eta > 0 else torch.zeros_like(x_t)
        dir_xt = torch.sqrt(torch.clamp(1 - alpha_bar_prev - sigma ** 2, min=0.0)) * eps
        x_prev = torch.sqrt(alpha_bar_prev) * x0_hat + dir_xt + sigma * noise
        return x_prev, x0_hat

    @torch.inference_mode()
    def p_sample_loop_ddpm_replace(self, shape_image, shape_mask, cond=None, cond_scale=1., device=None, image=None):
        b = shape_image[0]
        sample = self._init_joint_sample(shape_image, shape_mask, device)
        if image is None:
            raise ValueError('Guided sampling requires a real input image')

        for i in range(self.num_timesteps - 1, -1, -1):
            timestep = torch.full((b,), i, dtype=torch.long, device=device)
            sample = self._apply_guidance(sample, image, timestep)
            sample = self.p_sample(sample, timestep, cond=cond, cond_scale=cond_scale)
        return sample

    @torch.inference_mode()
    def sample_loop_ddim(self, shape_image, shape_mask, cond=None, cond_scale=1., device=None, image=None, ddim_steps=50, eta=0.0, guidance_mode='none', guidance_cfg=None):
        b = shape_image[0]
        sample = self._init_joint_sample(shape_image, shape_mask, device)
        timesteps = make_ddim_timesteps(ddim_steps, self.num_timesteps).to(device)

        for i, t_scalar in enumerate(timesteps):
            t = torch.full((b,), int(t_scalar.item()), dtype=torch.long, device=device)
            if guidance_mode == 'replace':
                if image is None:
                    raise ValueError('Replace guidance requires a real input image')
                sample = self._apply_guidance(sample, image, t)

            eps = self._predict_noise(sample, t, cond=cond, cond_scale=cond_scale)
            t_prev = None
            if i + 1 < len(timesteps):
                t_prev_scalar = timesteps[i + 1]
                t_prev = torch.full((b,), int(t_prev_scalar.item()), dtype=torch.long, device=device)
            sample, _ = self.ddim_step(sample, t, eps, t_prev=t_prev, eta=eta)
        return sample

    def _compute_guidance_loss(self, image_hat, image, guidance_cfg, lncc_loss, edge_loss):
        loss = image_hat.new_tensor(0.0)
        lambda_lncc = float(cfg_get(guidance_cfg, 'lambda_lncc', 1.0))
        lambda_edge = float(cfg_get(guidance_cfg, 'lambda_edge', 0.0))
        if lambda_lncc > 0:
            loss = loss + lambda_lncc * lncc_loss(image_hat, image)
        if lambda_edge > 0:
            loss = loss + lambda_edge * edge_loss(image_hat, image)
        return loss

    def _log_guidance(self, guidance_cfg, t_scalar, stats):
        if not cfg_get(guidance_cfg, 'log_every_step', False):
            return
        print(
            f"        step={int(t_scalar)} loss={stats['loss']:.6f} grad_norm={stats['grad_norm']:.6f} "
            f"gamma={stats['gamma']:.6f} i0_absmax={stats['i0_absmax']:.6f} s0_absmax={stats['s0_absmax']:.6f} "
            f"xvar_pre={stats['x_var_min_pre_clamp']:.6e} yvar_pre={stats['y_var_min_pre_clamp']:.6e}"
        )

    def sample_loop_ddim_dps(self, shape_image, shape_mask, cond=None, cond_scale=1., device=None, image=None, ddim_steps=50, eta=0.0, guidance_cfg=None, guidance_mode='dps'):
        if image is None:
            raise ValueError('DPS guidance requires a real input image')

        b = shape_image[0]
        sample = self._init_joint_sample(shape_image, shape_mask, device)
        timesteps = make_ddim_timesteps(ddim_steps, self.num_timesteps).to(device)
        lncc_loss = LNCCLoss3D(win=int(cfg_get(guidance_cfg, 'lncc_win', 9))).to(device)
        edge_loss = EdgeLoss3D().to(device)

        for i, t_scalar in enumerate(timesteps):
            t = torch.full((b,), int(t_scalar.item()), dtype=torch.long, device=device)
            t_prev = None
            if i + 1 < len(timesteps):
                t_prev_scalar = timesteps[i + 1]
                t_prev = torch.full((b,), int(t_prev_scalar.item()), dtype=torch.long, device=device)

            sample = sample.detach().to(dtype=torch.float32).requires_grad_(True)
            eps_theta = self._predict_noise(sample, t, cond=cond, cond_scale=cond_scale)
            x0_hat = clamp_joint_x0(self.predict_start_from_noise(sample, t=t, noise=eps_theta))
            image_hat = x0_hat[:, :1]
            loss = self._compute_guidance_loss(image_hat, image, guidance_cfg, lncc_loss, edge_loss)
            grad = torch.autograd.grad(loss, sample, retain_graph=False, create_graph=False)[0]

            if cfg_get(guidance_cfg, 'apply_to', 'mask_only') == 'mask_only':
                grad = grad.clone()
                grad[:, :1] = 0

            grad_flat = grad.reshape(b, -1)
            grad_norm = torch.linalg.norm(grad_flat, dim=1).view(b, 1, 1, 1, 1).clamp_min(1e-8)
            grad_unit = grad / grad_norm
            grad_clip = float(cfg_get(guidance_cfg, 'grad_clip', 1.0))
            grad_unit = torch.clamp(grad_unit, min=-grad_clip, max=grad_clip)

            gamma_t = build_guidance_scale(
                t=t,
                total_steps=self.num_timesteps,
                base_scale=float(cfg_get(guidance_cfg, 'gamma', 0.5)),
                mode=cfg_get(guidance_cfg, 'gamma_schedule', 'mid'),
            ).view(b, 1, 1, 1, 1)
            alpha_bar_t = extract(self.alphas_cumprod, t, sample.shape)
            eps_guided = eps_theta - torch.sqrt(1.0 - alpha_bar_t) * gamma_t * grad_unit
            sample_next, x0_hat = self.ddim_step(sample.detach(), t, eps_guided.detach(), t_prev=t_prev, eta=eta)

            if guidance_mode == 'hybrid' and t_prev is not None:
                noisy_img = self.q_sample(x_start=image, t=t_prev)
                sample_next[:, :1] = noisy_img[:, :1]

            stats = {
                'loss': float(loss.detach().cpu()),
                'grad_norm': float(grad_norm.mean().detach().cpu()),
                'gamma': float(gamma_t.mean().detach().cpu()),
                'i0_absmax': float(image_hat.abs().max().detach().cpu()),
                's0_absmax': float(x0_hat[:, 1:].abs().max().detach().cpu()),
                'x_var_min_pre_clamp': lncc_loss.last_stats.get('x_var_min_pre_clamp', 0.0),
                'y_var_min_pre_clamp': lncc_loss.last_stats.get('y_var_min_pre_clamp', 0.0),
            }
            self._log_guidance(guidance_cfg, t_scalar, stats)
            sample = sample_next.detach()
        return sample

    def sample(self, shape_image, shape_mask, cond=None, cond_scale=1., device=None, image=None, sampler='ddpm', ddim_steps=50, eta=0.0, guidance_mode='replace', guidance_cfg=None):
        if sampler == 'ddpm':
            if guidance_mode == 'replace':
                return self.p_sample_loop_ddpm_replace(
                    shape_image=shape_image,
                    shape_mask=shape_mask,
                    cond=cond,
                    cond_scale=cond_scale,
                    device=device,
                    image=image,
                )
            if guidance_mode == 'none':
                return self.sample_loop_ddim(
                    shape_image=shape_image,
                    shape_mask=shape_mask,
                    cond=cond,
                    cond_scale=cond_scale,
                    device=device,
                    image=None,
                    ddim_steps=self.num_timesteps,
                    eta=1.0,
                    guidance_mode='none',
                    guidance_cfg=guidance_cfg,
                )
            raise ValueError(f"Unsupported ddpm guidance mode: {guidance_mode}")

        if sampler == 'ddim':
            if guidance_mode in ('none', 'replace'):
                return self.sample_loop_ddim(
                    shape_image=shape_image,
                    shape_mask=shape_mask,
                    cond=cond,
                    cond_scale=cond_scale,
                    device=device,
                    image=image,
                    ddim_steps=ddim_steps,
                    eta=eta,
                    guidance_mode=guidance_mode,
                    guidance_cfg=guidance_cfg,
                )
            if guidance_mode in ('dps', 'hybrid'):
                return self.sample_loop_ddim_dps(
                    shape_image=shape_image,
                    shape_mask=shape_mask,
                    cond=cond,
                    cond_scale=cond_scale,
                    device=device,
                    image=image,
                    ddim_steps=ddim_steps,
                    eta=eta,
                    guidance_cfg=guidance_cfg,
                    guidance_mode=guidance_mode,
                )
            raise ValueError(f"Unsupported ddim guidance mode: {guidance_mode}")

        raise ValueError(f"Unsupported sampler: {sampler}")

    @torch.inference_mode()
    def p_sample_loop(self, shape_image, shape_mask, cond=None, cond_scale=1., device=None, image=None):
        if self.use_guide:
            return self.p_sample_loop_ddpm_replace(
                shape_image=shape_image,
                shape_mask=shape_mask,
                cond=cond,
                cond_scale=cond_scale,
                device=device,
                image=image,
            )
        return self.sample_loop_ddim(
            shape_image=shape_image,
            shape_mask=shape_mask,
            cond=cond,
            cond_scale=cond_scale,
            device=device,
            image=None,
            ddim_steps=self.num_timesteps,
            eta=1.0,
            guidance_mode='none',
            guidance_cfg=None,
        )

    def q_sample(self, x_start, t, noise=None):
        noise = default(noise, lambda: torch.randn_like(x_start))
        return (
            extract(self.sqrt_alphas_cumprod, t, x_start.shape) * x_start +
            extract(self.sqrt_one_minus_alphas_cumprod,
                    t, x_start.shape) * noise
        )

    def p_losses(self, x_start, t, mask_start, cond=None, noise_x=None, noise_m=None, **kwargs):
        device = x_start.device
        x_start = x_start.to(device=device, dtype=torch.float32)

        mask_start = mask_start.to(device=device, dtype=torch.float32)

        noise_x = default(noise_x, lambda: torch.randn_like(x_start))
        noise_m = default(noise_m, lambda: torch.randn_like(mask_start))

        x_noisy = self.q_sample(x_start=x_start, t=t, noise=noise_x)
        m_noisy = self.q_sample(x_start=mask_start, t=t, noise=noise_m)
        
        input = torch.cat((x_noisy, m_noisy), dim=1)

        if is_list_str(cond):
            cond = bert_embed(
                tokenize(cond), return_cls_repr=self.text_use_bert_cls)
            cond = cond.to(device)

        recon = self.denoise_fn(**dict(x=input, time=t, cond=cond, **kwargs))
        x_recon = recon[:,0,:,:,:]
        x_recon = x_recon.unsqueeze(1)
        m_recon = recon[:,1:(recon.size()[1]),:,:,:]
        m_recon = m_recon.squeeze(1)
        noise_m = noise_m.squeeze(1)
        if self.loss_type == 'l1':
            loss = F.l1_loss(noise_x, x_recon) + F.l1_loss(noise_m, m_recon)
        elif self.loss_type == 'l2':
            loss = F.mse_loss(noise_x, x_recon) + F.mse_loss(noise_m, m_recon)
        else:
            raise NotImplementedError()
        return loss
    
    def forward(self, x, mask, *args, **kwargs):
        b, device, img_size, = x.shape[0], x.device, self.image_size
        t = torch.randint(0, self.num_timesteps, (b,), device=device).long().to(self.device)
        return self.p_losses(**dict(x_start=x, t=t, mask_start=mask, *args, **kwargs))



class Trainer(object):
    def __init__(
        self,
        diffusion_model,
        cfg,
        dataset=None,
        *,
        ema_decay=0.995,
        train_batch_size=32,
        train_lr=1e-4,
        train_num_steps=100000,
        gradient_accumulate_every=2,
        amp=False,
        step_start_ema=2000,
        update_ema_every=10,
        save_and_sample_every=1000,
        results_folder='./results',
        max_grad_norm=None,
        num_workers=4,
        device=None,
    ):
        super().__init__()
        self.model = diffusion_model
        self.ema = EMA(ema_decay)
        self.ema_model = copy.deepcopy(self.model)
        self.update_ema_every = update_ema_every

        self.step_start_ema = step_start_ema
        self.save_and_sample_every = save_and_sample_every

        self.batch_size = train_batch_size
        self.image_size = diffusion_model.image_size
        self.gradient_accumulate_every = gradient_accumulate_every
        self.train_num_steps = train_num_steps
        self.device = device

        self.cfg = cfg

        self.ds = dataset
        dl = DataLoader(self.ds, batch_size=train_batch_size,
                        shuffle=True, pin_memory=True, num_workers=num_workers)

        self.len_dataloader = len(dl)
        print("len_dl ", len(dl))
        self.dl = cycle(dl)

        print(f'found {len(self.ds)} videos as gif files')
        assert len(
            self.ds) > 0, 'need to have at least 1 video to start training (although 1 is not great, try 100k)'

        self.opt = Adam(diffusion_model.parameters(), lr=train_lr)

        self.step = 0

        self.amp = amp
        self.scaler = GradScaler(enabled=amp)
        self.max_grad_norm = max_grad_norm

        self.results_folder = Path(results_folder)
        self.results_folder.mkdir(exist_ok=True, parents=True)

        self.reset_parameters()

    def reset_parameters(self):
        self.ema_model.load_state_dict(self.model.state_dict())

    def step_ema(self):
        if self.step < self.step_start_ema:
            self.reset_parameters()
            return
        self.ema.update_model_average(self.ema_model, self.model)

    def save(self, milestone):
        data = {
            'step': self.step,
            'model': self.model.state_dict(),
            'ema': self.ema_model.state_dict(),
            'scaler': self.scaler.state_dict(),
            'optimizer': self.opt.state_dict()  
        }
        torch.save(data, str(self.results_folder / f'model-{milestone}.pt'))

    def _resolve_checkpoint_path(self, milestone):
        if isinstance(milestone, (str, Path)):
            candidate = Path(milestone)
            if candidate.suffix == '.pt' or candidate.exists():
                return candidate
            milestone = int(milestone)
        return self.results_folder / f'model-{milestone}.pt'

    def load(self, milestone, map_location=None, **kwargs):
        if milestone == -1:
            all_milestones = [
                int(p.stem.split('-')[-1])
                for p in Path(self.results_folder).glob('model-*.pt')
            ]
            assert len(
                all_milestones) > 0, 'need to have at least one milestone to load from latest checkpoint (milestone == -1)'
            milestone = max(all_milestones)

        checkpoint_path = self._resolve_checkpoint_path(milestone)
        data = torch.load(str(checkpoint_path), map_location=map_location)

        self.step = data['step']
        self.model.load_state_dict(data['model'], **kwargs)
        self.ema_model.load_state_dict(data['ema'], **kwargs)
        self.scaler.load_state_dict(data['scaler'])
        self.opt.load_state_dict(data['optimizer'])

    def train(
        self,
        prob_focus_present=0.,
        focus_present_mask=None,
        log_fn=noop
    ):
        assert callable(log_fn)
        self.opt.zero_grad()

        while self.step < self.train_num_steps:
            for i in range(self.gradient_accumulate_every):

                data_frame = next(self.dl)
                data = data_frame['img'].to(self.device, dtype=torch.float32)
                mask_sdf = data_frame['mask_sdf'].to(self.device, dtype=torch.float32)

                with autocast(enabled=self.amp):

                    loss = self.model(**dict(
                        x=data,
                        mask=mask_sdf,
                        prob_focus_present=prob_focus_present,
                        focus_present_mask=focus_present_mask)
                    )

                    self.scaler.scale(
                        loss / self.gradient_accumulate_every).backward()

            log = {'loss': loss.item()}

            if exists(self.max_grad_norm):
                self.scaler.unscale_(self.opt)
                nn.utils.clip_grad_norm_(
                    self.model.parameters(), self.max_grad_norm)

            self.scaler.step(self.opt)
            self.scaler.update()
            self.opt.zero_grad()
            self.step += 1

            if self.step % self.update_ema_every == 0:
                self.step_ema()

            if self.step % self.save_and_sample_every == 0:
                self.ema_model.eval()
                milestone = self.step // self.save_and_sample_every
                self.save(milestone)

            print(f'{self.step}: {loss.item()}')
            log_fn(log)

        print('training completed')
