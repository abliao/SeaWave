from functools import wraps, partial

import torch
import torch.nn.functional as F
from torch import nn, einsum

from einops import rearrange, repeat, pack, unpack

from beartype import beartype
from beartype.typing import Callable, Tuple, Optional, List, Literal, Union
from beartype.door import is_bearable

from inspect import signature

from .t5 import T5Adapter
from .open_clip import OpenClipAdapter

# constants

COND_DROP_KEY_NAME = 'cond_drop_prob'

TEXTS_KEY_NAME = 'texts'
TEXT_EMBEDS_KEY_NAME = 'text_embeds'
TEXT_CONDITIONER_NAME = 'text_conditioner'
CONDITION_FUNCTION_KEY_NAME = 'cond_fns'

# helper functions

def exists(val):
    return val is not None

def default(*values):
    for value in values:
        if exists(value):
            return value
    return None

def cast_tuple(val, length = 1):
    return val if isinstance(val, tuple) else ((val,) * length)

def pack_one(x, pattern):
    return pack([x], pattern)

def unpack_one(x, ps, pattern):
    return unpack(x, ps, pattern)[0]

# tensor helpers

def prob_mask_like(shape, prob, device):
    if prob == 1:
        return torch.ones(shape, device = device, dtype = torch.bool)
    elif prob == 0:
        return torch.zeros(shape, device = device, dtype = torch.bool)
    else:
        return torch.zeros(shape, device = device).float().uniform_(0, 1) < prob

# classifier free guidance with automatic text conditioning

@beartype
def classifier_free_guidance(
    fn: Callable,
    cond_drop_prob_keyname = COND_DROP_KEY_NAME,
    texts_key_name = TEXTS_KEY_NAME,
    text_embeds_key_name = TEXT_EMBEDS_KEY_NAME,
    cond_fns_keyname = CONDITION_FUNCTION_KEY_NAME,
    text_conditioner_name = TEXT_CONDITIONER_NAME
):
    fn_params = signature(fn).parameters

    auto_handle_text_condition = texts_key_name not in fn_params and text_embeds_key_name not in fn_params
    assert not (auto_handle_text_condition and cond_fns_keyname not in fn_params), f'{cond_fns_keyname} must be in the wrapped function for autohandling texts -> conditioning functions - ex. forward(..., {cond_fns_keyname})'

    @wraps(fn)
    def inner(
        self,
        *args,
        cond_scale: float = 1.,
        **kwargs
    ):
        @wraps(fn)
        def fn_maybe_with_text(self, *args, **kwargs):
            if auto_handle_text_condition:
                texts = kwargs.pop('texts', None)
                text_embeds = kwargs.pop('text_embeds', None)

                assert not (exists(texts) and exists(text_embeds))

                cond_fns = None

                # auto convert texts -> conditioning functions

                if exists(texts) ^ exists(text_embeds):

                    assert is_bearable(texts, Optional[List[str]]), f'keyword `{texts_key_name}` must be a list of strings'

                    text_conditioner = getattr(self, text_conditioner_name, None)
                    assert exists(text_conditioner) and is_bearable(text_conditioner, Conditioner), 'text_conditioner must be set on your network with the correct hidden dimensions to be conditioned on'

                    cond_drop_prob = kwargs.pop(cond_drop_prob_keyname, None)

                    text_condition_input = dict(texts = texts) if exists(texts) else dict(text_embeds = text_embeds)

                    cond_fns = text_conditioner(**text_condition_input, cond_drop_prob = cond_drop_prob)

                kwargs.update(cond_fns = cond_fns)

            return fn(self, *args, **kwargs)

        # main classifier free guidance logic

        if self.training:
            assert cond_scale == 1, 'you cannot do condition scaling when in training mode'

            return fn_maybe_with_text(self, *args, **kwargs)

        assert cond_scale >= 1, 'invalid conditioning scale, must be greater or equal to 1'
        
        kwargs_without_cond_dropout = {**kwargs, cond_drop_prob_keyname: 0.}
        kwargs_with_cond_dropout = {**kwargs, cond_drop_prob_keyname: 1.}

        logits = fn_maybe_with_text(self, *args, **kwargs_without_cond_dropout)

        if cond_scale == 1:
            return logits

        null_logits = fn_maybe_with_text(self, *args, **kwargs_with_cond_dropout)
        return null_logits + (logits - null_logits) * cond_scale

    return inner

# attention

class Attention(nn.Module):
    def __init__(
        self,
        dim,
        dim_head = 64,
        heads = 8,
        dim_context = None,
        norm_context = False,
        num_null_kv = 0
    ):
        super().__init__()
        self.heads = heads
        self.scale = dim_head ** -0.5
        inner_dim = dim_head * heads

        dim_context = default(dim_context, dim)

        self.norm = nn.LayerNorm(dim)
        self.context_norm = nn.LayerNorm(dim_context) if norm_context else nn.Identity()

        self.num_null_kv = num_null_kv
        self.null_kv = nn.Parameter(torch.randn(2, num_null_kv, dim_head))

        self.to_q = nn.Linear(dim, inner_dim, bias = False)
        self.to_kv = nn.Linear(dim_context, dim_head * 2, bias = False)
        self.to_out = nn.Linear(inner_dim, dim, bias = False)

    def forward(
        self,
        x,
        context = None,
        mask = None
    ):
        b = x.shape[0]

        if exists(context):
            context = self.context_norm(context)

        kv_input = default(context, x)

        x = self.norm(x)

        q, k, v = self.to_q(x), *self.to_kv(kv_input).chunk(2, dim = -1)

        if self.num_null_kv > 0:
            null_k, null_v = repeat(self.null_kv, 'kv n d -> kv b n d', b = b).unbind(dim = 0)
            k = torch.cat((null_k, k), dim = -2)
            v = torch.cat((null_v, v), dim = -2)

        q = rearrange(q, 'b n (h d) -> b h n d', h = self.heads)

        q = q * self.scale

        sim = einsum('b h i d, b j d -> b h i j', q, k)

        if exists(mask):
            mask = F.pad(mask, (self.num_null_kv, 0), value = True)
            mask = rearrange(mask, 'b j -> b 1 1 j')
            sim = sim.masked_fill(~mask, -torch.finfo(sim.dtype).max)

        attn = sim.softmax(dim = -1)

        out = einsum('b h i j, b j d -> b h i d', attn, v)

        out = rearrange(out, 'b h n d -> b n (h d)')
        return self.to_out(out)

# dimension adapters

def rearrange_channel_last(fn):
    @wraps(fn)
    def inner(hiddens):
        hiddens, ps = pack_one(hiddens, 'b * d')
        conditioned = fn(hiddens)
        return unpack_one(conditioned, ps, 'b * d')
    return inner

def rearrange_channel_first(fn):
    """ will adapt shape of (batch, feature, ...) for conditioning """

    @wraps(fn)
    def inner(hiddens):
        hiddens, ps = pack_one(hiddens, 'b d *')
        hiddens = rearrange(hiddens, 'b d n -> b n d')
        conditioned =  fn(hiddens)
        conditioned = rearrange(conditioned, 'b n d -> b d n')
        return unpack_one(conditioned, ps, 'b d *')

    return inner

# conditioning modules

class FiLM(nn.Module):
    def __init__(
        self,
        dim,
        hidden_dim
    ):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim * 2),
            nn.SiLU(),
            nn.Linear(hidden_dim * 2, hidden_dim * 2)
        )

        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)

    def forward(self, conditions, hiddens):
        
        scale, shift = self.net(conditions).chunk(2, dim = -1)
        assert scale.shape[-1] == hiddens.shape[-1], f'unexpected hidden dimesion {hiddens.shape[-1]} used for conditioning'
        scale, shift = map(lambda t: rearrange(t, 'b d -> b 1 d'), (scale, shift))
        # print('FiLM')
        # print(scale[0][0][0])
        # print(shift[0][0][0])
        return hiddens * (scale + 1) + shift

class CrossAttention(nn.Module):
    def __init__(
        self,
        dim,
        hidden_dim,
        heads = 8,
        dim_head = 64
    ):
        super().__init__()
        self.attn = Attention(
            dim = hidden_dim,
            dim_context = dim,
            norm_context = True,
            num_null_kv = 1,
            dim_head = dim_head,
            heads = heads
        )

    def forward(
        self,
        condition,
        hiddens,
        mask = None
    ):
        return self.attn(hiddens, condition, mask = mask) + hiddens

# film text conditioning

CONDITION_CONFIG = dict(
    t5 = T5Adapter,
    clip = OpenClipAdapter
)

MODEL_TYPES = CONDITION_CONFIG.keys()

class Conditioner(nn.Module):
    pass


class TextEmbedder(nn.Module):
    def __init__(
        self,
        *,
        model_types = 't5',
        model_names = None,
        cond_drop_prob = 0.,
        text_embed_stem_dim = 512
    ):
        super().__init__()
        model_types = cast_tuple(model_types)
        model_names = cast_tuple(model_names, length = len(model_types))

        assert len(model_types) == len(model_names)
        assert all([model_type in MODEL_TYPES for model_type in model_types])

        text_models = []

        for model_type, model_name in zip(model_types, model_names):
            klass = CONDITION_CONFIG.get(model_type)
            model = klass(model_name)
            text_models.append(model)

        self.text_models = text_models
        self.latent_dims = [model.dim_latent for model in text_models]

        self.cond_drop_prob = cond_drop_prob

        total_latent_dim = sum(self.latent_dims)
        mlp_stem_output_dim = text_embed_stem_dim

        self.text_embed_stem_mlp = nn.Sequential(
            nn.Linear(total_latent_dim, mlp_stem_output_dim),
            nn.SiLU()
        )

        self.null_text_embed = nn.Parameter(torch.randn(total_latent_dim))

        self.register_buffer('_device_param', torch.tensor(0.), persistent = False)

    @property
    def device(self):
        return next(self.buffers()).device

    def embed_texts(self, texts: Union[List[str],Tuple[str]]):
        device = self.device

        text_embeds = []
        for text_model in self.text_models:
            text_model.to(device)
            text_embed = text_model.embed_text(texts)
            text_embeds.append(text_embed.to(device))

        return torch.cat(text_embeds, dim = -1)

    def forward(
        self,
        texts: Optional[Union[List[str],Tuple[str]]] = None,
        text_embeds: Optional[torch.Tensor] = None,
        cond_drop_prob = None,
    ) :

        assert exists(texts) ^ exists(text_embeds)

        if self.training:
            cond_drop_prob = default(cond_drop_prob, self.cond_drop_prob)
        else:
            cond_drop_prob = 0.0
            # assert exists(cond_drop_prob), 'when not training, cond_drop_prob must be explicitly set'

        if exists(texts):
            batch = len(texts)
        elif exists(text_embeds):
            batch = text_embeds.shape[0]

        if not exists(text_embeds):
            text_embeds = self.embed_texts(texts)

        if cond_drop_prob > 0.:
            prob_keep_mask = prob_mask_like((batch, 1), 1. - cond_drop_prob, device = self.device)
            null_text_embeds = rearrange(self.null_text_embed, 'd -> 1 d')

            text_embeds = torch.where(
                prob_keep_mask,
                text_embeds,
                null_text_embeds
            )

        # text embed mlp stem, as done in unet conditioning in guided diffusion

        text_embeds = self.text_embed_stem_mlp(text_embeds)
        return text_embeds.unsqueeze(1)
        
    
@beartype
class TextConditioner(Conditioner):
    def __init__(
        self,
        *,
        hidden_dims: Tuple[int, ...],
        model_types = 't5',
        model_names = None,
        cond_drop_prob = 0.,
        hiddens_channel_first = True,
        text_embed_stem_dim = 512
    ):
        super().__init__()
        model_types = cast_tuple(model_types)
        model_names = cast_tuple(model_names, length = len(model_types))

        assert len(model_types) == len(model_names)
        assert all([model_type in MODEL_TYPES for model_type in model_types])

        text_models = []

        for model_type, model_name in zip(model_types, model_names):
            klass = CONDITION_CONFIG.get(model_type)
            model = klass(model_name)
            text_models.append(model)

        self.text_models = text_models
        self.latent_dims = [model.dim_latent for model in text_models]

        self.conditioners = nn.ModuleList([])

        self.hidden_dims = hidden_dims
        self.num_condition_fns = len(hidden_dims)
        self.hiddens_channel_first = cast_tuple(hiddens_channel_first, self.num_condition_fns) # whether hiddens to be conditioned is channel first or last

        assert len(self.hiddens_channel_first) == self.num_condition_fns

        self.cond_drop_prob = cond_drop_prob

        total_latent_dim = sum(self.latent_dims)
        mlp_stem_output_dim = text_embed_stem_dim

        self.text_embed_stem_mlp = nn.Sequential(
            nn.Linear(total_latent_dim, mlp_stem_output_dim),
            nn.SiLU()
        )

        for hidden_dim in hidden_dims:
            self.conditioners.append(FiLM(mlp_stem_output_dim, hidden_dim))

        self.null_text_embed = nn.Parameter(torch.randn(total_latent_dim))

        self.register_buffer('_device_param', torch.tensor(0.), persistent = False)

    @property
    def device(self):
        return next(self.buffers()).device

    def embed_texts(self, texts: List[str]):
        device = self.device
        
        text_embeds = []
        for text_model in self.text_models:
            text_model.to(device)
            text_embed = text_model.embed_text(texts)
            text_embeds.append(text_embed.to(device))

        return torch.cat(text_embeds, dim = -1)

    def forward(
        self,
        texts: Optional[List[str]] = None,
        text_embeds: Optional[torch.Tensor] = None,
        cond_drop_prob = None,
        repeat_batch = 1,  # for robotic transformer edge case
    ) -> Tuple[Callable, ...]:

        assert exists(texts) ^ exists(text_embeds)

        if self.training:
            cond_drop_prob = default(cond_drop_prob, self.cond_drop_prob)
        else:
            assert exists(cond_drop_prob), 'when not training, cond_drop_prob must be explicitly set'

        if exists(texts):
            batch = len(texts)
        elif exists(text_embeds):
            batch = text_embeds.shape[0]

        if not exists(text_embeds):
            text_embeds = self.embed_texts(texts)

        if cond_drop_prob > 0.:
            prob_keep_mask = prob_mask_like((batch, 1), 1. - cond_drop_prob, device = self.device)
            null_text_embeds = rearrange(self.null_text_embed, 'd -> 1 d')

            text_embeds = torch.where(
                prob_keep_mask,
                text_embeds,
                null_text_embeds
            )

        # text embed mlp stem, as done in unet conditioning in guided diffusion

        text_embeds = self.text_embed_stem_mlp(text_embeds)

        # prepare the conditioning functions

        repeat_batch = cast_tuple(repeat_batch, self.num_condition_fns)

        cond_fns = []

        for cond, cond_hiddens_channel_first, cond_repeat_batch in zip(self.conditioners, self.hiddens_channel_first, repeat_batch):
            cond_text_embeds = repeat(text_embeds, 'b ... -> (b r) ...', r = cond_repeat_batch)
            cond_fn = partial(cond, cond_text_embeds)

            wrapper_fn = rearrange_channel_first if cond_hiddens_channel_first else rearrange_channel_last

            cond_fns.append(wrapper_fn(cond_fn))

        return tuple(cond_fns)

# cross attention text conditioner

@beartype
class AttentionTextConditioner(Conditioner):
    def __init__(
        self,
        *,
        hidden_dims: Tuple[int, ...],
        model_types = 't5',
        model_names = None,
        cond_drop_prob = 0.,
        hiddens_channel_first = True,
        dim_latent = None,
        attn_dim_head = 64,
        attn_heads = 8
    ):
        super().__init__()
        model_types = cast_tuple(model_types)
        model_names = cast_tuple(model_names, length = len(model_types))

        assert len(model_types) == len(model_names)
        assert all([model_type in MODEL_TYPES for model_type in model_types])

        text_models = []

        for model_type, model_name in zip(model_types, model_names):
            klass = CONDITION_CONFIG.get(model_type)
            model = klass(model_name)
            text_models.append(model)

        self.text_models = text_models

        self.to_latent_dims = nn.ModuleList([])

        dim_latent = default(dim_latent, max([model.dim_latent for model in text_models]))

        for model in text_models:
            self.to_latent_dims.append(nn.Linear(model.dim_latent, dim_latent))

        self.conditioners = nn.ModuleList([])

        self.hidden_dims = hidden_dims
        self.num_condition_fns = len(hidden_dims)
        self.hiddens_channel_first = cast_tuple(hiddens_channel_first, self.num_condition_fns) # whether hiddens to be conditioned is channel first or last

        assert len(self.hiddens_channel_first) == self.num_condition_fns

        self.cond_drop_prob = cond_drop_prob

        for hidden_dim in hidden_dims:
            self.conditioners.append(CrossAttention(dim_latent, hidden_dim))

        self.register_buffer('_device_param', torch.tensor(0.), persistent = False)

    @property
    def device(self):
        return next(self.buffers()).device

    def embed_texts(self, texts: List[str]):
        device = self.device

        text_embeds = []

        for text_model, to_latent in zip(self.text_models, self.to_latent_dims):
            text_model.to(device)
            text_embed = text_model.embed_text(texts, return_text_encodings = True)

            text_embed = text_embed.to(device)

            mask = (text_embed != 0).any(dim = -1)

            text_embed = to_latent(text_embed)
            text_embed = text_embed.masked_fill(~mask[..., None], 0.)

            text_embeds.append(text_embed)

        return torch.cat(text_embeds, dim = -2)

    def forward(
        self,
        texts: Optional[List[str]] = None,
        text_embeds: Optional[torch.Tensor] = None,
        cond_drop_prob = None,
        repeat_batch = 1,  # for robotic transformer edge case
    ) -> Tuple[Callable, ...]:

        assert exists(texts) ^ exists(text_embeds)

        if self.training:
            cond_drop_prob = default(cond_drop_prob, self.cond_drop_prob)
        else:
            assert exists(cond_drop_prob), 'when not training, cond_drop_prob must be explicitly set'

        if exists(texts):
            batch = len(texts)

        elif exists(text_embeds):
            batch = text_embeds.shape[0]

        if not exists(text_embeds):
            text_embeds = self.embed_texts(texts)

        mask = (text_embeds != 0).any(dim = -1)

        if cond_drop_prob > 0.:
            prob_keep_mask = prob_mask_like((batch, 1), 1. - cond_drop_prob, device = self.device)
            mask = mask & prob_keep_mask

        # prepare the conditioning functions

        repeat_batch = cast_tuple(repeat_batch, self.num_condition_fns)

        cond_fns = []

        for cond, cond_hiddens_channel_first, cond_repeat_batch in zip(self.conditioners, self.hiddens_channel_first, repeat_batch):
            cond_text_embeds = repeat(text_embeds, 'b ... -> (b r) ...', r = cond_repeat_batch)
            cond_mask = repeat(mask, 'b ... -> (b r) ...', r = cond_repeat_batch) if exists(mask) else None

            cond_fn = partial(cond, cond_text_embeds, mask = cond_mask)

            wrapper_fn = rearrange_channel_first if cond_hiddens_channel_first else rearrange_channel_last

            cond_fns.append(wrapper_fn(cond_fn))

        return tuple(cond_fns)
