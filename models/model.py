import os
import copy
import math
import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from typing import Tuple, Union
from collections import OrderedDict

import utils.box_ops as box_ops
from clip.model import Transformer, LayerNorm, MLP, QuickGELU
from clip.clip import _download
from .position_encoding import PositionEmbeddingSine
from .matcher import build_matcher
from .criterion import SetCriterion
from torchvision.ops import batched_nms
from .transformer import TransformerDecoderLayer, TransformerDecoder


_MODELS = {
    "RN50": "https://openaipublic.azureedge.net/clip/models/afeb0e10f9e5a86da6080e35cf09123aca3b358a0c3e3b6c78a7b63bc04b6762/RN50.pt",
    "RN101": "https://openaipublic.azureedge.net/clip/models/8fa8567bab74a42d41c5915025a8e4538c3bdbe8804a470a72f30b0d94fab599/RN101.pt",
    "RN50x4": "https://openaipublic.azureedge.net/clip/models/7e526bd135e493cef0776de27d5f42653e6b4c8bf9e0f653bb11773263205fdd/RN50x4.pt",
    "RN50x16": "https://openaipublic.azureedge.net/clip/models/52378b407f34354e150460fe41077663dd5b39c54cd0bfd2b27167a4a06ec9aa/RN50x16.pt",
    "ViT-B/32": "https://openaipublic.azureedge.net/clip/models/40d365715913c9da98579312b702a82c18be219cc2a73407c4526f58eba950af/ViT-B-32.pt",
    "ViT-B/16": "https://openaipublic.azureedge.net/clip/models/5806e77cd80f8b59890b7e101eabd078d9fb84e6937f9e85e4ecb61988df416f/ViT-B-16.pt",
}


class HOIResidualAttentionBlock(nn.Module):
    '''
    [CLS + PATCH], [HOI] attention block in HOI Vision Encoder:
        - [CLS + PATCH] x [CLS + PATCH]: original attention uses CLIP's pretrained weights.
        - [HOI] x [PATCH]: cross-attention between [HOI] tokens and image patches.
        - [HOI] x [CLS + HOI]: HOI sequential parsing.
    '''
    def __init__(self, d_model: int, n_head: int, parse_attn_mask: torch.Tensor = None):
        super().__init__()

        self.hoi_parse_attn = nn.MultiheadAttention(d_model, n_head, dropout=0.1)
        self.hoi_cross_attn = nn.MultiheadAttention(d_model, n_head, dropout=0.1)
        self.attn = nn.MultiheadAttention(d_model, n_head)

        self.ln_1 = LayerNorm(d_model)
        self.mlp = nn.Sequential(OrderedDict([
            ("c_fc", nn.Linear(d_model, d_model * 4)),
            ("gelu", QuickGELU()),
            ("c_proj", nn.Linear(d_model * 4, d_model))
        ]))
        self.ln_2 = LayerNorm(d_model)
        self.parse_attn_mask = parse_attn_mask

        self.hoi_ln1 = LayerNorm(d_model)
        self.hoi_ln2 = LayerNorm(d_model)
        self.dropout1 = nn.Dropout(0.1)
        self.dropout2 = nn.Dropout(0.1)
        self.dropout3 = nn.Dropout(0.1)

    def image_attention(self, image: torch.Tensor, mask: torch.Tensor = None):
        return self.attn(image, image, image, need_weights=False, key_padding_mask=mask)[0]

    def hoi_attention(self, hoi: torch.Tensor, attn_mask: torch.Tensor = None):
        attn_mask = attn_mask.type_as(hoi) if attn_mask is not None else None
        hoi, attn_map = self.hoi_parse_attn(hoi, hoi, hoi, attn_mask=attn_mask)
        return hoi

    def forward(self, image: torch.Tensor, hoi: torch.Tensor, mask: torch.Tensor = None):

        # [HOI] x [PATCH] cross attention. [CLS] is masked out.
        y, attn_map = self.hoi_cross_attn(self.ln_1(hoi), self.ln_1(image), self.ln_1(image), key_padding_mask=mask)
        hoi = hoi + self.dropout1(y)
        hoi = hoi + self.dropout2(self.mlp(self.ln_2(hoi)))

        # [CLS + PATCH] x [CLS + PATCH] using pretrained CLIP's weights
        image_mask = mask.clone()
        image_mask[:, 0] = False # enable [CLS] token for attention
        image = image + self.image_attention(self.ln_1(image), image_mask)
        image = image + self.mlp(self.ln_2(image))

        # [HOI] x [CLS + HOI], HOI sequential parsing
        x = torch.cat([image[0:1], hoi], dim=0)
        x = x + self.dropout3(self.hoi_attention(self.hoi_ln1(x), attn_mask=self.parse_attn_mask))
        hoi = x[1:]

        return image, hoi, attn_map


class HOITransformer(nn.Module):
    def __init__(self, width: int, layers: int, heads: int, attn_mask: torch.Tensor=None):
        super().__init__()
        self.width = width
        self.layers = layers
        self.resblocks = nn.Sequential(*[HOIResidualAttentionBlock(width, heads, attn_mask) for _ in range(layers)])

    def forward(self, image: torch.Tensor, hoi: torch.Tensor, mask: torch.Tensor = None):
        for resblock in self.resblocks:
            image, hoi, attn_map = resblock(image, hoi, mask)
        return image, hoi, attn_map


class HOIVisionTransformer(nn.Module):
    """ This module encodes RGB images and outputs HOI bounding box predictions and projected
        feature vectors in joint vision-and-text feature space.
    """
    def __init__(
        self,
        # vision backbone
        image_resolution: int,
        patch_size: int,
        width: int,
        layers: int,
        heads: int,
        output_dim: int,
        hoi_token_length: int = 5,
        hoi_parser_attn_mask: torch.Tensor = None,
        # bounding box head
        enable_dec: bool = False,
        dec_heads: int = 8,
        dec_layers: int = 6,
    ):
        super().__init__()
        self.image_resolution = image_resolution
        self.hoi_token_length = hoi_token_length
        self.output_dim = output_dim
        self.patch_size = patch_size
        # Weights in original CLIP model.
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=width, kernel_size=patch_size, stride=patch_size, bias=False)
        scale = width ** -0.5
        self.class_embedding = nn.Parameter(scale * torch.randn(width))
        self.positional_embedding = nn.Parameter(scale * torch.randn((image_resolution // patch_size) ** 2 + 1, width))
        self.ln_pre = LayerNorm(width)
        self.ln_post = LayerNorm(width)
        self.proj = nn.Parameter(scale * torch.randn(width, output_dim))

        # Modified Transformer blocks
        self.transformer = HOITransformer(width, layers, heads, hoi_parser_attn_mask)

        # Additional parameters for HOI detection
        self.hoi_token_embed = nn.Parameter(scale * torch.randn(hoi_token_length, width))
        self.hoi_pos_embed = nn.Parameter(scale * torch.randn(hoi_token_length, width))

        # Additional parameters for detection head
        self.enable_dec = enable_dec
        if enable_dec:
            self.image_patch_pos = PositionEmbeddingSine(width // 2, normalize=True)
            self.hoi_parser_attn_mask = hoi_parser_attn_mask
            decoder_layer = TransformerDecoderLayer(width, dec_heads, normalize_before=True)
            decoder_norm = LayerNorm(width)
            self.bbox_head = TransformerDecoder(decoder_layer, dec_layers, decoder_norm, True)

        self.bbox_score = nn.Linear(width, 1)
        self.bbox_embed = MLP(width, width, 8, 3)

        self.initialize_parameters()

    def initialize_parameters(self):
        nn.init.xavier_uniform_(self.bbox_score.weight, gain=1)
        nn.init.constant_(self.bbox_score.bias, 0)

        for layer in self.bbox_embed.layers:
            nn.init.xavier_uniform_(layer.weight, gain=1)
            nn.init.constant_(layer.bias, 0)

    def interpolate_pos_embedding(self, x, mask):
        """ Using fixed positional embedding to handle the changing image resolution.
        Refer to https://github.com/facebookresearch/dino/blob/main/vision_transformer.py#L174
        """
        ori_h = (~mask).cumsum(1, dtype=torch.float32)[:, -1, 0]
        ori_w = (~mask).cumsum(2, dtype=torch.float32)[:, 0, -1]
        ori_shapes = [(int(h), int(w)) for h, w in zip(ori_h, ori_w)]
        bs, h, w = mask.shape

        npatch = x.shape[0] - 1
        dim = x.shape[1]

        class_pos_embed = x[0, :]
        patch_pos_embed = x[1:, :]

        w0, h0 = w // self.patch_size, h // self.patch_size
        interploated_pos_embed = torch.zeros(bs, h0, w0, dim).type_as(x)
        # Add a small number to avoid floating point error in the interpolation
        # see discussion at https://github.com/facebookresearch/dino/issues/8
        for i, (hi, wi) in enumerate(ori_shapes):
            w0, h0 = wi // self.patch_size, hi // self.patch_size
            w0, h0 = w0 + 0.1, h0 + 0.1
            interploated = nn.functional.interpolate(
                patch_pos_embed.reshape(1, int(math.sqrt(npatch)), int(math.sqrt(npatch)), dim).permute(0, 3, 1, 2),
                scale_factor=(h0 / math.sqrt(npatch), w0 / math.sqrt(npatch)),
                mode='bicubic',
            )
            assert int(h0) == interploated.shape[-2] and int(w0) == interploated.shape[-1]
            interploated = interploated.permute(0, 2, 3, 1)
            interploated_pos_embed[i, :int(h0), :int(w0), :] = interploated

        interploated_pos_embed = interploated_pos_embed.view(bs, -1, dim)
        interploated_pos_embed = torch.cat([class_pos_embed + torch.zeros(bs, 1, dim).type_as(x), interploated_pos_embed], dim=1)
        return interploated_pos_embed

    def forward(self, image: torch.Tensor, mask: torch.Tensor = None):
        """ HOI visual encoder forward """
        # CLIP has fixed set of pos embedding. We apply interpolation to handle different image resolution.
        patch_pos_embed = self.interpolate_pos_embedding(self.positional_embedding, mask)

        image = self.conv1(image)  # shape = [*, width, grid, grid]
        bs, c = image.shape[0], image.shape[1]
        image = image.reshape(bs, c, -1)  # shape = [*, width, grid ** 2]
        image = image.permute(0, 2, 1)  # shape = [*, grid ** 2, width]

        if mask is not None:
            # mask = F.avg_pool2d(mask.float(), kernel_size=self.patch_size)
            # mask[mask < 1.] = 0.
            # mask = mask.bool()
            mask = F.avg_pool2d(mask.float(), kernel_size=self.patch_size).bool()
            mask_flatten = mask.reshape(mask.shape[0], -1) # shape = [*, grid ** 2]
            mask_flatten = torch.cat([torch.ones((mask_flatten.shape[0], 1), dtype=mask.dtype, device=mask.device), mask_flatten], dim=-1)

        # [CLS], [PATCH]_1, [PATCH]_2, ..., [PATCH]_N
        image = torch.cat([self.class_embedding + torch.zeros(bs, 1, c).type_as(image), image], dim=1)  # shape = [*, grid ** 2 + 1, width]
        # [HOI]_1, [HOI]_2, ..., [HOI]_n
        hoi = self.hoi_token_embed + torch.zeros(bs, self.hoi_token_length, c).type_as(image)
        # Positional embedding
        image = image + patch_pos_embed
        hoi = hoi + self.hoi_pos_embed

        image = self.ln_pre(image)
        hoi = self.ln_pre(hoi)
        # HOI visual transformers
        image = image.permute(1, 0, 2)  # NLD -> LND
        hoi = hoi.permute(1, 0, 2)  # NLD -> LND
        image, hoi, attn_map = self.transformer(image, hoi, mask_flatten)
        image = image.permute(1, 0, 2)  # LND -> NLD
        hoi = hoi.permute(1, 0, 2)  # LND -> NLD

        # Map to joint vision-and-text feature space
        image_features = self.ln_post(image[:, 0, :])
        hoi_features = self.ln_post(hoi)
        image_features = image_features @ self.proj
        hoi_features = hoi_features @ self.proj

        # Bounding box head
        if self.enable_dec:
            patch_pos = self.image_patch_pos(mask) # sin/cos pos embedding for bbox decoding
            patch_pos = patch_pos.flatten(-2).permute(2, 0, 1).type_as(image)

            hoi = hoi.permute(1, 0, 2) # NLD -> LND
            image = image.permute(1, 0, 2) # NLD -> LND

            hidden = self.bbox_head(
                tgt=hoi,
                tgt_mask=self.hoi_parser_attn_mask[1:, 1:].to(hoi.device), # exclude [CLS]
                query_pos=self.hoi_pos_embed[:, None, :],
                memory=image[1:], # exclude [CLS]
                memory_key_padding_mask=mask_flatten[:, 1:], # exclude [CLS]
                pos=patch_pos)

            box_scores = self.bbox_score(hidden) # [layers, L, N, 1]
            pred_boxes = self.bbox_embed(hidden).sigmoid() # [layers, L, N, 8]
            box_scores = box_scores.permute(0, 2, 1, 3) # [layers, N, L, 1]
            pred_boxes = pred_boxes.permute(0, 2, 1, 3) # [layers, N, L, 8]
            aux_outputs = [{"pred_boxes": a, "box_scores": b} for a, b in zip(pred_boxes[:-1], box_scores[:-1])]

            return_dict = {"image_features": image_features,
                           "hoi_features": hoi_features,
                           "pred_boxes": pred_boxes[-1],
                           "box_scores": box_scores[-1],
                           "attn_maps": attn_map,
                           "aux_outputs": aux_outputs}
        else:
            box_scores = self.bbox_score(hoi)
            pred_boxes = self.bbox_embed(hoi).sigmoid()
            return_dict = {"image_features": image_features,
                           "hoi_features": hoi_features,
                           "pred_boxes": pred_boxes,
                           "box_scores": box_scores,
                           "attn_maps": attn_map}
        return return_dict


class HOIDetector(nn.Module):
    def __init__(
        self,
        embed_dim: int,
        # vision
        image_resolution: int,
        vision_layers: Union[Tuple[int, int, int, int], int],
        vision_width: int,
        vision_patch_size: int,
        hoi_token_length: int,
        # detection head
        enable_dec: bool,
        dec_heads: int,
        dec_layers: int,
        # text
        context_length: int,
        vocab_size: int,
        transformer_width: int,
        transformer_heads: int,
        transformer_layers: int,
        prefix_length: int = 8,
        conjun_length: int = 4,
    ):
        super().__init__()

        self.context_length = context_length
        self.hoi_token_length = hoi_token_length

        # Vision
        vision_heads = vision_width // 64
        self.visual = HOIVisionTransformer(
            image_resolution=image_resolution,
            patch_size=vision_patch_size,
            width=vision_width,
            layers=vision_layers,
            heads=vision_heads,
            output_dim=embed_dim,
            hoi_token_length=hoi_token_length,
            hoi_parser_attn_mask=self.build_hoi_attention_mask(),
            enable_dec=enable_dec,
            dec_heads=dec_heads,
            dec_layers=dec_layers,
        )

        # Text
        self.transformer = Transformer(
            width=transformer_width,
            layers=transformer_layers,
            heads=transformer_heads,
            attn_mask=self.build_attention_mask()
        )
        self.vocab_size = vocab_size
        self.token_embedding = nn.Embedding(vocab_size, transformer_width)
        self.positional_embedding = nn.Parameter(torch.empty(self.context_length, transformer_width))
        self.ln_final = LayerNorm(transformer_width)

        self.text_projection = nn.Parameter(torch.empty(transformer_width, embed_dim))
        self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))

        self.prefix_length = prefix_length
        self.conjun_length = conjun_length
        self.hoi_prefix = nn.Parameter(torch.empty(prefix_length, transformer_width))
        self.hoi_conjun = nn.Parameter(torch.empty(conjun_length, transformer_width))

        self.initialize_parameters()

    def initialize_parameters(self):
        nn.init.normal_(self.token_embedding.weight, std=0.02)
        nn.init.normal_(self.positional_embedding, std=0.01)

        proj_std = (self.transformer.width ** -0.5) * ((2 * self.transformer.layers) ** -0.5)
        attn_std = self.transformer.width ** -0.5
        fc_std = (2 * self.transformer.width) ** -0.5
        for block in self.transformer.resblocks:
            nn.init.normal_(block.attn.in_proj_weight, std=attn_std)
            nn.init.normal_(block.attn.out_proj.weight, std=proj_std)
            nn.init.normal_(block.mlp.c_fc.weight, std=fc_std)
            nn.init.normal_(block.mlp.c_proj.weight, std=proj_std)

        if self.text_projection is not None:
            nn.init.normal_(self.text_projection, std=self.transformer.width ** -0.5)

        nn.init.normal_(self.hoi_prefix, std=0.01)
        nn.init.normal_(self.hoi_conjun, std=0.01)

    def build_attention_mask(self):
        # lazily create causal attention mask, with full attention between the vision tokens
        # pytorch uses additive attention mask; fill with -inf
        mask = torch.empty(self.context_length, self.context_length)
        mask.fill_(float("-inf"))
        mask.triu_(1)  # zero out the lower diagonal
        return mask

    def build_hoi_attention_mask(self):
        # lazily create causal attention mask, similar to text encoder
        # pytorch uses additive attention mask; fill with -inf
        mask = torch.empty(self.hoi_token_length + 1, self.hoi_token_length + 1)
        mask.fill_(float("-inf"))
        mask.triu_(1)  # zero out the lower diagonal
        return mask

    @property
    def dtype(self):
        return self.visual.conv1.weight.dtype

    def encode_image(self, image, image_mask=None):
        return self.visual(image.type(self.dtype), image_mask)

    def encode_text(self, text):
        # x = self.token_embedding(text).type(self.dtype)  # [batch_size, n_ctx, d_model]
        x, eot_indices = self.text_to_embedding(text)
        x = x + self.positional_embedding.type(self.dtype)
        x = x.permute(1, 0, 2)  # NLD -> LND
        x = self.transformer(x)
        x = x.permute(1, 0, 2)  # LND -> NLD
        x = self.ln_final(x).type(self.dtype)

        # x.shape = [batch_size, n_ctx, transformer.width]
        # take features from the eot embedding (eot_token is the highest number in each sequence)
        # x = x[torch.arange(x.shape[0]), text.argmax(dim=-1)] @ self.text_projection
        x = x[torch.arange(x.shape[0]), eot_indices] @ self.text_projection

        return x

    def text_to_embedding(self, text):
        """ text (List[List[Tensor]]): A list of action text tokens and object text tokens.
            [
                [action text 1, object text 1],
                [action text 2, object text 2],
                ...
                [action text n, object text n],
            ]
        """
        all_token_embeddings = []
        eot_indices = []
        for action_token, object_token in text:
            remain_length = self.context_length - self.prefix_length - self.conjun_length - len(action_token) - len(object_token)
            if remain_length < 0:
                raise RuntimeError(f"Input text is too long for context length {self.context_length}")
            eot_indices.append(self.context_length - remain_length - 1)
            padding_zeros = torch.zeros(remain_length, dtype=torch.long).to(action_token.device)
            token = torch.cat([action_token, object_token, padding_zeros])
            token_embedding = self.token_embedding(token).type(self.dtype)
            full_token_embedding = torch.cat([
                token_embedding[0:1, :], self.hoi_prefix, token_embedding[1:len(action_token), :],
                self.hoi_conjun, token_embedding[len(action_token):, :]], dim=0)
            all_token_embeddings.append(full_token_embedding)

        eot_indices = torch.as_tensor(eot_indices)
        x = torch.stack(all_token_embeddings, dim=0)  # [batch_size, n_ctx, d_model]
        return x, eot_indices

    def forward(self, image, text, image_mask=None):
        # vision encoder
        vision_outputs = self.encode_image(image, image_mask)

        # text encoder
        text_features = self.encode_text(text)

        # normalized features
        image_features = vision_outputs["image_features"]
        image_features = image_features / image_features.norm(dim=-1, keepdim=True)
        hoi_features = vision_outputs["hoi_features"]
        hoi_features = hoi_features / hoi_features.norm(dim=-1, keepdim=True)
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)

        # [image level] cosine similarity as logits
        logit_scale = self.logit_scale.exp()
        logits_per_image = logit_scale * image_features @ text_features.t()
        logits_per_text = logits_per_image.t()

        # [hoi level] cosine similarity between hoi_features and text_features
        logits_per_hoi = logit_scale * hoi_features @ text_features.t()

        return_dict = {
            "logits_per_image": logits_per_image,
            "logits_per_text": logits_per_text,
            "logits_per_hoi": logits_per_hoi,
            "pred_boxes": vision_outputs["pred_boxes"],
            "box_scores": vision_outputs["box_scores"],
            "attn_maps": vision_outputs["attn_maps"],
        }
        if "aux_outputs" in vision_outputs:
            return_dict.update({"aux_outputs": vision_outputs["aux_outputs"]})

        return return_dict


class PostProcess(object):
    """ This module converts the model's output into the format expected by the coco api"""
    def __init__(self, score_threshold):
        self.score_threshold = score_threshold

    def __call__(self, outputs, original_size, hoi_mapper):
        """ Perform the computation
        Parameters:
            outputs: raw outputs of the model
            original_size: For evaluation, this must be the original image size (before any data augmentation)
                          For visualization, this should be the image size after data augment, but before padding
            hoi_mapper: map the predicted classes to the hoi id specified by the dataset.
        """
        # Recover the bounding boxes based on the original image size
        pred_boxes = outputs['pred_boxes']
        pred_person_boxes = box_ops.box_cxcywh_to_xyxy(pred_boxes[:, :4])
        pred_object_boxes = box_ops.box_cxcywh_to_xyxy(pred_boxes[:, 4:])
        pred_person_boxes = pred_person_boxes.clamp(min=0, max=1)
        pred_object_boxes = pred_object_boxes.clamp(min=0, max=1)
        ori_h, ori_w = original_size
        pred_person_boxes[:, 0::2] = pred_person_boxes[:, 0::2] * ori_w
        pred_person_boxes[:, 1::2] = pred_person_boxes[:, 1::2] * ori_h
        pred_object_boxes[:, 0::2] = pred_object_boxes[:, 0::2] * ori_w
        pred_object_boxes[:, 1::2] = pred_object_boxes[:, 1::2] * ori_h

        hoi_scores = outputs['pred_logits'].softmax(dim=-1)
        box_scores = outputs['box_scores'].sigmoid()
        scores = hoi_scores * box_scores

        # Filter out low confident ones
        keep = torch.nonzero(scores > self.score_threshold, as_tuple=True)
        scores = scores[keep]
        classes = keep[1]
        pred_person_boxes = pred_person_boxes[keep[0]]
        pred_object_boxes = pred_object_boxes[keep[0]]

        person_keep = batched_nms(pred_person_boxes, scores, classes, 0.5)
        object_keep = batched_nms(pred_object_boxes, scores, classes, 0.5)

        person_filter_mask = torch.zeros_like(scores, dtype=torch.bool)
        object_filter_mask = torch.zeros_like(scores, dtype=torch.bool)
        person_filter_mask[person_keep] = True
        object_filter_mask[object_keep] = True
        filter_mask = torch.logical_or(person_filter_mask, object_filter_mask)

        scores = scores[filter_mask].detach().cpu().numpy().tolist()
        classes = classes[filter_mask].detach().cpu().numpy().tolist()
        pred_boxes = torch.cat([pred_person_boxes, pred_object_boxes], dim=-1)
        pred_boxes = pred_boxes[filter_mask].detach().cpu().numpy().tolist()

        results = []
        for score, hoi_id, boxes in zip(scores, classes, pred_boxes):
            results.append([hoi_mapper[int(hoi_id)], score] + boxes)

        return results


def _get_clones(module, N):
    """ Clone a moudle N times """
    return nn.ModuleList([copy.deepcopy(module) for i in range(N)])


def convert_weights(model: nn.Module):
    """Convert applicable model parameters to fp16"""

    def _convert_weights_to_fp16(l):
        if isinstance(l, (nn.Conv1d, nn.Conv2d, nn.Linear)):
            l.weight.data = l.weight.data.half()
            if l.bias is not None:
                l.bias.data = l.bias.data.half()

        if isinstance(l, nn.MultiheadAttention):
            for attr in [*[f"{s}_proj_weight" for s in ["in", "q", "k", "v"]], "in_proj_bias", "bias_k", "bias_v"]:
                tensor = getattr(l, attr)
                if tensor is not None:
                    tensor.data = tensor.data.half()

        nnParams_modules = [
            "text_projection", "proj", "hoi_prefix", "hoi_conjun", "hoi_pos_embed",
            "hoi_token_embed", "class_embedding", "positional_embedding"]
        for name in nnParams_modules:
            if hasattr(l, name):
                attr = getattr(l, name)
                if attr is not None:
                    attr.data = attr.data.half()

    model.apply(_convert_weights_to_fp16)



def build_model(args):
    ''' Build HOI detector and load pretrained CLIP weights '''
    # Build HOI detector
    model = HOIDetector(
        embed_dim=args.embed_dim,
        # vision encoder
        image_resolution=args.image_resolution, # CLIP uses fixed image resolution
        vision_layers=args.vision_layers,
        vision_width=args.vision_width,
        vision_patch_size=args.vision_patch_size,
        hoi_token_length=args.hoi_token_length,
        # bounding box head
        enable_dec=args.enable_dec,
        dec_heads=args.dec_heads,
        dec_layers=args.dec_layers,
        # text encoder
        context_length=args.context_length,
        vocab_size=args.vocab_size,
        transformer_width=args.transformer_width,
        transformer_heads=args.transformer_heads,
        transformer_layers=args.transformer_layers,
        prefix_length=args.prefix_length,
        conjun_length=args.conjun_length,
    )

    # Load pretrained CLIP weights
    if args.clip_model in _MODELS:
        model_path = _download(_MODELS[args.clip_model], os.path.expanduser("~/.cache/clip"))
        clip_model = torch.jit.load(model_path).eval()
        # Copy the pretrained CLIP parameters as the initilized weights for our newly added modules. 
        state_dict = clip_model.state_dict()
        for n, p in model.named_parameters():
            if "hoi_cross_attn" in n:
                copy_n = n.replace("hoi_cross_attn", "attn")
                state_dict.update({n: state_dict[copy_n].clone()})
        model.load_state_dict(state_dict, strict=False)

    if args.pretrained:
        checkpoint = torch.load(args.pretrained, map_location='cpu')
        model.load_state_dict(checkpoint["model"], strict=True)

    # Build matcher and criterion
    matcher = build_matcher(args)
    weight_dict = {
        'loss_ce': args.class_loss_coef, # previously, = 1
        'loss_bbox': args.bbox_loss_coef,
        'loss_giou': args.giou_loss_coef,
        'loss_conf': args.conf_loss_coef,
    }
    if args.aux_loss:
        aux_weight_dict = {}
        for i in range(args.dec_layers):
            aux_weight_dict.update({k + f'_{i}': weight_dict[k] for k in ['loss_bbox', 'loss_giou', 'loss_conf']})
            weight_dict.update(aux_weight_dict)

    losses = ['labels', 'boxes', "confidences"]
    criterion = SetCriterion(
        matcher=matcher,
        weight_dict=weight_dict,
        eos_coef=args.eos_coef,
        losses=losses
    )
    device = torch.device(args.device)
    criterion.to(device)

    # Postprocessor for inference
    postprocessors = PostProcess(args.test_score_thresh)

    return model, criterion, postprocessors