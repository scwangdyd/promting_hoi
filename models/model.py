import os
import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from typing import Tuple, Union

import utils.box_ops as box_ops
from clip.model import Transformer, HOIVisionTransformer, LayerNorm, MLP
from clip.clip import _download
from .matcher import build_matcher
from .criterion import SetCriterion


_MODELS = {
    "RN50": "https://openaipublic.azureedge.net/clip/models/afeb0e10f9e5a86da6080e35cf09123aca3b358a0c3e3b6c78a7b63bc04b6762/RN50.pt",
    "RN101": "https://openaipublic.azureedge.net/clip/models/8fa8567bab74a42d41c5915025a8e4538c3bdbe8804a470a72f30b0d94fab599/RN101.pt",
    "RN50x4": "https://openaipublic.azureedge.net/clip/models/7e526bd135e493cef0776de27d5f42653e6b4c8bf9e0f653bb11773263205fdd/RN50x4.pt",
    "RN50x16": "https://openaipublic.azureedge.net/clip/models/52378b407f34354e150460fe41077663dd5b39c54cd0bfd2b27167a4a06ec9aa/RN50x16.pt",
    "ViT-B/32": "https://openaipublic.azureedge.net/clip/models/40d365715913c9da98579312b702a82c18be219cc2a73407c4526f58eba950af/ViT-B-32.pt",
    "ViT-B/16": "https://openaipublic.azureedge.net/clip/models/5806e77cd80f8b59890b7e101eabd078d9fb84e6937f9e85e4ecb61988df416f/ViT-B-16.pt",
}


class CLIP_HOI_PROMPTER(nn.Module):
    def __init__(
        self,
        embed_dim: int,
        # vision
        image_resolution: int,
        vision_layers: Union[Tuple[int, int, int, int], int],
        vision_width: int,
        vision_patch_size: int,
        hoi_token_length: int,
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
            input_resolution=image_resolution,
            patch_size=vision_patch_size,
            width=vision_width,
            layers=vision_layers,
            heads=vision_heads,
            output_dim=embed_dim,
            hoi_token_length=hoi_token_length,
            attn_mask=self.build_hoi_attention_mask()
        )
        self.bbox_embed = MLP(embed_dim, embed_dim, 8, 3)
        self.hoi_confidence_embed = nn.Linear(embed_dim, 1)

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

        for layer in self.bbox_embed.layers:
            nn.init.normal_(layer.weight, std=0.01)
            layer.bias.data.fill_(0.01)
        
        nn.init.normal_(self.hoi_confidence_embed.weight, std=0.01)
        self.hoi_confidence_embed.bias.data.fill_(0.01)
        
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
        # lazily create causal attention mask, with full attention between the vision tokens
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
        image_features, hoi_features, bbox_features, conf_features, attn_maps = self.encode_image(image, image_mask)
        text_features = self.encode_text(text)

        # normalized features
        image_features = image_features / image_features.norm(dim=-1, keepdim=True)
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)

        # cosine similarity as logits
        logit_scale = self.logit_scale.exp()
        logits_per_image = logit_scale * image_features @ text_features.t()
        logits_per_text = logits_per_image.t()

        # person and object box regression
        boxes = self.bbox_embed(bbox_features).sigmoid()
        confidence_scores = self.hoi_confidence_embed(F.relu(conf_features))

        # cosine similarity between hoi_features and text_features
        hoi_features = hoi_features / hoi_features.norm(dim=-1, keepdim=True)
        logits_per_hoi = logit_scale * hoi_features @ text_features.t()
        # shape = [global_batch_size, global_batch_size]
        return {
            "logits_per_image": logits_per_image,
            "logits_per_text": logits_per_text,
            "pred_boxes": boxes,
            "logits_per_hoi": logits_per_hoi,
            "confidence_scores": confidence_scores,
            "attention_maps": attn_maps,
        }


class PostProcess(nn.Module):
    """ This module converts the model's output into the format expected by the coco api"""
    @torch.no_grad()
    def forward(self, outputs, target_sizes):
        """ Perform the computation
        Parameters:
            outputs: raw outputs of the model
            target_sizes: tensor of dimension [batch_size x 2] containing the size of each images of the batch
                          For evaluation, this must be the original image size (before any data augmentation)
                          For visualization, this should be the image size after data augment, but before padding
        """
        out_logits, out_bbox = outputs['pred_logits'], outputs['pred_boxes']

        assert len(out_logits) == len(target_sizes)
        assert target_sizes.shape[1] == 2

        prob = F.softmax(out_logits, -1)
        scores, labels = prob[..., :-1].max(-1)

        # convert to [x0, y0, x1, y1] format
        boxes = box_ops.box_cxcywh_to_xyxy(out_bbox)
        # and from relative [0, 1] to absolute [0, height] coordinates
        img_h, img_w = target_sizes.unbind(1)
        scale_fct = torch.stack([img_w, img_h, img_w, img_h], dim=1)
        boxes = boxes * scale_fct[:, None, :]

        results = [{'scores': s, 'labels': l, 'boxes': b} for s, l, b in zip(scores, labels, boxes)]

        return results


def build_model(args):

    device = torch.device(args.device)

    model = CLIP_HOI_PROMPTER(
        args.embed_dim,
        args.image_resolution,
        args.vision_layers,
        args.vision_width,
        args.vision_patch_size,
        args.hoi_token_length,
        args.context_length,
        args.vocab_size,
        args.transformer_width,
        args.transformer_heads,
        args.transformer_layers,
        args.prefix_length,
        args.conjun_length,
    )

    # convert_weights(model)

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
        model.load_state_dict(checkpoint["model"], strict=False)

    matcher = build_matcher(args)
    weight_dict = {'loss_ce': 1., 'loss_bbox': args.bbox_loss_coef, 'loss_confi': 1.}
    weight_dict['loss_giou'] = args.giou_loss_coef

    losses = ['labels', 'boxes', "confidences"]
    criterion = SetCriterion(
        matcher=matcher,
        weight_dict=weight_dict,
        eos_coef=args.eos_coef,
        losses=losses
    )
    criterion.to(device)

    postprocessors = PostProcess()

    return model, criterion, postprocessors