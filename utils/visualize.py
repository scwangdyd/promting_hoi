import torch
import numpy as np
import matplotlib.pyplot as plt
import utils.box_ops as box_ops
import torch.nn.functional as F
from PIL import Image, ImageDraw


def visualize_targets(images, targets):
    vis_images = images.tensors.permute(0, 2, 3, 1).detach().cpu().numpy()
    for i in range(len(vis_images)):
        img_rgb = vis_images[i]
        img_rgb = img_rgb - img_rgb.min()
        img_rgb = (img_rgb / img_rgb.max()) * 255
        img_rgb = Image.fromarray(np.uint8(img_rgb))
        
        drawing = ImageDraw.Draw(img_rgb)

        boxes = targets[i]["boxes"].detach().cpu() # cxcywh
        boxes = box_ops.box_cxcywh_to_xyxy(boxes)
        w, h = img_rgb.size
        boxes[:, 0::2] = boxes[:, 0::2] * w
        boxes[:, 1::2] = boxes[:, 1::2] * h
        boxes = boxes.numpy()
        
        for j in range(len(boxes)):
            top_left = (int(boxes[j, 0]), int(boxes[j, 1]))
            bottom_right = (int(boxes[j, 2]), int(boxes[j, 3]))
            draw_rectangle(drawing, (top_left, bottom_right), color="red", width=1)
        
        img_rgb.save(f"image{i}.jpg")


def draw_rectangle(draw, coordinates, color, width=1):
    for i in range(width):
        rect_start = (coordinates[0][0] - i, coordinates[0][1] - i)
        rect_end = (coordinates[1][0] + i, coordinates[1][1] + i)
        draw.rectangle((rect_start, rect_end), outline = color)


def visualize_preds(images, targets, outputs, indices):
    vis_images = images.tensors.permute(0, 2, 3, 1).detach().cpu().numpy()
    for i in range(len(vis_images)):
        img_rgb = vis_images[i]
        img_rgb = img_rgb - img_rgb.min()
        img_rgb = (img_rgb / img_rgb.max()) * 255
        img_gt = Image.fromarray(np.uint8(img_rgb))
        img_pd = Image.fromarray(np.uint8(img_rgb))
        
        img_id = int(targets[i]["image_id"])
        
        # visualize ground truth
        drawing = ImageDraw.Draw(img_gt)

        boxes = targets[i]["boxes"].detach().cpu() # cxcywh
        boxes = box_ops.box_cxcywh_to_xyxy(boxes)
        w, h = img_gt.size
        boxes[:, 0::2] = boxes[:, 0::2] * w
        boxes[:, 1::2] = boxes[:, 1::2] * h
        boxes = boxes.numpy()
        
        for j in range(len(boxes)):
            top_left = (int(boxes[j, 0]), int(boxes[j, 1]))
            bottom_right = (int(boxes[j, 2]), int(boxes[j, 3]))
            draw_rectangle(drawing, (top_left, bottom_right), color="red", width=1)
            
        # visualize preds
        logits_per_hoi = outputs["logits_per_hoi"][i]
        logits_per_hoi = logits_per_hoi.softmax(dim=-1)
        boxes = outputs["pred_boxes"][i].detach().cpu()
        pboxes = box_ops.box_cxcywh_to_xyxy(boxes[:, :4])
        oboxes = box_ops.box_cxcywh_to_xyxy(boxes[:, 4:])
        pboxes[:, 0::2] = pboxes[:, 0::2] * w
        pboxes[:, 1::2] = pboxes[:, 1::2] * h
        oboxes[:, 0::2] = oboxes[:, 0::2] * w
        oboxes[:, 1::2] = oboxes[:, 1::2] * h
        
        pboxes = pboxes.numpy()
        oboxes = oboxes.numpy()

        hoi_ids = indices[i][0]
        for hoi_id in hoi_ids:
            hoi_id = int(hoi_id)
            img_pd = Image.fromarray(np.uint8(img_rgb))
            drawing = ImageDraw.Draw(img_pd)
            top_left = (int(pboxes[hoi_id, 0]), int(pboxes[hoi_id, 1]))
            bottom_right = (int(pboxes[hoi_id, 2]), int(pboxes[hoi_id, 3]))
            draw_rectangle(drawing, (top_left, bottom_right), color="blue")
            
            top_left = (int(oboxes[hoi_id, 0]), int(oboxes[hoi_id, 1]))
            bottom_right = (int(oboxes[hoi_id, 2]), int(oboxes[hoi_id, 3]))
            draw_rectangle(drawing, (top_left, bottom_right), color="red")
        
            dst = Image.new('RGB', (img_gt.width, img_gt.height))
            dst.paste(img_pd, (0, 0))
            dst.save(f"./hico_figures/image_{img_id}_hoi{hoi_id}.jpg")
            
            # visualize attention maps
            attn_map = outputs["attention_maps"][i]
            attn = attn_map[hoi_id, 1:].view(1, 14, 14)
            attn = attn - attn.min()
            attn = attn / attn.max()
            attn = F.interpolate(attn.unsqueeze(0), scale_factor=16, mode="nearest")[0][0].detach().cpu().numpy()
            plt.imsave(f"./hico_figures/attn_map_{img_id}_hoi{hoi_id}.jpg", arr=attn, format="png")



# def visualize_preds(images, targets, outputs, indices):
#     vis_images = images.tensors.permute(0, 2, 3, 1).detach().cpu().numpy()
#     for i in range(len(vis_images)):
#         img_rgb = vis_images[i]
#         img_rgb = img_rgb - img_rgb.min()
#         img_rgb = (img_rgb / img_rgb.max()) * 255
#         img_gt = Image.fromarray(np.uint8(img_rgb))
#         img_pd = Image.fromarray(np.uint8(img_rgb))
        
#         # visualize ground truth
#         drawing = ImageDraw.Draw(img_gt)

#         boxes = targets[i]["boxes"].detach().cpu() # cxcywh
#         boxes = box_ops.box_cxcywh_to_xyxy(boxes)
#         w, h = img_gt.size
#         boxes[:, 0::2] = boxes[:, 0::2] * w
#         boxes[:, 1::2] = boxes[:, 1::2] * h
#         boxes = boxes.numpy()
        
#         for j in range(len(boxes)):
#             top_left = (int(boxes[j, 0]), int(boxes[j, 1]))
#             bottom_right = (int(boxes[j, 2]), int(boxes[j, 3]))
#             draw_rectangle(drawing, (top_left, bottom_right), color="red", width=1)
            
#         # visualize preds
#         logits_per_hoi = outputs["logits_per_hoi"][i]
#         logits_per_hoi = logits_per_hoi.softmax(dim=-1)
#         boxes = outputs["pred_boxes"][i].detach().cpu()
#         pboxes = box_ops.box_cxcywh_to_xyxy(boxes[:, :4])
#         oboxes = box_ops.box_cxcywh_to_xyxy(boxes[:, 4:])
#         pboxes[:, 0::2] = pboxes[:, 0::2] * w
#         pboxes[:, 1::2] = pboxes[:, 1::2] * h
#         oboxes[:, 0::2] = oboxes[:, 0::2] * w
#         oboxes[:, 1::2] = oboxes[:, 1::2] * h
        
#         pboxes = pboxes.numpy()
#         oboxes = oboxes.numpy()
        
#         drawing = ImageDraw.Draw(img_pd)
#         hoi_ids = indices[i][0]
#         for hoi_id in hoi_ids:
#             hoi_id = int(hoi_id)
#             top_left = (int(pboxes[hoi_id, 0]), int(pboxes[hoi_id, 1]))
#             bottom_right = (int(pboxes[hoi_id, 2]), int(pboxes[hoi_id, 3]))
#             draw_rectangle(drawing, (top_left, bottom_right), color="blue")
            
#             top_left = (int(oboxes[hoi_id, 0]), int(oboxes[hoi_id, 1]))
#             bottom_right = (int(oboxes[hoi_id, 2]), int(oboxes[hoi_id, 3]))
#             draw_rectangle(drawing, (top_left, bottom_right), color="red")
        
#         dst = Image.new('RGB', (img_gt.width + img_pd.width, img_gt.height))
#         dst.paste(img_gt, (0, 0))
#         dst.paste(img_pd, (img_gt.width, 0))
        
#         dst.save(f"./figures/image_{i}.jpg")
        
#         # visualize attention maps
#         attn_map = outputs["attention_maps"][i]
#         hoi_ids = indices[i][0]
#         for hoi_id in hoi_ids:
#             hoi_id = int(hoi_id)
#             attn = attn_map[hoi_id, 1:].view(1, 14, 14)
#             attn = F.interpolate(attn.unsqueeze(0), scale_factor=16, mode="nearest")[0][0].detach().cpu().numpy()
#             plt.imsave(f"./figures/attn_map{i}.jpg", arr=attn, format="png")
        
        # cx = int(targets[i]["boxes"][0][0] * 14)
        # cy = int(targets[i]["boxes"][0][1] * 14)
        # token_id = (cy - 1) * 14 + cx
        # # token_id = 0
        # attn_map = outputs["attention_maps"][i][token_id, 1:]
        # attn_map = attn_map.view(1, 14, 14)
        # attn_map = F.interpolate(attn_map.unsqueeze(0), scale_factor=16, mode="nearest")[0][0].detach().cpu().numpy()
        # plt.imsave(f"./figures/attn_map{i}.jpg", arr=attn_map, format="png")
        
        # Visualize high confident ones but not in indices from matcher
        # vis_thresh = 0.2
        # conf_scores = outputs["confidence_scores"][i].sigmoid()
        # kept_indices = torch.nonzero(conf_scores > vis_thresh, as_tuple=True)[0]
        # for hoi_id in kept_indices:
        #     if int(hoi_id) in indices[i][0]:
        #         continue
        #     hoi_id = int(hoi_id)
        #     img_pd = Image.fromarray(np.uint8(img_rgb))
        #     drawing = ImageDraw.Draw(img_pd)
        #     top_left = (int(pboxes[hoi_id, 0]), int(pboxes[hoi_id, 1]))
        #     bottom_right = (int(pboxes[hoi_id, 2]), int(pboxes[hoi_id, 3]))
        #     draw_rectangle(drawing, (top_left, bottom_right), color="blue")
            
        #     top_left = (int(oboxes[hoi_id, 0]), int(oboxes[hoi_id, 1]))
        #     bottom_right = (int(oboxes[hoi_id, 2]), int(oboxes[hoi_id, 3]))
        #     draw_rectangle(drawing, (top_left, bottom_right), color="red")
            
        #     img_pd.save(f"./figures/image_{i}_fp_{hoi_id}.jpg")