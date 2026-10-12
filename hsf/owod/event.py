"""Event-camera cues for unknown objects (step 9).

The event input of the network is the GEAR png prepared like an Ultralytics image, so channel 0 of
the tensor is `num` (event count, black = 0) and the letterbox / augmentation border is the constant
fill 114/255 on all three channels. Event activity is class-agnostic: moving or textured objects
fire more events than road and sky, without using any unknown-class label.

  event_density_map  num channel with the border zeroed + validity mask
  box_density        mean event count inside boxes (integral image), relative to the image mean
  rescore_unknown    option 1: unknown score *= clamp(relative density, max)^alpha, before NMS
  EventIgnoreLoss    option 3: v8DetectionLoss whose classification loss ignores background anchors
                     in the most event-active share q of each image (outside assigned positives), so
                     the known-only detector is not taught that event-active unknown objects are
                     background
"""
import torch
import torch.nn.functional as F
from ultralytics.utils.loss import v8DetectionLoss
from ultralytics.utils.tal import make_anchors

FILL = 114 / 255


def event_density_map(evt):
    """evt (B, 3, H, W) network input -> num map (B, H, W) with border pixels zeroed, valid mask (B, H, W)."""
    evt = evt.float()
    valid = ~((evt - FILL).abs() < 1e-3).all(1)
    return evt[:, 0] * valid, valid


def image_mean(num, valid):
    return num.sum((1, 2)) / valid.sum((1, 2)).clamp(min=1)


def box_density(num, valid, boxes):
    """num, valid (B, H, W); boxes (B, N, 4) xyxy pixels -> relative density (B, N): mean num in the
    box / mean num of the valid image area."""
    B, H, W = num.shape
    ii = F.pad(num.cumsum(1).cumsum(2), (1, 0, 1, 0))                      # (B, H+1, W+1)
    x1 = boxes[..., 0].clamp(0, W - 1).floor().long()
    y1 = boxes[..., 1].clamp(0, H - 1).floor().long()
    x2 = boxes[..., 2].clamp(1, W).ceil().long().clamp(min=x1 + 1)
    y2 = boxes[..., 3].clamp(1, H).ceil().long().clamp(min=y1 + 1)
    flat = ii.flatten(1)
    at = lambda y, x: flat.gather(1, y * (W + 1) + x)                       # noqa: E731
    s = at(y2, x2) - at(y1, x2) - at(y2, x1) + at(y1, x1)
    mean = s / ((x2 - x1) * (y2 - y1)).float()
    return mean / (image_mean(num, valid)[:, None] + 1e-4)


def rescore_unknown(y, evt, ch_unk, alpha, max_ratio=4.0):
    """y: decoded detector output (B, 4 + nc, A) with xywh pixel boxes and sigmoid scores. Returns a
    copy whose channel 4 + ch_unk is multiplied by clamp(relative event density, max_ratio)^alpha."""
    if alpha == 0:
        return y
    num, valid = event_density_map(evt)
    xy, wh = y[:, :2].transpose(1, 2), y[:, 2:4].transpose(1, 2)
    rel = box_density(num, valid, torch.cat([xy - wh / 2, xy + wh / 2], 2).float())
    y = y.clone()
    y[:, 4 + ch_unk] = (y[:, 4 + ch_unk].float() * rel.clamp(min=0, max=max_ratio) ** alpha).clamp(max=1).to(y.dtype)
    return y


class EventIgnoreLoss(v8DetectionLoss):
    """v8DetectionLoss (Ultralytics 8.3.158 __call__, unchanged except for the classification weight)."""

    def __init__(self, model, q=0.1):
        super().__init__(model)
        self.q = q
        self.ignored_share = 0.0          # last batch: share of anchors whose background loss was ignored

    def _ignore_mask(self, evt, feats, fg_mask):
        num, valid = event_density_map(evt.to(self.device, non_blocking=True))
        m = image_mean(num, valid)[:, None, None]
        dens, ok = [], []
        for f, s in zip(feats, self.stride.tolist()):
            s = int(s)
            dens.append((F.avg_pool2d(num[:, None], s) / (m[:, None] + 1e-4)).flatten(1))
            ok.append((F.avg_pool2d(valid[:, None].float(), s) > 0.5).flatten(1))
        dens, ok = torch.cat(dens, 1), torch.cat(ok, 1)                     # (B, A), anchor order of make_anchors
        thr = torch.nanquantile(torch.where(ok, dens, torch.full_like(dens, float('nan'))).float(),
                             1 - self.q, dim=1, keepdim=True)
        thr = torch.nan_to_num(thr, nan=float('inf'))
        return (dens >= thr) & ok & ~fg_mask

    def __call__(self, preds, batch):
        loss = torch.zeros(3, device=self.device)
        feats = preds[1] if isinstance(preds, tuple) else preds
        pred_distri, pred_scores = torch.cat([xi.view(feats[0].shape[0], self.no, -1) for xi in feats], 2).split(
            (self.reg_max * 4, self.nc), 1)
        pred_scores = pred_scores.permute(0, 2, 1).contiguous()
        pred_distri = pred_distri.permute(0, 2, 1).contiguous()
        dtype = pred_scores.dtype
        batch_size = pred_scores.shape[0]
        imgsz = torch.tensor(feats[0].shape[2:], device=self.device, dtype=dtype) * self.stride[0]
        anchor_points, stride_tensor = make_anchors(feats, self.stride, 0.5)
        targets = torch.cat((batch['batch_idx'].view(-1, 1), batch['cls'].view(-1, 1), batch['bboxes']), 1)
        targets = self.preprocess(targets.to(self.device), batch_size, scale_tensor=imgsz[[1, 0, 1, 0]])
        gt_labels, gt_bboxes = targets.split((1, 4), 2)
        mask_gt = gt_bboxes.sum(2, keepdim=True).gt_(0.0)
        pred_bboxes = self.bbox_decode(anchor_points, pred_distri)
        _, target_bboxes, target_scores, fg_mask, _ = self.assigner(
            pred_scores.detach().sigmoid(), (pred_bboxes.detach() * stride_tensor).type(gt_bboxes.dtype),
            anchor_points * stride_tensor, gt_labels, gt_bboxes, mask_gt)
        target_scores_sum = max(target_scores.sum(), 1)

        ignore = self._ignore_mask(batch['evt'], feats, fg_mask.bool())
        self.ignored_share = float(ignore.float().mean())
        w = (~ignore).to(dtype)[..., None]
        loss[1] = (self.bce(pred_scores, target_scores.to(dtype)) * w).sum() / target_scores_sum

        if fg_mask.sum():
            target_bboxes /= stride_tensor
            loss[0], loss[2] = self.bbox_loss(pred_distri, pred_bboxes, anchor_points, target_bboxes, target_scores,
                                              target_scores_sum, fg_mask)
        loss[0] *= self.hyp.box
        loss[1] *= self.hyp.cls
        loss[2] *= self.hyp.dfl
        return loss * batch_size, loss.detach()
