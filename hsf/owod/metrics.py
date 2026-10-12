"""Open-world detection metrics (ORE / OW-DETR / PROB protocol, as reported by YOLO-UniOW Tab. 2-3).

Per task t with known class set K (previously known PK plus current known CK) and unknown set U
(classes introduced in later tasks; their ground truth boxes are relabelled "unknown"):

  mAP (PK, CK, Both)  AP50 per known class (VOC-style metric used by OWOD papers: IoU 0.5), averaged
                      over previously-known / current-known / all known classes. Unknown predictions are
                      ignored here.
  U-Recall            recall of unknown ground truth at IoU >= 0.5 by predictions labelled "unknown"
                      (top max_det per image).
  WI                  Wilderness Impact = P_K / P_{K u U} - 1 at known recall 0.8: precision of known-class
                      predictions on known objects only vs. when unknown objects are also present
                      (a known prediction matching an unknown object is a false positive in P_{K u U}).
  A-OSE               Absolute Open-Set Error: number of unknown objects detected as a known class
                      (known-class prediction with IoU >= 0.5 to an unknown box and no known match), counted
                      over predictions with score >= ose_score.
"""
import numpy as np
import torch
from ultralytics.utils.metrics import box_iou


def _ap_voc(rec, prec):
    """All-point interpolated AP (same as Ultralytics/COCO-style 101-point is close; we use all-point)."""
    mrec = np.concatenate(([0.0], rec, [1.0]))
    mpre = np.concatenate(([1.0], prec, [0.0]))
    mpre = np.flip(np.maximum.accumulate(np.flip(mpre)))
    i = np.where(mrec[1:] != mrec[:-1])[0]
    return float(np.sum((mrec[i + 1] - mrec[i]) * mpre[i + 1]))


class OWODEvaluator:
    """Accumulate per-image predictions and ground truth, then compute OWOD metrics.

    known_prev, known_cur: lists of class ids; unknown_classes: class ids treated as unknown.
    Predictions: (N, 6) [x1, y1, x2, y2, score, label] where label is a known class id or UNK (= -1).
    Ground truth: (M, 5) [cls, x1, y1, x2, y2] with the ORIGINAL class ids.
    """
    UNK = -1

    def __init__(self, known_prev, known_cur, unknown_classes, iou_thr=0.5, ose_score=0.0, wi_recall=0.8):
        self.prev, self.cur = list(known_prev), list(known_cur)
        self.known = self.prev + self.cur
        self.unknown = set(unknown_classes)
        self.iou_thr, self.ose_score, self.wi_recall = iou_thr, ose_score, wi_recall
        self.det = {c: [] for c in self.known}        # per known class: list of (score, is_tp, hits_unknown)
        self.n_gt = {c: 0 for c in self.known}
        self.unk_gt, self.unk_hit = 0, 0
        self.topk = (10, 30)
        self.unk_hit_k = {k: 0 for k in self.topk}     # U-Recall@k: only the k best unknown predictions
        self.aose = 0

    def update(self, pred, gt):
        pred = pred.detach().cpu().float()
        gt = gt.detach().cpu().float()
        gcls = gt[:, 0].long()
        is_unk_gt = torch.tensor([int(c) in self.unknown for c in gcls.tolist()], dtype=torch.bool)
        known_gt, unk_gt = gt[~is_unk_gt], gt[is_unk_gt]
        for c in self.known:
            self.n_gt[c] += int((known_gt[:, 0] == c).sum())

        # --- unknown recall: unknown predictions vs unknown ground truth
        self.unk_gt += len(unk_gt)
        up = pred[pred[:, 5] == self.UNK]
        if len(unk_gt) and len(up):
            iou = box_iou(unk_gt[:, 1:5], up[:, :4])
            self.unk_hit += int((iou.max(1).values >= self.iou_thr).sum())
            order = up[:, 4].argsort(descending=True)
            for k in self.topk:
                self.unk_hit_k[k] += int((iou[:, order[:k]].max(1).values >= self.iou_thr).sum())

        # --- known classes: greedy matching per class (VOC), plus "hits an unknown object" flag for WI/A-OSE
        unk_matched = torch.zeros(len(unk_gt), dtype=torch.bool)
        for c in self.known:
            p = pred[pred[:, 5] == c]
            if not len(p):
                continue
            p = p[p[:, 4].argsort(descending=True)]
            g = known_gt[known_gt[:, 0] == c]
            used = torch.zeros(len(g), dtype=torch.bool)
            iou_k = box_iou(p[:, :4], g[:, 1:5]) if len(g) else torch.zeros((len(p), 0))
            iou_u = box_iou(p[:, :4], unk_gt[:, 1:5]) if len(unk_gt) else torch.zeros((len(p), 0))
            for j in range(len(p)):
                tp = False
                if iou_k.shape[1]:
                    best = iou_k[j].clone()
                    best[used] = -1
                    k = int(best.argmax())
                    if best[k] >= self.iou_thr:
                        used[k], tp = True, True
                hits_u = bool(iou_u.shape[1]) and bool((iou_u[j] >= self.iou_thr).any())
                self.det[c].append((float(p[j, 4]), tp, hits_u and not tp))
                if hits_u and not tp and p[j, 4] >= self.ose_score:
                    k = int(iou_u[j].argmax())
                    if not unk_matched[k]:          # count each unknown object once
                        unk_matched[k] = True
                        self.aose += 1

    def _class_pr(self, c, count_unknown_as_fp):
        d = sorted(self.det[c], key=lambda x: -x[0])
        if not d or self.n_gt[c] == 0:
            return None, None
        tp = np.array([x[1] for x in d], float)
        fp = 1 - tp
        if not count_unknown_as_fp:          # P_K: predictions on unknown objects are ignored, not FP
            keep = np.array([not x[2] for x in d])
            tp, fp = tp[keep], fp[keep]
        ctp, cfp = np.cumsum(tp), np.cumsum(fp)
        return ctp / self.n_gt[c], ctp / np.maximum(ctp + cfp, 1e-9)

    def summarize(self):
        ap = {}
        for c in self.known:
            rec, prec = self._class_pr(c, True)
            ap[c] = _ap_voc(rec, prec) if rec is not None else 0.0
        mean = lambda cs: float(np.mean([ap[c] for c in cs])) * 100 if cs else float('nan')   # noqa: E731

        # Wilderness Impact at known recall wi_recall (pooled over known classes, as in ORE)
        def precision_at_recall(count_unknown):
            all_d = sorted([(s, tp, hu) for c in self.known for (s, tp, hu) in self.det[c]], key=lambda x: -x[0])
            n_gt = sum(self.n_gt.values())
            if not all_d or n_gt == 0:
                return float('nan')
            tp = np.array([x[1] for x in all_d], float)
            keep = np.ones(len(all_d), bool) if count_unknown else np.array([not x[2] for x in all_d])
            tp_k, fp_k = tp[keep], 1 - tp[keep]
            rec = np.cumsum(tp_k) / n_gt
            prec = np.cumsum(tp_k) / np.maximum(np.cumsum(tp_k) + np.cumsum(fp_k), 1e-9)
            i = np.searchsorted(rec, self.wi_recall)
            return float(prec[min(i, len(prec) - 1)])

        p_k, p_ku = precision_at_recall(False), precision_at_recall(True)
        wi = p_k / p_ku - 1 if p_ku and not np.isnan(p_ku) else float('nan')
        return {
            'mAP_PK': mean(self.prev), 'mAP_CK': mean(self.cur), 'mAP_Both': mean(self.known),
            'U-Recall': 100.0 * self.unk_hit / self.unk_gt if self.unk_gt else float('nan'),
            **{f'U-Recall@{k}': 100.0 * v / self.unk_gt if self.unk_gt else float('nan')
               for k, v in self.unk_hit_k.items()},
            'WI': wi, 'A-OSE': self.aose,
            'AP50_per_class': {c: round(ap[c] * 100, 2) for c in self.known},
            'n_unknown_gt': self.unk_gt,
        }
