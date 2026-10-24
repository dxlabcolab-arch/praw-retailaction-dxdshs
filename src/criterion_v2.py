"""Hungarian matching + losses, mirroring the paper's cost design.

Cost = classification error + temporal IoU + summed spatial L2 over both views.
Losses = CE(class) + L1(start,end) + L2(points per view).
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.optimize import linear_sum_assignment


def temporal_iou_matrix(ps, pe, gs, ge):
    inter = (torch.min(pe[:, None], ge[None]) - torch.max(ps[:, None], gs[None])).clamp(min=0)
    union = (torch.max(pe[:, None], ge[None]) - torch.min(ps[:, None], gs[None])).clamp(min=1e-6)
    return inter / union


class SetCriterion(nn.Module):
    def __init__(self, num_classes=3, w_cls=1.0, w_span=1.0, w_point=10.0,
                 w_iou=8.0, w_anchor=2.0, w_qual=2.0, eos_coef=0.1,
                 c_cls=1.0, c_span=1.0, c_point=10.0, c_iou=6.0):
        super().__init__()
        self.num_classes = num_classes
        self.w = dict(cls=w_cls, span=w_span, point=w_point, iou=w_iou,
                      anchor=w_anchor, qual=w_qual)
        self.c = dict(cls=c_cls, span=c_span, point=c_point, iou=c_iou)
        w = torch.ones(num_classes + 1)
        w[-1] = eos_coef            # down-weight the "no action" class
        self.register_buffer("cls_weight", w)

    @torch.no_grad()
    def match(self, out, targets):
        B, Q = out["logits"].shape[:2]
        idxs = []
        prob = out["logits"].softmax(-1)
        for b in range(B):
            t = targets[b]
            n = t["labels"].numel()
            if n == 0:
                idxs.append((torch.empty(0, dtype=torch.long),
                             torch.empty(0, dtype=torch.long)))
                continue
            dev = out["logits"].device
            gl = t["labels"].to(dev)
            gs, ge = t["spans"][:, 0].to(dev), t["spans"][:, 1].to(dev)
            gp = t["points"].to(dev)

            c_cls = -prob[b][:, gl]                                  # (Q,n)
            c_span = torch.cdist(
                torch.stack([out["start"][b], out["end"][b]], -1),
                torch.stack([gs, ge], -1), p=1)
            iou = temporal_iou_matrix(out["start"][b], out["end"][b], gs, ge)
            # spatial cost = SUM of per-view distances (paper)
            p = out["points"][b]
            c_pt = (torch.cdist(p[:, :2], gp[:, :2]) + torch.cdist(p[:, 2:], gp[:, 2:]))

            C = (self.c["cls"] * c_cls + self.c["span"] * c_span
                 + self.c["point"] * c_pt - self.c["iou"] * iou)
            r, cix = linear_sum_assignment(C.float().cpu().numpy())
            idxs.append((torch.as_tensor(r, dtype=torch.long),
                         torch.as_tensor(cix, dtype=torch.long)))
        return idxs

    def forward(self, out, targets):
        idxs = self.match(out, targets)
        dev = out["logits"].device
        B, Q = out["logits"].shape[:2]

        tgt_cls = torch.full((B, Q), self.num_classes, dtype=torch.long, device=dev)
        for b, (qi, gi) in enumerate(idxs):
            if qi.numel():
                tgt_cls[b, qi] = targets[b]["labels"].to(dev)[gi]
        loss_cls = F.cross_entropy(
            out["logits"].flatten(0, 1), tgt_cls.flatten(),
            weight=self.cls_weight.to(dev))

        ls = torch.zeros((), device=dev)
        lp = torch.zeros((), device=dev)
        liou = torch.zeros((), device=dev)
        lanc = torch.zeros((), device=dev)
        lqual = torch.zeros((), device=dev)
        n_tot = sum(int(qi.numel()) for qi, _ in idxs)
        if n_tot:
            for b, (qi, gi) in enumerate(idxs):
                if not qi.numel():
                    continue
                qi, gi = qi.to(dev), gi.to(dev)
                gs = targets[b]["spans"].to(dev)[gi]
                gp = targets[b]["points"].to(dev)[gi]
                ps = torch.stack([out["start"][b][qi], out["end"][b][qi]], -1)
                ls = ls + F.l1_loss(ps, gs, reduction="sum")
                pp = out["points"][b][qi]
                # L2 per view, summed (paper: "sum of the L2 loss ... for each view")
                lp = lp + (torch.norm(pp[:, :2] - gp[:, :2], dim=-1).sum()
                           + torch.norm(pp[:, 2:] - gp[:, 2:], dim=-1).sum())
                iou = temporal_iou_matrix(ps[:, 0], ps[:, 1], gs[:, 0], gs[:, 1]).diagonal()
                liou = liou + (1 - iou).sum()
                if "quality" in out:
                    with torch.no_grad():
                        # target = temporal IoU x spatial closeness of the match
                        dv = torch.minimum(
                            torch.norm(pp[:, :2] - gp[:, :2], dim=-1),
                            torch.norm(pp[:, 2:] - gp[:, 2:], dim=-1))
                        q = iou.clamp(0, 1) * torch.exp(-dv / 0.05)
                    lqual = lqual + F.binary_cross_entropy_with_logits(
                        out["quality"][b][qi].squeeze(-1), q, reduction="sum")
                if "anchor" in out:
                    pa = out["anchor"][b][qi]
                    lanc = lanc + (
                        F.smooth_l1_loss(pa[:, :2], gp[:, :2], beta=0.1, reduction="sum")
                        + F.smooth_l1_loss(pa[:, 2:], gp[:, 2:], beta=0.1, reduction="sum"))
            ls, lp = ls / n_tot, lp / n_tot
            liou, lanc, lqual = liou / n_tot, lanc / n_tot, lqual / n_tot

        loss = (self.w["cls"] * loss_cls + self.w["span"] * ls
                + self.w["point"] * lp + self.w["iou"] * liou
                + self.w["anchor"] * lanc + self.w["qual"] * lqual)
        return loss, {"cls": loss_cls.detach(), "span": ls.detach(),
                      "point": lp.detach(), "iou": liou.detach(),
                      "anchor": lanc.detach(), "qual": lqual.detach(),
                      "total": loss.detach()}
