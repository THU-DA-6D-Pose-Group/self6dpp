import copy
import logging
import os.path as osp

cur_dir = osp.dirname(osp.abspath(__file__))
import math
import torch
from torch import nn
import mmcv
from .model_utils.common import (
    Conv,
    SPP,
    SPPCSP,
    Bottleneck,
    VoVCSP,
    Focus,
    BottleneckCSP,
    BottleneckCSP2,
    Concat,
)
from .model_utils.experimental import DWConv, MixConv2d, CrossConv, C3
from lib.torch_utils import torch_utils
from det.yolov4.yolo_utils.utils import (
    check_anchor_order,
    make_divisible,
    bbox_iou,
)
from lib.utils.utils import iprint
from core.utils.solver_utils import build_optimizer_with_params
from det.yolov4.losses.yolo_loss_utils import smooth_BCE, FocalLoss

logger = logging.getLogger(__name__)


class Detect(nn.Module):
    def __init__(self, nc=80, anchors=(), ch=()):  # detection layer
        super().__init__()
        self.stride = None  # strides computed during build
        self.nc = nc  # number of classes
        self.no = nc + 5  # number of outputs per anchor
        self.nl = len(anchors)  # number of detection layers
        self.na = len(anchors[0]) // 2  # number of anchors
        self.grid = [torch.zeros(1)] * self.nl  # init grid
        a = torch.tensor(anchors).float().view(self.nl, -1, 2)
        self.register_buffer("anchors", a)  # shape(nl,na,2)
        self.register_buffer("anchor_grid", a.clone().view(self.nl, 1, -1, 1, 1, 2))  # shape(nl,1,na,1,1,2)
        self.m = nn.ModuleList(nn.Conv2d(x, self.no * self.na, 1) for x in ch)  # output conv
        self.export = False  # onnx export

    def forward(self, x):
        # x = x.copy()  # for profiling
        z = []  # inference output
        self.training |= self.export
        for i in range(self.nl):
            x[i] = self.m[i](x[i])  # conv
            bs, _, ny, nx = x[i].shape  # x(bs,255,20,20) to x(bs,3,20,20,85)
            x[i] = x[i].view(bs, self.na, self.no, ny, nx).permute(0, 1, 3, 4, 2).contiguous()

            if not self.training:  # inference
                if self.grid[i].shape[2:4] != x[i].shape[2:4]:
                    self.grid[i] = self._make_grid(nx, ny).to(x[i].device)

                y = x[i].sigmoid()
                y[..., 0:2] = (y[..., 0:2] * 2.0 - 0.5 + self.grid[i].to(x[i].device)) * self.stride[i]  # xy
                y[..., 2:4] = (y[..., 2:4] * 2) ** 2 * self.anchor_grid[i]  # wh
                z.append(y.view(bs, -1, self.no))

        return x if self.training else (torch.cat(z, 1), x)

    @staticmethod
    def _make_grid(nx=20, ny=20):
        yv, xv = torch.meshgrid([torch.arange(ny), torch.arange(nx)])
        return torch.stack((xv, yv), 2).view((1, 1, ny, nx, 2)).float()


class Model(nn.Module):
    def __init__(self, cfg, ch=3):  # model, input channels, number of classes
        super().__init__()
        self.cfg = cfg
        # build model from yaml file --------------------------------------------
        self.yaml_file = cfg.MODEL.YOLO.MODEL_CFG
        yaml_path = osp.join(cur_dir, "model_utils", self.yaml_file)
        assert osp.exists(yaml_path), yaml_path
        self.yaml = mmcv.load(yaml_path)  # model dict
        self.num_classes = num_classes = cfg.MODEL.YOLO.NUM_CLASSES
        # Define model
        if num_classes != self.yaml["nc"]:
            logger.warning("Overriding %s nc=%g with nc=%g" % (cfg, self.yaml["nc"], num_classes))
            self.yaml["nc"] = num_classes  # override yaml value
        self.model, self.save = parse_model(copy.deepcopy(self.yaml), ch=[ch])  # model, savelist, ch_out
        # iprint([x.shape for x in self.forward(torch.zeros(1, ch, 64, 64))])

        # Build strides, anchors -----------------------------------------------
        m = self.model[-1]  # Detect()
        if isinstance(m, Detect):
            s = 128  # 2x min stride
            m.stride = torch.tensor([s / x.shape[-2] for x in self.forward(torch.zeros(1, ch, s, s))])  # forward
            m.anchors /= m.stride.view(-1, 1, 1)
            check_anchor_order(m)
            self.stride = m.stride
            self._initialize_biases()  # only run once
            # iprint('Strides: %s' % m.stride.tolist())

        # Init weights, biases
        torch_utils.initialize_weights(self)
        self.info()
        logger.info("")

    def forward(self, x, targets=None, augment=False, profile=False, do_loss=False):
        if augment:
            img_size = x.shape[-2:]  # height, width
            s = [0.83, 0.67]  # scales
            y = []
            for i, xi in enumerate(
                (
                    x,
                    torch_utils.scale_img(x.flip(3), s[0]),  # flip-lr and scale
                    torch_utils.scale_img(x, s[1]),  # scale
                )
            ):
                # cv2.imwrite('img%g.jpg' % i, 255 * xi[0].numpy().transpose((1, 2, 0))[:, :, ::-1])
                y.append(self.forward_once(xi)[0])

            y[1][..., :4] /= s[0]  # scale
            y[1][..., 0] = img_size[1] - y[1][..., 0]  # flip lr
            y[2][..., :4] /= s[1]  # scale
            # return torch.cat(y, 1), None  # augmented inference, train
            preds = torch.cat(y, 1)
        else:
            # return self.forward_once(x, profile)  # single-scale inference, train
            preds = self.forward_once(x, profile)
        if do_loss:
            assert targets is not None
            out_dict = {}
            loss_dict = self.yolo_loss(preds, targets)
            return out_dict, loss_dict
        else:
            # nms
            return preds

    def yolo_loss(self, preds, targets):
        loss_cfg = self.cfg.MODEL.YOLO.LOSS_CFG
        device = targets.device
        ft = torch.cuda.FloatTensor if preds[0].is_cuda else torch.Tensor
        lcls, lbox, lobj = (
            ft([0]).to(device),
            ft([0]).to(device),
            ft([0]).to(device),
        )
        tcls, tbox, indices, anchors = self.build_targets(preds, targets)  # targets

        # Define criteria
        BCEcls = nn.BCEWithLogitsLoss(pos_weight=ft([loss_cfg.CLS_PW]), reduction="mean").to(device)
        BCEobj = nn.BCEWithLogitsLoss(pos_weight=ft([loss_cfg.OBJ_PW]), reduction="mean").to(device)

        # class label smoothing https://arxiv.org/pdf/1902.04103.pdf eqn 3
        cp, cn = smooth_BCE(eps=0.0)

        # focal loss
        focal_gamma = loss_cfg.FOCAL_LOSS_GAMMA
        if focal_gamma > 0:  # default 0
            BCEcls = FocalLoss(BCEcls, focal_gamma)
            BCEobj = FocalLoss(BCEobj, focal_gamma)

        # per output
        n_target = 0  # number of targets
        n_pred = len(preds)  # number of outputs
        balance = [4.0, 1.0, 0.4] if n_pred == 3 else [4.0, 1.0, 0.4, 0.1]  # P3-5 or P3-6
        giou_ratio = loss_cfg.GIOU_RATIO
        for i, pi in enumerate(preds):  # layer index, layer predictions
            b, a, gj, gi = indices[i]  # image, anchor, gridy, gridx
            tobj = torch.zeros_like(pi[..., 0]).to(device)  # target obj

            nb = b.shape[0]  # number of targets
            if nb:
                n_target += nb  # cumulative targets
                ps = pi[b, a, gj, gi]  # prediction subset corresponding to targets

                # GIoU
                pxy = ps[:, :2].sigmoid() * 2.0 - 0.5
                pwh = (ps[:, 2:4].sigmoid() * 2) ** 2 * anchors[i]
                pbox = torch.cat((pxy, pwh), 1).to(device)  # predicted box
                giou = bbox_iou(pbox.t(), tbox[i], x1y1x2y2=False, GIoU=True)  # giou(prediction, target)
                lbox += (1.0 - giou).mean()  # giou loss

                # Obj

                tobj[b, a, gj, gi] = (1.0 - giou_ratio) + giou_ratio * giou.detach().clamp(0).type(
                    tobj.dtype
                )  # giou ratio

                # Class
                if self.num_classes > 1:  # cls loss (only if multiple classes)
                    t = torch.full_like(ps[:, 5:], cn).to(device)  # targets
                    t[range(nb), tcls[i]] = cp
                    lcls += BCEcls(ps[:, 5:], t)  # BCE

                # Append targets to text file
                # with open('targets.txt', 'a') as file:
                #     [file.write('%11.5g ' * 4 % tuple(x) + '\n') for x in torch.cat((txy[i], twh[i]), 1)]

            lobj += BCEobj(pi[..., 4], tobj) * balance[i]  # obj loss

        s = 3 / n_pred  # output count scaling
        lbox *= loss_cfg.GIOU_LW * s
        lobj *= loss_cfg.OBJ_LW * s * (1.4 if n_pred == 4 else 1.0)
        lcls *= loss_cfg.CLS_LW * s

        bs = tobj.shape[0]  # batch size

        # loss = lbox + lobj + lcls
        # return loss * bs, torch.cat((lbox, lobj, lcls, loss)).detach()
        loss_dict = {
            "loss_bbox": lbox * bs,
            "loss_obj": lobj * bs,
            "loss_cls": lcls * bs,
        }
        return loss_dict

    def build_targets(self, preds, targets):
        # Build targets for compute_loss(), input targets(image,class,x,y,w,h)
        # Detect() module
        net_cfg = self.cfg.MODEL.YOLO
        det = self.model[-1]
        na, nt = det.na, targets.shape[0]  # number of anchors, targets
        tcls, tbox, indices, anch = [], [], [], []
        gain = torch.ones(6, device=targets.device)  # normalized to gridspace gain
        off = torch.tensor([[1, 0], [0, 1], [-1, 0], [0, -1]], device=targets.device).float()  # overlap offsets
        at = torch.arange(na).view(na, 1).repeat(1, nt)  # anchor tensor, same as .repeat_interleave(nt)

        g = 0.5  # offset
        style = "rect4"
        for i in range(det.nl):
            anchors = det.anchors[i]
            gain[2:] = torch.tensor(preds[i].shape)[[3, 2, 3, 2]]  # xyxy gain

            # Match targets to anchors
            a, t, offsets = [], targets * gain, 0
            if nt:
                r = t[None, :, 4:6] / anchors[:, None]  # wh ratio
                j = torch.max(r, 1.0 / r).max(2)[0] < net_cfg.ANCHOR_THR  # compare
                a, t = at[j], t.repeat(na, 1, 1)[j]  # filter

                # overlaps
                gxy = t[:, 2:4]  # grid xy
                z = torch.zeros_like(gxy)
                if style == "rect2":
                    j, k = ((gxy % 1.0 < g) & (gxy > 1.0)).T
                    a, t = torch.cat((a, a[j], a[k]), 0), torch.cat((t, t[j], t[k]), 0)
                    offsets = torch.cat((z, z[j] + off[0], z[k] + off[1]), 0) * g
                elif style == "rect4":
                    j, k = ((gxy % 1.0 < g) & (gxy > 1.0)).T
                    l, m = ((gxy % 1.0 > (1 - g)) & (gxy < (gain[[2, 3]] - 1.0))).T
                    a, t = (
                        torch.cat((a, a[j], a[k], a[l], a[m]), 0),
                        torch.cat((t, t[j], t[k], t[l], t[m]), 0),
                    )
                    offsets = (
                        torch.cat(
                            (
                                z,
                                z[j] + off[0],
                                z[k] + off[1],
                                z[l] + off[2],
                                z[m] + off[3],
                            ),
                            0,
                        )
                        * g
                    )

            # Define
            b, c = t[:, :2].long().T  # image, class
            gxy = t[:, 2:4]  # grid xy
            gwh = t[:, 4:6]  # grid wh
            gij = (gxy - offsets).long()
            gi, gj = gij.T  # grid xy indices

            # Append
            indices.append((b, a, gj, gi))  # image, anchor, grid indices
            tbox.append(torch.cat((gxy - gij, gwh), 1))  # box
            anch.append(anchors[a])  # anchors
            tcls.append(c)  # class

        return tcls, tbox, indices, anch

    def forward_once(self, x, profile=False):
        y, dt = [], []  # outputs
        for m in self.model:
            if m.f != -1:  # if not from previous layer
                x = y[m.f] if isinstance(m.f, int) else [x if j == -1 else y[j] for j in m.f]  # from earlier layers

            if profile:
                try:
                    import thop

                    o = thop.profile(m, inputs=(x,), verbose=False)[0] / 1e9 * 2  # FLOPS
                except:
                    o = 0
                t = torch_utils.time_synchronized()
                for _ in range(10):
                    _ = m(x)
                dt.append((torch_utils.time_synchronized() - t) * 100)
                logger.info("%10.1f%10.0f%10.1fms %-40s" % (o, m.np, dt[-1], m.type))

            x = m(x)  # run
            y.append(x if m.i in self.save else None)  # save output

        if profile:
            logger.info("%.1fms total" % sum(dt))
        return x

    def _initialize_biases(self, cf=None):  # initialize biases into Detect(), cf is class frequency
        # cf = torch.bincount(torch.tensor(np.concatenate(dataset.labels, 0)[:, 0]).long(), minlength=nc) + 1.
        m = self.model[-1]  # Detect() module
        for mi, s in zip(m.m, m.stride):  #  from
            b = mi.bias.view(m.na, -1)  # conv.bias(255) to (3,85)
            b.data[:, 4] += math.log(8 / (640 / s) ** 2)  # obj (8 objects per 640 image)
            b.data[:, 5:] += math.log(0.6 / (m.nc - 0.99)) if cf is None else torch.log(cf / cf.sum())  # cls
            mi.bias = torch.nn.Parameter(b.view(-1), requires_grad=True)

    def _print_biases(self):
        m = self.model[-1]  # Detect() module
        for mi in m.m:  #  from
            b = mi.bias.detach().view(m.na, -1).T  # conv.bias(255) to (3,85)
            logger.info(
                ("%6g Conv2d.bias:" + "%10.3g" * 6) % (mi.weight.shape[1], *b[:5].mean(1).tolist(), b[5:].mean())
            )

    def fuse(self):  # fuse model Conv2d() + BatchNorm2d() layers
        logger.info("Fusing layers... ")
        for m in self.model.modules():
            if type(m) is Conv:
                m._non_persistent_buffers_set = set()  # pytorch 1.6.0 compatability
                m.conv = torch_utils.fuse_conv_and_bn(m.conv, m.bn)  # update conv
                m.bn = None  # remove batchnorm
                m.forward = m.fuseforward  # update forward
        self.info()
        return self

    def info(self):  # print model information
        torch_utils.model_info(self)


def parse_model(d, ch):  # model_dict, input_channels(3)
    logger.info("\n%3s%18s%3s%10s  %-40s%-30s" % ("", "from", "n", "params", "module", "arguments"))
    anchors, nc, gd, gw = (
        d["anchors"],
        d["nc"],
        d["depth_multiple"],
        d["width_multiple"],
    )
    na = (len(anchors[0]) // 2) if isinstance(anchors, list) else anchors  # number of anchors
    no = na * (nc + 5)  # number of outputs = anchors * (classes + 5)

    layers, save, c2 = [], [], ch[-1]  # layers, savelist, ch out
    for i, (f, n, m, args) in enumerate(d["backbone"] + d["head"]):  # from, number, module, args
        m = eval(m) if isinstance(m, str) else m  # eval strings
        for j, a in enumerate(args):
            try:
                args[j] = eval(a) if isinstance(a, str) else a  # eval strings
            except:
                pass

        n = max(round(n * gd), 1) if n > 1 else n  # depth gain
        if m in [
            nn.Conv2d,
            Conv,
            Bottleneck,
            SPP,
            DWConv,
            MixConv2d,
            Focus,
            CrossConv,
            BottleneckCSP,
            BottleneckCSP2,
            SPPCSP,
            VoVCSP,
            C3,
        ]:
            c1, c2 = ch[f], args[0]

            # Normal
            # if i > 0 and args[0] != no:  # channel expansion factor
            #     ex = 1.75  # exponential (default 2.0)
            #     e = math.log(c2 / ch[1]) / math.log(2)
            #     c2 = int(ch[1] * ex ** e)
            # if m != Focus:

            c2 = make_divisible(c2 * gw, 8) if c2 != no else c2

            # Experimental
            # if i > 0 and args[0] != no:  # channel expansion factor
            #     ex = 1 + gw  # exponential (default 2.0)
            #     ch1 = 32  # ch[1]
            #     e = math.log(c2 / ch1) / math.log(2)  # level 1-n
            #     c2 = int(ch1 * ex ** e)
            # if m != Focus:
            #     c2 = make_divisible(c2, 8) if c2 != no else c2

            args = [c1, c2, *args[1:]]
            if m in [BottleneckCSP, BottleneckCSP2, SPPCSP, VoVCSP, C3]:
                args.insert(2, n)
                n = 1
        elif m is nn.BatchNorm2d:
            args = [ch[f]]
        elif m is Concat:
            c2 = sum([ch[-1 if x == -1 else x + 1] for x in f])
        elif m is Detect:
            args.append([ch[x + 1] for x in f])
            if isinstance(args[1], int):  # number of anchors
                args[1] = [list(range(args[1] * 2))] * len(f)
        else:
            c2 = ch[f]

        m_ = nn.Sequential(*[m(*args) for _ in range(n)]) if n > 1 else m(*args)  # module
        t = str(m)[8:-2].replace("__main__.", "")  # module type
        np = sum([x.numel() for x in m_.parameters()])  # number params
        m_.i, m_.f, m_.type, m_.np = (
            i,
            f,
            t,
            np,
        )  # attach index, 'from' index, type, number params
        logger.info("%3s%18s%3s%10.0f  %-40s%-30s" % (i, f, n, np, t, args))  # print
        save.extend(x % i for x in ([f] if isinstance(f, int) else f) if x != -1)  # append to savelist
        layers.append(m_)
        ch.append(c2)
    return nn.Sequential(*layers), sorted(save)


def build_model_optimizer(cfg, is_test=False):
    # net_cfg = cfg.MODEL.YOLO
    model = Model(cfg)
    # get optimizer
    if is_test:
        optimizer = None
    else:
        pg0, pg1, pg2 = [], [], []  # optimizer parameter groups
        for k, v in model.named_parameters():
            if v.requires_grad:
                if ".bias" in k:
                    pg2.append(v)  # biases
                elif ".weight" in k and ".bn" not in k:
                    pg1.append(v)  # apply weight decay
                else:
                    pg0.append(v)  # all else
        optimizer = build_optimizer_with_params(cfg, pg0)
        optimizer.add_param_group({"params": pg1, "weight_decay": cfg.SOLVER.WEIGHT_DECAY})  # add pg1 with weight_decay
        optimizer.add_param_group({"params": pg2})  # add pg2 (biases)
        iprint("Optimizer groups: %g .bias, %g conv.weight, %g other" % (len(pg2), len(pg1), len(pg0)))
        del pg0, pg1, pg2

    model.to(torch.device(cfg.MODEL.DEVICE))
    return model, optimizer
