import sys
sys.path.insert(1, 'r2d2')
import os, pdb
from PIL import Image
import numpy as np
import torch
import glob
from tools import common
from tools.dataloader import norm_RGB
from nets.patchnet import *
import time
from skimage.util.shape import view_as_windows
from scipy.optimize import least_squares
import copy

import numpy as np
import cv2
import torch
import glob
import pickle
import matplotlib.pyplot as plt
import logging
logger = logging.getLogger('module_R2D2')
logger.setLevel(logging.DEBUG)

#For debug
def breakpoint():
    inp = input("Waiting for input...")

def mnn_matcher(descriptors_a, descriptors_b, threshold = 0.9):
    device = descriptors_a.device
    sim = descriptors_a @ descriptors_b.t()
    nn_sim, nn12 = torch.max(sim, dim=1)
    nn21 = torch.max(sim, dim=0)[1]
    ids1 = torch.arange(0, sim.shape[0], device=device)
    mask = ((nn_sim >= threshold) & (ids1 == nn21[nn12]))
    matches = torch.stack([ids1[mask], nn12[mask]])
    return matches.t().data.cpu().numpy()


def similarity_matcher(descriptors1, descriptors2, threshold=0.9):
    # Similarity threshold matcher for L2 normalized descriptors.
    device = descriptors1.device
    
    sim = descriptors1 @ descriptors2.t()
    nn_sim, nn12 = torch.max(sim, dim=1)
    nn_dist = torch.sqrt(2 - 2 * nn_sim)
    nn21 = torch.max(sim, dim=0)[1]
    ids1 = torch.arange(0, sim.shape[0], device=device)
    mask = (nn_sim >= threshold)
    matches = torch.stack([ids1[mask], nn12[mask]])
    return matches.t(), nn_dist[mask]

def ratio_mutual_nn_matcher(descriptors1, descriptors2, ratio=0.90):
    # Lowe's ratio matcher + mutual NN for L2 normalized descriptors.
    device = descriptors1.device
    sim = descriptors1 @ descriptors2.t()
    nns_sim, nns = torch.topk(sim, 2, dim=1)
    nn12 = nns[:, 0]
    nns_dist = torch.sqrt(2 - 2 * nns_sim)
    nn21 = torch.max(sim, dim=0)[1]
    ids1 = torch.arange(0, sim.shape[0], device=device)
    matches = torch.stack([ids1, nns[:, 0]])
    ratios = nns_dist[:, 0] / (nns_dist[:, 1] + 1e-8)
    mask = torch.min(ids1 == nn21[nn12], ratios <= ratio)
    matches = matches[:, mask]
    return matches.t().data.cpu().numpy(), nns_dist[mask, 0]

def load_network(model_fn): 
    checkpoint = torch.load(model_fn)
    print("\n>> Creating net = " + checkpoint['net']) 
    net = eval(checkpoint['net'])
    nb_of_weights = common.model_size(net)
    print(f" ( Model size: {nb_of_weights/1000:.0f}K parameters )")

    # initialization
    weights = checkpoint['state_dict']
    net.load_state_dict({k.replace('module.',''):v for k,v in weights.items()})
    return net.eval()


class NonMaxSuppression (torch.nn.Module):
    def __init__(self, rel_thr=0.7, rep_thr=0.7):
        nn.Module.__init__(self)
        self.max_filter = torch.nn.MaxPool2d(kernel_size=3, stride=1, padding=1)
        self.rel_thr = rel_thr
        self.rep_thr = rep_thr
    
    def forward(self, reliability, repeatability, **kw):
        assert len(reliability) == len(repeatability) == 1
        reliability, repeatability = reliability[0], repeatability[0]

        # local maxima
        maxima = (repeatability == self.max_filter(repeatability))

        # remove low peaks
        maxima *= (repeatability >= self.rep_thr)
        maxima *= (reliability   >= self.rel_thr)

        return maxima.nonzero().t()[2:4]


def extract_multiscale( net, img, detector, scale_f=2**0.25, 
                        min_scale=0.0, max_scale=1, 
                        min_size=256, max_size=1280, 
                        verbose=False):
    old_bm = torch.backends.cudnn.benchmark 
    torch.backends.cudnn.benchmark = False # speedup
    
    # extract keypoints at multiple scales
    B, three, H, W = img.shape
    assert B == 1 and three == 3, "should be a batch with a single RGB image"
    
    assert max_scale <= 1
    s = 1.0 # current scale factor
    
    X,Y,S,C,Q,D = [],[],[],[],[],[]
    while  s+0.001 >= max(min_scale, min_size / max(H,W)):
        if s-0.001 <= min(max_scale, max_size / max(H,W)):
            nh, nw = img.shape[2:]
            if verbose: print(f"extracting at scale x{s:.02f} = {nw:4d}x{nh:3d}")
            # extract descriptors
            with torch.no_grad():
                res = net(imgs=[img])
            
            # logger.info("type(res):" + str(type(res)))
            # breakpoint()
                
            # get output and reliability map
            descriptors = res['descriptors'][0]
            reliability = res['reliability'][0]
            repeatability = res['repeatability'][0]

            # normalize the reliability for nms
            # extract maxima and descs
            y,x = detector(**res) # nms
            c = reliability[0,0,y,x]
            q = repeatability[0,0,y,x]
            d = descriptors[0,:,y,x].t()
            n = d.shape[0]

            # accumulate multiple scales
            X.append(x.float() * W/nw)
            Y.append(y.float() * H/nh)
            S.append((32/s) * torch.ones(n, dtype=torch.float32, device=d.device))
            C.append(c)
            Q.append(q)
            D.append(d)
        s /= scale_f

        # down-scale the image for next iteration
        break
        nh, nw = round(H*s), round(W*s)
        img = F.interpolate(img, (nh,nw), mode='bilinear', align_corners=False)

    # restore value
    torch.backends.cudnn.benchmark = old_bm

    Y = torch.cat(Y)
    X = torch.cat(X)
    S = torch.cat(S) # scale
    scores = torch.cat(C) * torch.cat(Q) # scores = reliability * repeatability
    XYS = torch.stack([X,Y,S], dim=-1)
    D = torch.cat(D)
    return XYS, D, scores


def extract_keypoints(img, args):
    t1 = time.time()

        
    xys, desc, scores = extract_multiscale(net, img, detector,
        scale_f   = args['scale_f'], 
        min_scale = args['min_scale'], 
        max_scale = args['max_scale'],
        min_size  = args['min_size'], 
        max_size  = args['max_size'], 
        verbose = False)

    xys = xys.cpu().numpy()
    scores = scores.cpu().numpy()
    idxs = np.argwhere(scores>0.85)

    return (xys[idxs], desc[idxs])
    

args = {'model' : 'r2d2/models/r2d2_WAF_N16.pt', 'scale_f' : 2**0.25, 'min_size' : 256, 'max_size' : 1380, 'min_scale' : 0, 'max_scale' : 1, 'reliability_thr' : 0.7, 'repeatability_thr' : 0.7 , 'gpu' : [0]}
net = load_network(args['model'])
iscuda = common.torch_set_gpu(args['gpu'])
if iscuda: net = net.cuda()
detector = NonMaxSuppression( rel_thr = args['reliability_thr'], rep_thr = args['repeatability_thr'])


def extract_features_and_desc(image):
    '''
    image: np.uint8
    '''
    img_pil = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    img_pil = Image.fromarray(img_pil)
    img_cpu = img_pil
    # print("type(img_cpu):",type(img_cpu))
    img = norm_RGB(img_cpu)[None]
    if iscuda: 
      img = img.cuda()
    kps, desc = extract_keypoints(img, args)
    # alldesc = np.transpose(alldesc, (1, 2,0))

    return np.squeeze(kps), np.squeeze(desc)

def get_matches(ref_kp, ref_desc, cur_kp, cur_desc, imgshape):
    matches = ratio_mutual_nn_matcher(ref_desc, cur_desc)[0]

    return matches

