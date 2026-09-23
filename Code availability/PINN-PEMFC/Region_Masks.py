import torch

def get_mask(x,x_cl,x_gdl):
    # input: x
## Plan 1
    k = 500   ##
    x1 = x_cl  ## cl-gdl
    x2 = x_gdl ## gdl-ch

    #  Sigmoid
    # # sigma1: 0 -> 1 at CL/GDL interface
    # sigma1 = torch.sigmoid(k * (x - x1))
    # # sigma2: 0 -> 1 at GDL/CH interface
    # sigma2 = torch.sigmoid(k * (x - x2))
    #
    # # # 2.  (Masks)
    # mask_CL = 1.0 - sigma1
    # mask_GDL = sigma1 * (1.0 - sigma2)
    # mask_CH = sigma2

## Plan 2
    mask_CL = (x < x1).float()
    mask_GDL = ((x >= x1) & (x < x2)).float()
    mask_CH = (x >= x2).float()

    return mask_CL, mask_GDL, mask_CH