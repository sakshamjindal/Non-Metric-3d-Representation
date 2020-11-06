import os
import numpy as np
from os.path import isfile
import torch
import torch.nn.functional as F
from imageio import imwrite

import ipdb 
st = ipdb.set_trace
EPS = 1e-6


##########################################################
####				 utils_basic					#####
##########################################################

def assert_same_shape(t1, t2):
    for (x, y) in zip(list(t1.shape), list(t2.shape)):
        assert(x==y)

def print_stats_py(name, tensor):
    print('%s (%s) min = %.2f, mean = %.2f, max = %.2f' % (name, tensor.dtype, np.min(tensor), np.mean(tensor), np.max(tensor)))

def tensor2summ(tensor, permute_dim=False):
    # if permute_dim = True: 
    # for 2D tensor, assume input is torch format B x S x C x H x W, we want B x S x H x W x C
    # for 3D tensor, assume input is torch format B x S x C x H x W x D, we want B x S x H x W x C x D
    # and finally unbind the sequeence dimension and return a list of [B x H x W x C].
    assert(tensor.ndim == 5 or tensor.ndim == 6)
    assert(tensor.size()[1] == 2) #sequense length should be 2
    if permute_dim:
        if tensor.ndim == 6: #3D tensor
            tensor = tensor.permute(0, 1, 3, 4, 5, 2)
        elif tensor.ndim == 5: #2D tensor
            tensor = tensor.permute(0, 1, 3, 4, 2)

    tensor = torch.unbind(tensor, dim=1)
    return tensor


def save_rgbs(rgbs,classes,main_folder="dump"):
    mkdir(main_folder)
    for index, rgb in enumerate(rgbs):
        imsave(f'{main_folder}/{classes[index]}_rgb.png',rgb.permute(1,2,0).cpu().numpy())
def normalize_single(d):
    # d is a whatever shape torch tensor
    dmin = torch.min(d)
    dmax = torch.max(d)
    d = (d-dmin)/(EPS+(dmax-dmin))
    return d


def mkdir(path):
    if not os.path.exists(path):
        os.makedirs(path)


def normalize(d):
    # d is B x whatever. normalize within each element of the batch
    out = torch.zeros(d.size())
    if d.is_cuda:
        out = out.cuda()
    B = list(d.size())[0]
    for b in range(B):
        out[b] = normalize_single(d[b])
    return out

def reduce_masked_mean(x, mask, dim=None, keepdim=False):
    # x and mask are the same shape
    # returns shape-1
    # axis can be a list of axes
    # st()
    assert(x.size() == mask.size())
    prod = x*mask
    if dim is None:
        numer = torch.sum(prod)
        denom = EPS+torch.sum(mask)
    
    else:
        numer = torch.sum(prod, dim=dim, keepdim=keepdim)
        denom = EPS+torch.sum(mask, dim=dim, keepdim=keepdim)
        
    mean = numer/denom
    return mean

# def save_rgbs(tensors):
#     # assume it to be B,C,H,W
#     for index,tensor in enumerate(tensors):
#         img = tensor.permute(1,2,0).detach().cpu().numpy()
#         imwrite(f"dump/{index}.png",img)

# def save_rgbs_np(tensors):
#     # assume it to be B,C,H,W
#     for index,tensor in enumerate(tensors):
#         img = tensor
#         imwrite(f"dump/{index}.png",img)

def most_frequent(List): 
    counter = 0
    num = List[0] 
    for i in List: 
        curr_frequency = List.count(i) 
        if(curr_frequency> counter): 
            counter = curr_frequency 
            num = i 
    return num

def pack_seqdim(tensor, B):
    shapelist = list(tensor.shape)
    B_, S = shapelist[:2]
    assert(B==B_)
    otherdims = shapelist[2:]
    tensor = torch.reshape(tensor, [B*S]+otherdims)
    return tensor

def unpack_seqdim(tensor, B):
    shapelist = list(tensor.shape)
    BS = shapelist[0]
    assert(BS%B==0)
    otherdims = shapelist[1:]
    S = int(BS/B)
    tensor = torch.reshape(tensor, [B,S]+otherdims)
    return tensor


def pack_boxdim(tensor, N):
    shapelist = list(tensor.shape)
    B, N_, C = shapelist[:3]
    assert(N==N_)
    # assert(C==8)
    otherdims = shapelist[3:]
    tensor = torch.reshape(tensor, [B,N*C]+otherdims)
    return tensor



def unpack_boxdim(tensor, N):
    shapelist = list(tensor.shape)
    B,NS = shapelist[:2]
    assert(NS%N==0)
    otherdims = shapelist[2:]
    S = int(NS/N)
    tensor = torch.reshape(tensor, [B,N,S]+otherdims)
    return tensor



def pack_boxbatchdim(tensor, N):
    shapelist = list(tensor.shape)
    B, N_, S = shapelist[:3]
    assert(N==N_)
    # assert(C==8)
    otherdims = shapelist[3:]
    tensor = torch.reshape(tensor, [B*N,S]+otherdims)
    return tensor



def unpack_boxbatchdim(tensor, N):
    shapelist = list(tensor.shape)
    BN,S = shapelist[:2]
    assert(BN%N==0)
    otherdims = shapelist[2:]
    B = int(BN/N)
    tensor = torch.reshape(tensor, [B,N,S]+otherdims)
    return tensor


def pack_boxbatchviewdim(tensor,B,N):
    shapelist = list(tensor.shape)
    B, N_, S = shapelist[:3]
    assert(N==N_)
    # assert(C==8)
    otherdims = shapelist[3:]
    tensor = torch.reshape(tensor, [B*N*S]+otherdims)
    return tensor


def pack_boxbatchviewdim_box(tensor,B,N):
    shapelist = list(tensor.shape)
    B, S,N_, = shapelist[:3]
    assert(N==N_)
    otherdims = shapelist[3:]
    tensor = torch.reshape(tensor, [B*N*S]+otherdims)
    return tensor



def unpack_boxbatchviewdim(tensor,B,N):
    shapelist = list(tensor.shape)
    BNS = shapelist[0]
    assert(BNS%(N*B)==0)
    otherdims = shapelist[1:]
    S = int(BNS/(B*N))
    tensor = torch.reshape(tensor, [B,N,S]+otherdims)
    return tensor



def gridcloud3D(B, Z, Y, X, norm=False):
    # we want to sample for each location in the grid
    grid_z, grid_y, grid_x = meshgrid3D(B, Z, Y, X, norm=norm)
    x = torch.reshape(grid_x, [B, -1])
    y = torch.reshape(grid_y, [B, -1])
    z = torch.reshape(grid_z, [B, -1])
    # these are B x N
    xyz = torch.stack([x, y, z], dim=2)
    # this is B x N x 3
    return xyz

def gridcloud3D_py(Z, Y, X):
    # we want to sample for each location in the grid
    grid_z, grid_y, grid_x = meshgrid3D_py(Z, Y, X)
    x = np.reshape(grid_x, [-1])
    y = np.reshape(grid_y, [-1])
    z = np.reshape(grid_z, [-1])
    # these are N
    xyz = np.stack([x, y, z], axis=1)
    # this is N x 3
    return xyz

def meshgrid2D_py(Y, X):
    grid_y = np.linspace(0.0, Y-1, Y)
    grid_y = np.reshape(grid_y, [Y, 1])
    grid_y = np.tile(grid_y, [1, X])

    grid_x = np.linspace(0.0, X-1, X)
    grid_x = np.reshape(grid_x, [1, X])
    grid_x = np.tile(grid_x, [Y, 1])

    return grid_y, grid_x

def gridcloud2D_py(Y, X):
    # we want to sample for each location in the grid
    grid_y, grid_x = meshgrid2D_py(Y, X)
    x = np.reshape(grid_x, [-1])
    y = np.reshape(grid_y, [-1])
    # these are N
    xy = np.stack([x, y], axis=1)
    # this is N x 2
    return xy

def normalize_grid3D(grid_z, grid_y, grid_x, Z, Y, X, clamp_extreme=True):
    # make things in [-1,1]
    grid_z = 2.0*(grid_z / float(Z-1)) - 1.0
    grid_y = 2.0*(grid_y / float(Y-1)) - 1.0
    grid_x = 2.0*(grid_x / float(X-1)) - 1.0
    
    if clamp_extreme:
        grid_z = torch.clamp(grid_z, min=-2.0, max=2.0)
        grid_y = torch.clamp(grid_y, min=-2.0, max=2.0)
        grid_x = torch.clamp(grid_x, min=-2.0, max=2.0)
    
    return grid_z, grid_y, grid_x

def normalize_grid2D(grid_y, grid_x, Y, X, clamp_extreme=True):
    # make things in [-1,1]
    grid_y = 2.0*(grid_y / float(Y-1)) - 1.0
    grid_x = 2.0*(grid_x / float(X-1)) - 1.0
    
    if clamp_extreme:
        grid_y = torch.clamp(grid_y, min=-2.0, max=2.0)
        grid_x = torch.clamp(grid_x, min=-2.0, max=2.0)
        
    return grid_y, grid_x

def normalize_gridcloud(xyz, Z, Y, X, clamp_extreme=True):
    # make things in [-1,1]
    x = xyz[...,0]
    y = xyz[...,1]
    z = xyz[...,2]
    
    z = 2.0*(z / float(Z-1)) - 1.0
    y = 2.0*(y / float(Y-1)) - 1.0
    x = 2.0*(x / float(X-1)) - 1.0

    xyz = torch.stack([x,y,z], dim=-1)
    
    if clamp_extreme:
        xyz = torch.clamp(xyz, min=-2.0, max=2.0)
    return xyz

def meshgrid3D_yxz(B, Y, X, Z):
    # returns a meshgrid sized B x Y x X x Z
    # this ordering makes sense since usually Y=height, X=width, Z=depth

	grid_y = torch.linspace(0.0, Y-1, Y)
	grid_y = torch.reshape(grid_y, [1, Y, 1, 1])
	grid_y = grid_y.repeat(B, 1, X, Z)
	
	grid_x = torch.linspace(0.0, X-1, X)
	grid_x = torch.reshape(grid_x, [1, 1, X, 1])
	grid_x = grid_x.repeat(B, Y, 1, Z)

	grid_z = torch.linspace(0.0, Z-1, Z)
	grid_z = torch.reshape(grid_z, [1, 1, 1, Z])
	grid_z = grid_z.repeat(B, Y, X, 1)
	
	return grid_y, grid_x, grid_z

def meshgrid2D(B, Y, X, stack=False, norm=False):
    # returns a meshgrid sized B x Y x X
    grid_y = torch.linspace(0.0, Y-1, Y, device=torch.device('cuda'))
    grid_y = torch.reshape(grid_y, [1, Y, 1])
    grid_y = grid_y.repeat(B, 1, X)

    grid_x = torch.linspace(0.0, X-1, X, device=torch.device('cuda'))
    grid_x = torch.reshape(grid_x, [1, 1, X])
    grid_x = grid_x.repeat(B, Y, 1)

    if norm:
        grid_y, grid_x = normalize_grid2D(
            grid_y, grid_x, Y, X)

    if stack:
        # note we stack in xy order
        # (see https://pytorch.org/docs/stable/nn.functional.html#torch.nn.functional.grid_sample)
        grid = torch.stack([grid_x, grid_y], dim=-1)
        return grid
    else:
        return grid_y, grid_x


def meshgrid2D_cpu(B, Y, X, stack=False, norm=False):
    # returns a meshgrid sized B x Y x X
    grid_y = torch.linspace(0.0, Y-1, Y)
    grid_y = torch.reshape(grid_y, [1, Y, 1])
    grid_y = grid_y.repeat(B, 1, X)

    grid_x = torch.linspace(0.0, X-1, X)
    grid_x = torch.reshape(grid_x, [1, 1, X])
    grid_x = grid_x.repeat(B, Y, 1)

    if norm:
        grid_y, grid_x = normalize_grid2D(
            grid_y, grid_x, Y, X)

    if stack:
        # note we stack in xy order
        # (see https://pytorch.org/docs/stable/nn.functional.html#torch.nn.functional.grid_sample)
        grid = torch.stack([grid_x, grid_y], dim=-1)
        return grid
    else:
        return grid_y, grid_x

    
def meshgrid3D(B, Z, Y, X, stack=False, norm=False):
    # returns a meshgrid sized B x Z x Y x X

    grid_z = torch.linspace(0.0, Z-1, Z, device=torch.device('cuda'))
    grid_z = torch.reshape(grid_z, [1, Z, 1, 1])
    grid_z = grid_z.repeat(B, 1, Y, X)

    grid_y = torch.linspace(0.0, Y-1, Y, device=torch.device('cuda'))
    grid_y = torch.reshape(grid_y, [1, 1, Y, 1])
    grid_y = grid_y.repeat(B, Z, 1, X)

    grid_x = torch.linspace(0.0, X-1, X, device=torch.device('cuda'))
    grid_x = torch.reshape(grid_x, [1, 1, 1, X])
    grid_x = grid_x.repeat(B, Z, Y, 1)

    if norm:
        grid_z, grid_y, grid_x = normalize_grid3D(
            grid_z, grid_y, grid_x, Z, Y, X)

    if stack:
        # note we stack in xyz order
        # (see https://pytorch.org/docs/stable/nn.functional.html#torch.nn.functional.grid_sample)
        grid = torch.stack([grid_x, grid_y, grid_z], dim=-1)
        return grid
    else:
        return grid_z, grid_y, grid_x

def meshgrid3D_py(Z, Y, X, stack=False, norm=False):
    grid_z = np.linspace(0.0, Z-1, Z)
    grid_z = np.reshape(grid_z, [Z, 1, 1])
    grid_z = np.tile(grid_z, [1, Y, X])

    grid_y = np.linspace(0.0, Y-1, Y)
    grid_y = np.reshape(grid_y, [1, Y, 1])
    grid_y = np.tile(grid_y, [Z, 1, X])

    grid_x = np.linspace(0.0, X-1, X)
    grid_x = np.reshape(grid_x, [1, 1, X])
    grid_x = np.tile(grid_x, [Z, Y, 1])

    if norm:
        grid_z, grid_y, grid_x = normalize_grid3D(
            grid_z, grid_y, grid_x, Z, Y, X)

    if stack:
        # note we stack in xyz order
        # (see https://pytorch.org/docs/stable/nn.functional.html#torch.nn.functional.grid_sample)
        grid = np.stack([grid_x, grid_y, grid_z], dim=-1)
        return grid
    else:
        return grid_z, grid_y, grid_x

def sub2ind(height, width, y, x):
    return y*width + x

def sql2_on_axis(x, axis, keepdim=True):
    return torch.sum(x**2, axis, keepdim=keepdim)

def l2_on_axis(x, axis, keepdim=True):
    return torch.sqrt(EPS + sql2_on_axis(x, axis, keepdim=keepdim))

def l1_on_axis(x, axis, keepdim=True):
    return torch.sum(torch.abs(x), axis, keepdim=keepdim)

def sub2ind3D(depth, height, width, d, h, w):
    # when gathering/scattering with these inds, the tensor should be Z x Y x X
    return d*height*width + h*width + w

def gradient3D(x, absolute=False, square=False):
    # x should be B x C x D x H x W
    dz = x[:, :, 1:, :, :] - x[:, :, :-1, :, :]
    dy = x[:, :, :, 1:, :] - x[:, :, :, :-1, :]
    dx = x[:, :, :, :, 1:] - x[:, :, :, :, :-1]

    # zeros = tf.zeros_like(x)
    zeros = torch.zeros_like(x)
    zero_z = zeros[:, :, 0:1, :, :]
    zero_y = zeros[:, :, :, 0:1, :]
    zero_x = zeros[:, :, :, :, 0:1]
    dz = torch.cat([dz, zero_z], axis=2)
    dy = torch.cat([dy, zero_y], axis=3)
    dx = torch.cat([dx, zero_x], axis=4)
    if absolute:
        dz = torch.abs(dz)
        dy = torch.abs(dy)
        dx = torch.abs(dx)
    if square:
        dz = dz ** 2
        dy = dy ** 2
        dx = dx ** 2
    return dz, dy, dx

def gradient2D(x, absolute=False, square=False):
    # x should be B x C x H x W
    dh = x[:, :, 1:, :] - x[:, :, :-1, :]
    dw = x[:, :, :, 1:] - x[:, :, :, :-1]

    # zeros = tf.zeros_like(x)
    zeros = torch.zeros_like(x)
    zero_h = zeros[:, :, 0:1, :]
    zero_w = zeros[:, :, :, 0:1]
    dh = torch.cat([dh, zero_h], axis=2)
    dw = torch.cat([dw, zero_w], axis=3)
    if absolute:
        dh = torch.abs(dh)
        dw = torch.abs(dw)
    if square:
        dh = dh ** 2
        dw = dw ** 2
    return dh, dw

def matmul2(mat1, mat2):
    return torch.matmul(mat1, mat2)

def matmul3(mat1, mat2, mat3):
    return torch.matmul(mat1, torch.matmul(mat2, mat3))

def downsample(img, factor):
    down = torch.nn.AvgPool2d(factor)
    img = down(img)
    return img

def downsample3D(vox, factor):
    down = torch.nn.AvgPool3d(factor)
    vox = down(vox)
    return vox

def downsample3Dflow(flow, factor):
    down = torch.nn.AvgPool3d(factor)
    flow = down(flow) * 1./factor
    return flow

def l2_normalize(x, dim=1):
    # dim1 is the channel dim
    return F.normalize(x, p=2, dim=dim)

def hard_argmax3D(tensor):
    B, Z, Y, X = list(tensor.shape)

    flat_tensor = tensor.reshape(B, -1)
    argmax = torch.argmax(flat_tensor, dim=1)

    # convert the indices into 3D coordinates
    argmax_z = argmax // (Y*X)
    argmax_y = (argmax % (Y*X)) // X
    argmax_x = (argmax % (Y*X)) % X

    argmax_z = argmax_z.reshape(B)
    argmax_y = argmax_y.reshape(B)
    argmax_x = argmax_x.reshape(B)
    return argmax_z, argmax_y, argmax_x

def argmax3D(heat, hard=False):
    B, Z, Y, X = list(heat.shape)

    if hard:
        # hard argmax
        loc_z, loc_y, loc_x = hard_argmax3D(heat)
        loc_z = loc_z.float()
        loc_y = loc_y.float()
        loc_x = loc_x.float()
    else:
        heat = heat.reshape(B, Z*Y*X)
        prob = torch.nn.functional.softmax(heat, dim=1)

        grid_z, grid_y, grid_x = meshgrid3D(B, Z, Y, X)

        grid_z = grid_z.reshape(B, -1)
        grid_y = grid_y.reshape(B, -1)
        grid_x = grid_x.reshape(B, -1)
        
        loc_z = torch.sum(grid_z*prob, dim=1)
        loc_y = torch.sum(grid_y*prob, dim=1)
        loc_x = torch.sum(grid_x*prob, dim=1)
        # these are B
    return loc_z, loc_y, loc_x

##########################################################


##########################################################
####				 NLU							  ####
##########################################################

def get_alignedboxes2thetaformat(aligned_boxes):
    B,N,_,_ = list(aligned_boxes.shape)
    aligned_boxes = torch.reshape(aligned_boxes,[B,N,6])
    B,N,_ = list(aligned_boxes.shape)
    xmin,ymin,zmin,xmax,ymax,zmax = torch.unbind(torch.tensor(aligned_boxes), dim=-1)
    xc = (xmin+xmax)/2.0
    yc = (ymin+ymax)/2.0
    zc = (zmin+zmax)/2.0
    w = xmax-xmin
    h = ymax - ymin
    d = zmax - zmin
    zeros = torch.zeros([B,N]).cuda()
    boxes = torch.stack([xc,yc,zc,w,h,d,zeros,zeros,zeros],dim=-1)
    return boxes

def get_ends_of_corner(boxes):
    min_box = torch.min(boxes,dim=2,keepdim=True).values
    max_box = torch.max(boxes,dim=2,keepdim=True).values
    boxes_ends = torch.cat([min_box,max_box],dim=2)
    return boxes_ends


##########################################################
####				 utils_geom					     #####
##########################################################

def apply_4x4(RT, xyz):
    B, N, _ = list(xyz.shape)
    ones = torch.ones_like(xyz[:,:,0:1])
    xyz1 = torch.cat([xyz, ones], 2)
    xyz1_t = torch.transpose(xyz1, 1, 2)
    # this is B x 4 x N
    xyz2_t = torch.matmul(RT, xyz1_t)
    xyz2 = torch.transpose(xyz2_t, 1, 2)
    xyz2 = xyz2[:,:,:3]
    return xyz2

def eye_3x3(B):
    rt = torch.eye(3, device=torch.device('cuda')).view(1,3,3).repeat([B, 1, 1])
    return rt

def eye_4x4(B):
    rt = torch.eye(4, device=torch.device('cuda')).view(1,4,4).repeat([B, 1, 1])
    return rt

def merge_rt(r, t):
    # r is B x 3 x 3
    # t is B x 3
    B, C, D = list(r.shape)
    B2, D2 = list(t.shape)
    assert(C==3)
    assert(D==3)
    assert(B==B2)
    assert(D2==3)
    t = t.view(B, 3)
    rt = eye_4x4(B)
    rt[:,:3,:3] = r
    rt[:,:3,3] = t
    return rt

def transform_boxes_to_corners(boxes):
	# returns corners, shaped B x N x 8 x 3
	B, N, D = list(boxes.shape)
	assert(D==9)

	__p = lambda x: pack_seqdim(x, B)
	__u = lambda x: unpack_seqdim(x, B)

	boxes_ = __p(boxes)
	corners_ = transform_boxes_to_corners_single(boxes_)
	corners = __u(corners_)
	return corners

def transform_boxes_to_corners_single(boxes):
    N, D = list(boxes.shape)
    assert(D==9)
    
    xc,yc,zc,lx,ly,lz,rx,ry,rz = torch.unbind(boxes, axis=1)
    # these are each shaped N

    ref_T_obj = convert_box_to_ref_T_obj(boxes)

    xs = torch.stack([-lx/2., -lx/2., -lx/2., -lx/2., lx/2., lx/2., lx/2., lx/2.], axis=1)
    ys = torch.stack([-ly/2., -ly/2., ly/2., ly/2., -ly/2., -ly/2., ly/2., ly/2.], axis=1)
    zs = torch.stack([-lz/2., lz/2., -lz/2., lz/2., -lz/2., lz/2., -lz/2., lz/2.], axis=1)
    
    xyz_obj = torch.stack([xs, ys, zs], axis=2)
    # centered_box is N x 8 x 3

    xyz_ref = apply_4x4(ref_T_obj, xyz_obj)
    # xyz_ref is N x 8 x 3
    return xyz_ref


def convert_box_to_ref_T_obj(box3D):
    # turn the box into obj_T_ref (i.e., obj_T_cam)
    B = list(box3D.shape)[0]
    
    # box3D is B x 9
    x, y, z, lx, ly, lz, rx, ry, rz = torch.unbind(box3D, axis=1)
    rot0 = eye_3x3(B)
    tra = torch.stack([x, y, z], axis=1)
    center_T_ref = merge_rt(rot0, -tra)
    # center_T_ref is B x 4 x 4
    
    t0 = torch.zeros([B, 3])
    rot = eul2rotm(rx, -ry, -rz)
    obj_T_center = merge_rt(rot, t0)
    # this is B x 4 x 4

    # we want obj_T_ref
    # first we to translate to center,
    # and then rotate around the origin
    obj_T_ref = matmul2(obj_T_center, center_T_ref)

    # return the inverse of this, so that we can transform obj corners into cam coords
    ref_T_obj = obj_T_ref.inverse()
    return ref_T_obj

def eul2rotm(rx, ry, rz):
    # inputs are shaped B
    # this func is copied from matlab
    # R = [  cy*cz   sy*sx*cz-sz*cx    sy*cx*cz+sz*sx
    #        cy*sz   sy*sx*sz+cz*cx    sy*cx*sz-cz*sx
    #        -sy            cy*sx             cy*cx]
    rx = torch.unsqueeze(rx, dim=1)
    ry = torch.unsqueeze(ry, dim=1)
    rz = torch.unsqueeze(rz, dim=1)
    # these are B x 1
    sinz = torch.sin(rz)
    siny = torch.sin(ry)
    sinx = torch.sin(rx)
    cosz = torch.cos(rz)
    cosy = torch.cos(ry)
    cosx = torch.cos(rx)
    r11 = cosy*cosz
    r12 = sinx*siny*cosz - cosx*sinz
    r13 = cosx*siny*cosz + sinx*sinz
    r21 = cosy*sinz
    r22 = sinx*siny*sinz + cosx*cosz
    r23 = cosx*siny*sinz - sinx*cosz
    r31 = -siny
    r32 = sinx*cosy
    r33 = cosx*cosy
    r1 = torch.stack([r11,r12,r13],dim=2)
    r2 = torch.stack([r21,r22,r23],dim=2)
    r3 = torch.stack([r31,r32,r33],dim=2)
    r = torch.cat([r1,r2,r3],dim=1)
    return r

def safe_inverse(a): #parallel version
    B, _, _ = list(a.shape)
    inv = a.clone()
    r_transpose = a[:, :3, :3].transpose(1,2) #inverse of rotation matrix

    inv[:, :3, :3] = r_transpose
    inv[:, :3, 3:4] = -torch.matmul(r_transpose, a[:, :3, 3:4])

    return inv

def rotm2eul(r):
    # r is Bx3x3
    r00 = r[:,0,0]
    r10 = r[:,1,0]
    r11 = r[:,1,1]
    r12 = r[:,1,2]
    r20 = r[:,2,0]
    r21 = r[:,2,1]
    r22 = r[:,2,2]
    
    ## python guide:
    # if sy > 1e-6: # singular
    #     x = math.atan2(R[2,1] , R[2,2])
    #     y = math.atan2(-R[2,0], sy)
    #     z = math.atan2(R[1,0], R[0,0])
    # else:
    #     x = math.atan2(-R[1,2], R[1,1])
    #     y = math.atan2(-R[2,0], sy)
    #     z = 0
    
    sy = torch.sqrt(r00*r00 + r10*r10)
    
    cond = (sy > 1e-6)
    rx = torch.where(cond, torch.atan2(r21, r22), torch.atan2(-r12, r11))
    ry = torch.where(cond, torch.atan2(-r20, sy), torch.atan2(-r20, sy))
    rz = torch.where(cond, torch.atan2(r10, r00), torch.zeros_like(r20))

    # rx = torch.atan2(r21, r22)
    # ry = torch.atan2(-r20, sy)
    # rz = torch.atan2(r10, r00)
    # rx[cond] = torch.atan2(-r12, r11)
    # ry[cond] = torch.atan2(-r20, sy)
    # rz[cond] = 0.0
    return rx, ry, rz


def apply_pix_T_cam(pix_T_cam, xyz):

    fx, fy, x0, y0 = split_intrinsics(pix_T_cam)
    
    # xyz is shaped B x H*W x 3
    # returns xy, shaped B x H*W x 2
    
    B, N, C = list(xyz.shape)
    assert(C==3)
    
    x, y, z = torch.unbind(xyz, axis=-1)

    fx = torch.reshape(fx, [B, 1])
    fy = torch.reshape(fy, [B, 1])
    x0 = torch.reshape(x0, [B, 1])
    y0 = torch.reshape(y0, [B, 1])

    EPS=1e-6
    x = (x*fx)/(z+EPS)+x0
    y = (y*fy)/(z+EPS)+y0
    xy = torch.stack([x, y], axis=-1)
    return xy


def split_intrinsics(K):
    # K is B x 3 x 3 or B x 4 x 4
    fx = K[:,0,0]
    fy = K[:,1,1]
    x0 = K[:,0,2]
    y0 = K[:,1,2]
    return fx, fy, x0, y0


def transform_corners_to_boxes(corners):
    # corners is B x N x 8 x 3
    B, N, C, D = corners.shape
    assert(C==8)
    assert(D==3)
    # do them all at once
    __p = lambda x: pack_seqdim(x, B)
    __u = lambda x: unpack_seqdim(x, B)
    corners_ = __p(corners)
    boxes_ = transform_corners_to_boxes_single(corners_)
    boxes_ = boxes_.cuda()
    boxes = __u(boxes_)
    return boxes

def transform_corners_to_boxes_single(corners):
    # corners is B x 8 x 3
    corners = corners.detach().cpu().numpy()

    # assert(False) # this function has a flaw; use rigid_transform_boxes instead, or fix it.
    # # i believe you can fix it using what i noticed in rigid_transform_boxes:
    # # if we are looking at the box backwards, the rx/rz dirs flip

    # we want to transform each one to a box
    # note that the rotation may flip 180deg, since corners do not have this info
    
    boxes = []
    for ind, corner_set in enumerate(corners):
        xs = corner_set[:,0]
        ys = corner_set[:,1]
        zs = corner_set[:,2]
        # these are 8 each

        xc = np.mean(xs)
        yc = np.mean(ys)
        zc = np.mean(zs)

        # we constructed the corners like this:
        # xs = tf.stack([-lx/2., -lx/2., -lx/2., -lx/2., lx/2., lx/2., lx/2., lx/2.], axis=1)
        # ys = tf.stack([-ly/2., -ly/2., ly/2., ly/2., -ly/2., -ly/2., ly/2., ly/2.], axis=1)
        # zs = tf.stack([-lz/2., lz/2., -lz/2., lz/2., -lz/2., lz/2., -lz/2., lz/2.], axis=1)
        # # so we can recover lengths like this:
        # lx = np.linalg.norm(xs[-1] - xs[0])
        # ly = np.linalg.norm(ys[-1] - ys[0])
        # lz = np.linalg.norm(zs[-1] - zs[0])
        # but that's a noisy estimate apparently. let's try all pairs

        # rotations are a bit more interesting...

        # defining the corners as: clockwise backcar face, clockwise frontcar face:
        #   E -------- F
        #  /|         /|
        # A -------- B .
        # | |        | |
        # . H -------- G
        # |/         |/
        # D -------- C

        # the ordered eight indices are:
        # A E D H B F C G

        # unstack on first dim
        A, E, D, H, B, F, C, G = corner_set

        back = [A, B, C, D] # back of car is closer to us
        front = [E, F, G, H]
        top = [A, E, B, F]
        bottom = [D, C, H, G]

        front = np.stack(front, axis=0)
        back = np.stack(back, axis=0)
        top = np.stack(top, axis=0)
        bottom = np.stack(bottom, axis=0)
        # these are 4 x 3

        back_z = np.mean(back[:,2])
        front_z = np.mean(front[:,2])
        # usually the front has bigger coords than back
        backwards = not (front_z > back_z)

        front_y = np.mean(front[:,1])
        back_y = np.mean(back[:,1])
        # someetimes the front dips down
        dips_down = front_y > back_y

        # the bottom should have bigger y coords than the bottom (since y increases down)
        top_y = np.mean(top[:,2])
        bottom_y = np.mean(bottom[:,2])
        upside_down = not (top_y < bottom_y)

        # rx: i need anything but x-aligned bars
        # there are 8 of these
        # atan2 wants the y part then the x part; here this means y then z

        x_bars = [[A, B], [D, C], [E, F], [H, G]]
        y_bars = [[A, D], [B, C], [E, H], [F, G]]
        z_bars = [[A, E], [B, F], [D, H], [C, G]]

        lx = 0.0
        for x_bar in x_bars:
            x0, x1 = x_bar
            lx += np.linalg.norm(x1-x0)
        lx /= 4.0

        ly = 0.0
        for y_bar in y_bars:
            y0, y1 = y_bar
            ly += np.linalg.norm(y1-y0)
        ly /= 4.0

        lz = 0.0
        for z_bar in z_bars:
            z0, z1 = z_bar
            lz += np.linalg.norm(z1-z0)
        lz /= 4.0
        rx = 0.0
        for bar in z_bars:
            pt1, pt2 = bar
            intermed = np.arctan2((pt1[1] - pt2[1]), (pt1[2] - pt2[2]))
            rx += intermed

        rx /= 4.0

        ry = 0.0
        for bar in z_bars:
            pt1, pt2 = bar
            intermed = np.arctan2((pt1[2] - pt2[2]), (pt1[0] - pt2[0]))
            ry += intermed

        ry /= 4.0

        rz = 0.0
        for bar in x_bars:
            pt1, pt2 = bar
            intermed = np.arctan2((pt1[1] - pt2[1]), (pt1[0] - pt2[0]))
            rz += intermed

        rz /= 4.0

        ry += np.pi/2.0

        if backwards:
            ry = -ry
        if not backwards:
            ry = ry - np.pi

        box = np.array([xc, yc, zc, lx, ly, lz, rx, ry, rz])
        boxes.append(box)
    boxes = np.stack(boxes, axis=0).astype(np.float32)
    return torch.from_numpy(boxes)


def get_camM_T_camXs(origin_T_camXs, ind=0):
    B, S = list(origin_T_camXs.shape)[0:2]
    camM_T_camXs = torch.zeros_like(origin_T_camXs)
    for b in range(B):
        camM_T_origin = safe_inverse_single(origin_T_camXs[b,ind])
        for s in range(S):
            camM_T_camXs[b,s] = torch.matmul(camM_T_origin, origin_T_camXs[b,s])
    return camM_T_camXs

def split_rt_single(rt):
    r = rt[:3, :3]
    t = rt[:3, 3].view(3)
    return r, t

def split_rt(rt):
    r = rt[:, :3, :3]
    t = rt[:, :3, 3].view(-1, 3)
    return r, t

def safe_inverse_single(a):
    r, t = split_rt_single(a)
    t = t.view(3,1)
    r_transpose = r.t()
    inv = torch.cat([r_transpose, -torch.matmul(r_transpose, t)], 1)
    bottom_row = a[3:4, :] # this is [0, 0, 0, 1]
    # bottom_row = torch.tensor([0.,0.,0.,1.]).view(1,4) 
    inv = torch.cat([inv, bottom_row], 0)
    return inv