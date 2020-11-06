import torch
import numpy as np
import ipdb
import core.utils_data as utils_disco
st = ipdb.set_trace
import torch.nn.functional as F


XMIN = -7.5 # right (neg is left)
XMAX = 7.5 # right
YMIN = -7.5 # down (neg is up)
YMAX = 7.5 # down
ZMIN = 0.0 # forward
ZMAX = 16.0 # forward 


def Ref2Mem(xyz, Z, Y, X):
    # xyz is B x N x 3, in ref coordinates
    # transforms velo coordinates into mem coordinates
    B, N, C = list(xyz.shape)
    mem_T_ref = get_mem_T_ref(B, Z, Y, X)
    xyz = utils_disco.apply_4x4(mem_T_ref, xyz)
    return xyz

def get_mem_T_ref(B, Z, Y, X):
    # sometimes we want the mat itself
    # note this is not a rigid transform
    
    # for interpretability, let's construct this in two steps...

    # translation
    center_T_ref = utils_disco.eye_4x4(B)
    center_T_ref[:,0,3] = -XMIN
    center_T_ref[:,1,3] = -YMIN
    center_T_ref[:,2,3] = -ZMIN

    VOX_SIZE_X = (XMAX-XMIN)/float(X)
    VOX_SIZE_Y = (YMAX-YMIN)/float(Y)
    VOX_SIZE_Z = (ZMAX-ZMIN)/float(Z)
    
    # scaling
    mem_T_center = utils_disco.eye_4x4(B)
    mem_T_center[:,0,0] = 1./VOX_SIZE_X
    mem_T_center[:,1,1] = 1./VOX_SIZE_Y
    mem_T_center[:,2,2] = 1./VOX_SIZE_Z
    mem_T_ref = utils_disco.matmul2(mem_T_center, center_T_ref)
    
    return mem_T_ref

def Mem2Ref(xyz_mem, Z, Y, X):
    # xyz is B x N x 3, in mem coordinates
    # transforms mem coordinates into ref coordinates
    B, N, C = list(xyz_mem.shape)
    ref_T_mem = get_ref_T_mem(B, Z, Y, X)
    xyz_ref = utils_disco.apply_4x4(ref_T_mem, xyz_mem)
    return xyz_ref

def get_ref_T_mem(B, Z, Y, X):
    mem_T_ref = get_mem_T_ref(B, Z, Y, X)
    # note safe_inverse is inapplicable here,
    # since the transform is nonrigid
    ref_T_mem = mem_T_ref.inverse()
    return ref_T_mem

