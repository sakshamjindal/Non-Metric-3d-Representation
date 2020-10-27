import torch 
import numpy as np 
import torch.nn.functional as F
import matplotlib.pyplot as plt
import ipdb 
st = ipdb.set_trace
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

def scale_intrinsics(K, sx, sy):
    fx, fy, x0, y0 = split_intrinsics(K)
    fx = fx*sx
    fy = fy*sy
    x0 = x0*sx
    y0 = y0*sy
    K = pack_intrinsics(fx, fy, x0, y0)
    return K

def split_intrinsics(K):
    # K is B x 3 x 3 or B x 4 x 4
    fx = K[:,0,0]
    fy = K[:,1,1]
    x0 = K[:,0,2]
    y0 = K[:,1,2]
    return fx, fy, x0, y0

def pack_intrinsics(fx, fy, x0, y0):
    B = list(fx.shape)[0]
    K = torch.zeros(B, 4, 4, dtype=torch.float32, device=torch.device('cuda'))
    K[:,0,0] = fx
    K[:,1,1] = fy
    K[:,0,2] = x0
    K[:,1,2] = y0
    K[:,2,2] = 1.0
    K[:,3,3] = 1.0
    return K

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

def get_ends_of_corner(boxes):
    min_box = torch.min(boxes,dim=2,keepdim=True).values
    max_box = torch.max(boxes,dim=2,keepdim=True).values
    boxes_ends = torch.cat([min_box,max_box],dim=2)
    return boxes_ends

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


def transform_boxes_to_corners(boxes, B):
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
    
def matmul2(mat1, mat2):
    return torch.matmul(mat1, mat2)

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


def eye_3x3(B):
    rt = torch.eye(3, device=torch.device('cuda')).view(1,3,3).repeat([B, 1, 1])
    return rt

def eye_3x3s(B, S):
    rt = torch.eye(3, device=torch.device('cuda')).view(1,1,3,3).repeat([B, S, 1, 1])
    return rt

def eye_4x4(B):
    rt = torch.eye(4, device=torch.device('cuda')).view(1,4,4).repeat([B, 1, 1])
    return rt

def eye_4x4s(B, S):
    rt = torch.eye(4, device=torch.device('cuda')).view(1,1,4,4).repeat([B, S, 1, 1])
    return rt

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

def get_cropped_rgb(x_data, v_data, metadata, args, __p, __u, view_idx, writer=None, step=None):
    N = args.N
    __pb = lambda x: pack_boxdim(x, N)
    __ub = lambda x: unpack_boxdim(x, N)

    B = args.batch_size
    gt_boxes_origin_corners = metadata['bbox_origin']
    pix_T_cams_raw = metadata['pix_T_cams_raw']
    score = metadata['score']
    camR_T_origin_raw = metadata['camR_T_origin_raw']
    origin_T_camXs_raw = metadata['origin_T_camXs_raw']
    object_category = metadata['object_category']
    # st()
    camXs_T_origin = __u(safe_inverse(__p(origin_T_camXs_raw)))
    gt_boxes_origin = get_ends_of_corner(gt_boxes_origin_corners)
    gt_boxes_origin_end = torch.reshape(gt_boxes_origin,[B,-1,2,3])
    gt_boxes_origin_theta = get_alignedboxes2thetaformat(gt_boxes_origin_end)
    gt_boxes_origin_corners = transform_boxes_to_corners(gt_boxes_origin_theta, B)
    # st()
    gt_boxesXI_corners = __ub(apply_4x4(camXs_T_origin[:, view_idx], __pb(gt_boxes_origin_corners)))
    bbox_camX_ends = get_ends_of_corner(gt_boxesXI_corners)
    bbox_camX_theta = get_alignedboxes2thetaformat(bbox_camX_ends)
    bbox_camXI_corners = transform_boxes_to_corners(bbox_camX_theta, B)
    pix_T_camXI = pix_T_cams_raw[:,view_idx]
    bbox_pix_corners = __ub(apply_pix_T_cam(pix_T_camXI, __pb(bbox_camXI_corners)))
    # bbox_pix_corners = torch.clamp(bbox_pix_corners,0,63).long()

    extrinsics_list = []
    rgb_list = []
    label_list = []
    for b in range(B):
        for n in range(N):
            if score[b,n] == 1:
                rgb = x_data[b:b+1,0]
                _,_,He, Wi = rgb.shape
                # st()
                objcat = object_category[n][b]
                label_list.append(objcat)
                bbox = bbox_pix_corners[b,n].long()
                # st()
                bbox[:,0] = torch.clamp(bbox[:,0], 0, Wi-1)
                bbox[:,1] = torch.clamp(bbox[:,1], 0, He-1)

                xmin, xmax = torch.min(bbox[:,0]).item(), torch.max(bbox[:,0]).item()
                ymin, ymax = torch.min(bbox[:,1]).item(), torch.max(bbox[:,1]).item()
                
                # st()
                rgb_masked = rgb.clone()
                rgb_masked[:,:,ymin:ymax, xmin:xmax] = 0
                
                cropped_rgb = F.interpolate(rgb[:, :, ymin:ymax, xmin:xmax], (64,64))
                rgb_vis = torch.cat([rgb, rgb_masked], dim=-1)

                # writer.add_image("Cropped_rgb/RGB_RGBMasked", rgb_vis[0])
                # writer.add_image("Cropped_rgb/cropped", cropped_rgb[0])
                # writer.add_text("Cropped_rgb/cropped", objcat)
                rgb_list.append(cropped_rgb[0])
                extrinsics_list.append(v_data[b])
    # st()
    rgb_list_vis = torch.cat(rgb_list, dim=-1)
    concat_text = "__".join(label_list)
#     writer.add_image("Cropped_rgb/RGB_Cropped", rgb_list_vis)
#     writer.add_text("Cropped_rgb/RGB_Cropped", concat_text)

    x_data_ = torch.stack(rgb_list, dim=0)
    v_data_ = torch.stack(extrinsics_list, dim=0)
    return x_data_.unsqueeze(1), v_data_, label_list

    # gt_boxesXI_theta = transform_corners_to_boxes(gt_boxesXI_corners)
    # gt_boxesR_end = nlu.get_ends_of_corner(gt_boxesR_corners)
    # camXI_T_origin = camXs_T_origin[:, view_idx]
    # S = camXs_T_origin.shape[1]
    # N = gt_boxes_origin_corners.shape[1]
    # gt_boxes_origin_corners_ = gt_boxes_origin_corners.unsqueeze(1).repeat(1,S,1,1,1) # B, S, N, 8, 3
    # camXs_T_origin_ = camXs_T_origin.unsqueeze(2).repeat(1,1,N,1,1) # # B, S, N, 4, 4
    # _,_,_,num_pts,xyz = gt_boxes_origin_corners_.shape
    # _,_,_,dim1,dim2 = camXs_T_origin_.shape
    # gt_boxes_origin_corners_ = gt_boxes_origin_corners_.reshape(-1, num_pts, xyz)
    # camXs_T_origin_ = camXs_T_origin_.reshape(-1, dim1, dim2)
    
    # gt_boxesR_corners_ = apply_4x4(camXs_T_origin_, gt_boxes_origin_corners_)
    # gt_boxesR_theta = transform_corners_to_boxes(gt_boxesR_corners_.reshape(-1, N, 8, 3))
    # gt_boxesR_end = get_ends_of_corner(gt_boxesR_corners)
    # return None

def is_dicts_filled(content, style, few_shot_size):
    if len(list(content.keys())) < 3:
        return False

    if len(list(style.keys())) < 16:
        return False
    
    for ckey in content.keys():
        if len(content[ckey]) < few_shot_size:
            return False 
    
    for skey in style.keys():
        if len(style[skey]) < few_shot_size:
            return False

    return True