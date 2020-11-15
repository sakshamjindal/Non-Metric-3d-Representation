from torch.utils.data import DataLoader
import torch

import random
import matplotlib.pyplot as plt
import numpy as np


class DoublePool_O():
    def __init__(self, pool_size, isnode=True):
        self.pool_size = pool_size
        self.isnode = isnode
        random.seed(125)
        if self.pool_size > 0:
            self.num = 0
            self.embeds = []
            #self.scene_nums = []
            self.images = []
            self.sub_box = []
            self.scene_num = []
            self.view_num = []
            if not self.isnode:
                self.obj_box = []

    def fetch(self):
        if self.isnode:
            return self.embeds, self.images, self.sub_box #, self.classes,None, self.visual2D
        return self.embeds, self.images, self.sub_box, self.obj_box
            
    def is_full(self):
        full = self.num==self.pool_size
        # print 'num = %d; full = %s' % (self.num, full)
        return full
            
    def update(self, embeds, images, sub_boxs=None, obj_boxs=None, scene_num=None, view_num=None):
        # embeds is B x ... x C
        # images is B x ... x 3
        assert embeds.shape[0]==images.shape[0]
        assert embeds.shape[0]==sub_boxs.shape[0]
        

        if self.isnode:
            for embed, image, sub_box in zip(embeds, images, sub_boxs):
                if self.num < self.pool_size:
                    # the pool is not full, so let's add this in
                    self.num = self.num + 1
                else:
                    # the pool is full
                    # pop from the front
                    self.embeds.pop(0)
                    self.images.pop(0)
             
                    self.sub_box.pop(0)
                    #self.visual2D.pop(0)
                # add to the back
                self.embeds.append(embed)
                self.images.append(image)
           
                self.sub_box.append(sub_box)
        else:
            assert sub_boxs.shape[0]==obj_boxs.shape[0]
            
            for embed, image, sub_box, obj_box in zip(embeds, images, sub_boxs, obj_boxs):
                if self.num < self.pool_size:
                    # the pool is not full, so let's add this in
                    self.num = self.num + 1
                else:
                    # the pool is full
                    # pop from the front
                    self.embeds.pop(0)
                    self.images.pop(0)
               
                    self.sub_box.pop(0)
                    self.obj_box.pop(0)
                    
                    self.scene_num.pop(0)
                    self.view_num.pop(0)
                    
                # add to the back
                self.embeds.append(embed)
                self.images.append(image)
        
                self.sub_box.append(sub_box)
                self.obj_box.append(obj_box)
                
                self.scene_num.append(scene_num)
                self.view_num.append(view_num)
                
from tqdm import tqdm

def store_to_pool_e(pool_e, feed_dict_q, metadata, model, args, scene_num, view_num):
    
#     st()
    #print('Storing to pool...')    
    
    model.eval()
    with torch.no_grad():
        feat_q = model(feed_dict_q, None, metadata, is_eval=True)
        
        dim1 = feat_q.shape[0]
        img_q = torch.zeros([dim1, 3, 256, 256])        
        
        if args.mode=='node':
            cnt = 0
            
            for b in range(feed_dict_q["objects_boxes"].shape[0]//args.hyp_N):
                for s in range(args.hyp_N):
                    img_q[cnt] = feed_dict_q["images"][b]
                    cnt += 1
                    
            pool_e.update(feat_q, img_q, feed_dict_q["objects_boxes"], None)
            
        else:
            dim1 = feat_q.shape[0]
            subj_q = torch.zeros([dim1, 4])
            obj_q = torch.zeros([dim1, 4])
            
            cnt = 0
            for b in range(feed_dict_q["objects_boxes"].shape[0]//args.hyp_N):
                for s in range(args.hyp_N):
                    for o in range(args.hyp_N):
                        img_q[cnt] = feed_dict_q["images"][b]
                        cnt += 1
            
            cnt = 0
            for b in range(feed_dict_q["objects_boxes"].shape[0]//args.hyp_N):
                for s in range(args.hyp_N):
                    for o in range(args.hyp_N):
                        start_idx = b*args.hyp_N
                        subj_q[cnt] = feed_dict_q["objects_boxes"][start_idx + s]
                        obj_q[cnt] = feed_dict_q["objects_boxes"][start_idx + o]
                        cnt += 1
                    
            pool_e.update(feat_q, img_q, subj_q, obj_q, scene_num, view_num)           
                
    return

def store_to_pool_g(pool_g, feed_dict_k, metadata, model, args, scene_num, view_num):
    
    
    model.eval()
    with torch.no_grad():
        feat_k = model(feed_dict_k, None, metadata, is_viewpoint_eval=True)
        
        dim1 = feat_k.shape[0]
        img_k = torch.zeros([dim1, 3, 256, 256])        
        
        if args.mode=='node':
            print("")
            cnt = 0
            
            for b in range(feed_dict_k["objects_boxes"].shape[0]//args.hyp_N):
                for s in range(args.hyp_N):
                    img_k[cnt] = feed_dict_k["images"][b]
                    cnt += 1
                    
            pool_g.update(feat_k, img_k, feed_dict_k["objects_boxes"], None)
            
        else:
            dim1 = feat_k.shape[0]
            subj_k= torch.zeros([dim1, 4])
            obj_k = torch.zeros([dim1, 4])
            
            cnt = 0
            for b in range(feed_dict_k["objects_boxes"].shape[0]//args.hyp_N):
                for s in range(args.hyp_N):
                    for o in range(args.hyp_N):
                        img_k[cnt] = feed_dict_k["images"][b]
                        cnt += 1
            
            cnt = 0
            for b in range(feed_dict_k["objects_boxes"].shape[0]//args.hyp_N):
                for s in range(args.hyp_N):
                    for o in range(args.hyp_N):
                        start_idx = b*args.hyp_N
                        subj_k[cnt] = feed_dict_k["objects_boxes"][start_idx + s]
                        obj_k[cnt] = feed_dict_k["objects_boxes"][start_idx + o]
                        cnt += 1
                    
            pool_g.update(feat_k, img_k, subj_k, obj_k, scene_num, view_num)           
                
    return

import cv2

def draw_bounding_box(image, sub_box, obj_box):
        
    x1,y1,x2,y2 = sub_box
    x1 = int(x1.item()); x2 = int(x2.item()); y1 = int(y1.item()); y2 = int(y2.item())
    img1 = cv2.rectangle(image.copy(),(x1,y1),(x2,y2),(0,255,0),2)
    
    if not obj_box==None:
        x1,y1,x2,y2 = obj_box
        x1 = int(x1.item()); x2 = int(x2.item()); y1 = int(y1.item()); y2 = int(y2.item())
        img = cv2.rectangle(img1.copy(),(x1,y1),(x2,y2),(255,0,0),2)
        return img
    
    return img
        

def plot_query_retrieval(imgs_retrieval, outFile, args):
#     st()
    n_retrieval = len(imgs_retrieval)
    fig = plt.figure(figsize=(20, 20))
    for idx in range(n_retrieval):
        for im in range(0, 12):
            img = imgs_retrieval[idx][0][im]
            img = img*255
            img = np.array(img,np.int32)
            ax = fig.add_subplot(n_retrieval, 12, 12*idx+im+1,xticks=[], yticks=[])
            if args.mode=='node':
                im_to_plot = draw_bounding_box(img, imgs_retrieval[idx][1][im], None)
            else:
                im_to_plot = draw_bounding_box(img, imgs_retrieval[idx][1][im], imgs_retrieval[idx][2][im])
            if im>1:
                ax.set_title('S:{}, V:{}'.format(imgs_retrieval[idx][-2][im], imgs_retrieval[idx][-1][im]))
            elif im==0:          
                ax.set_title('S:{}, V:{}'.format(imgs_retrieval[idx][-2][im], imgs_retrieval[idx][-1][im]))
            else:
                ax.set_title('S:{}, V:{}'.format(imgs_retrieval[idx][-2][im], imgs_retrieval[idx][-1][im]))
            ax.imshow((im_to_plot))
    plt.show()
    plt.close(fig)
    return fig

from sklearn.neighbors import NearestNeighbors

def random_retrieve_topk(args, pool_e=None, pool_g=None, pool_f_=None,imgs_to_view=3, k=10):
    
    knn = NearestNeighbors(n_neighbors=k, metric="cosine")
    knn.fit(torch.stack(pool_g.embeds).cpu())
    
    # select imgs_to_view images from pool q randomly
#     query_indices_to_use = random.sample(range(0, pool_e.num), imgs_to_view)

    query_indices_to_use=[]

    while(len(query_indices_to_use)<imgs_to_view):
        ret = np.random.randint(0,pool_e.num,1)[0]
        if ret in query_indices_to_use or ret%4==0 or ret%4==3:
            continue
        else:
            query_indices_to_use.append(ret)
    
    figures_gen = []
    print(query_indices_to_use)

    # Perform image retrieval on test images
    for i, index in enumerate(query_indices_to_use):
        temp = []
        distances, indices = knn.kneighbors(torch.reshape(pool_e.embeds[index], (1,-1)).cpu()) # find k nearest train neighbours

        img_query = pool_e.images[index].permute(1,2,0).cpu().numpy() # query image
        img_key = pool_f_.images[index].permute(1,2,0).cpu().numpy()
        temp.append(img_query)
        temp.append(img_key)
        
        imgs_retrieval = [pool_g.images[idx].permute(1,2,0).cpu().numpy() for idx in indices.flatten()]# retrieval images
        temp.extend(imgs_retrieval)
        
        imgs_sub_boxes = [pool_e.sub_box[index], pool_f_.sub_box[index]]
        imgs_sub_boxes.extend([pool_g.sub_box[idx] for idx in indices.flatten()])

        imgs_obj_boxes = None
        
        if args.mode=='spatial':
            imgs_obj_boxes = [pool_e.obj_box[index], pool_f_.obj_box[index]]
            imgs_obj_boxes.extend([pool_g.obj_box[idx] for idx in indices.flatten()])

            
        scene_nums=[pool_e.scene_num[index], pool_f_.scene_num[index]]
        scene_nums.extend([pool_g.scene_num[idx] for idx in indices.flatten()])
        
        view_nums = [pool_e.view_num[index], pool_f_.view_num[index]]
        view_nums.extend([pool_g.view_num[idx] for idx in indices.flatten()])

        figures_gen.append([temp, imgs_sub_boxes, imgs_obj_boxes, distances[0],scene_nums, view_nums])
        
    fig = plot_query_retrieval(figures_gen, None, args)

    return figures_gen