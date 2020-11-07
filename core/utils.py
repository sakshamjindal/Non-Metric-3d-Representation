import torch
import torch.nn as nn
import numpy as np
import faiss
from tqdm import tqdm
import random
import matplotlib.pyplot as plt
import cv2

import ipdb
st = ipdb.set_trace

def compute_features(eval_loader, model, arg):
    
    if arg.mode=="node":
        num_embedding = arg.hyp_N
        
    elif arg.mode=="spatial":
        num_embedding = arg.hyp_N**2
        
    print('Computing features...')
    model.eval()
    features = torch.zeros(len(eval_loader.dataset),num_embedding, 256).cuda()
    print(features.shape)
    for i, (feed_dict_q, metadata) in enumerate(tqdm(eval_loader)):
        with torch.no_grad():
            feat = model(feed_dict_q, None, metadata, is_eval=True)
            feat = feat.reshape(-1,num_embedding, 256)
            index = metadata['index']
            features[index] = feat
            
    with torch.no_grad():
        features = features.view(-1,256)
        
    return features.cpu()

    
def save_checkpoint(state, is_best, filename='checkpoint.pth.tar'):
    torch.save(state, filename)
    if is_best:
        shutil.copyfile(filename, 'model_best.pth.tar')


class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self, name, fmt=':f'):
        self.name = name
        self.fmt = fmt
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def __str__(self):
        fmtstr = '{name} {val' + self.fmt + '} ({avg' + self.fmt + '})'
        return fmtstr.format(**self.__dict__)


class ProgressMeter(object):
    def __init__(self, num_batches, meters, prefix=""):
        self.batch_fmtstr = self._get_batch_fmtstr(num_batches)
        self.meters = meters
        self.prefix = prefix

    def display(self, batch):
        entries = [self.prefix + self.batch_fmtstr.format(batch)]
        entries += [str(meter) for meter in self.meters]
        print('\t'.join(entries))

    def _get_batch_fmtstr(self, num_batches):
        num_digits = len(str(num_batches // 1))
        fmt = '{:' + str(num_digits) + 'd}'
        return '[' + fmt + '/' + fmt.format(num_batches) + ']'


def adjust_learning_rate(optimizer, epoch, args):
    """Decay the learning rate based on schedule"""
    lr = args.lr
    if args.cos:  # cosine lr schedule
        lr *= 0.5 * (1. + math.cos(math.pi * epoch / args.epochs))
    else:  # stepwise lr schedule
        for milestone in args.schedule:
            lr *= 0.1 if epoch >= milestone else 1.
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].view(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res

def run_kmeans(x, args):
    """
    Args:
        x: data to be clustered
    """
    
    
    results = {'im2cluster':[],'centroids':[],'density':[]}
    
    for seed, num_cluster in enumerate(args.num_cluster):
        print('performing kmeans clustering on ...',num_cluster)
        # intialize faiss clustering parameters
        d = x.shape[1]
        k = int(num_cluster)
        clus = faiss.Clustering(d, k)
        clus.verbose = True
        clus.niter = 20
        clus.nredo = 5
        clus.seed = seed
        clus.max_points_per_centroid = 1000
        clus.min_points_per_centroid = 10

        res = faiss.StandardGpuResources()

        cfg = faiss.GpuIndexFlatConfig()
        cfg.useFloat16 = False
        cfg.device = 0   
        index = faiss.GpuIndexFlatL2(res, d, cfg)  

        clus.train(x, index)   

        D, I = index.search(x, 1) # for each sample, find cluster distance and assignments
        im2cluster = [int(n[0]) for n in I]
        
        # get cluster centroids
        centroids = faiss.vector_to_array(clus.centroids).reshape(k,d)
        
        # sample-to-centroid distances for each cluster 
        Dcluster = [[] for c in range(k)]          
        for im,i in enumerate(im2cluster):
            Dcluster[i].append(D[im][0])
        
        # concentration estimation (phi)        
        density = np.zeros(k)
        for i,dist in enumerate(Dcluster):
            if len(dist)>1:
                d = (np.asarray(dist)**0.5).mean()/np.log(len(dist)+10)            
                density[i] = d     
                
        #if cluster only has one point, use the max to estimate its concentration        
        dmax = density.max()
        for i,dist in enumerate(Dcluster):
            if len(dist)<=1:
                density[i] = dmax 

        density = density.clip(np.percentile(density,10),np.percentile(density,90)) #clamp extreme values for stability
        density = args.temperature*density/density.mean()  #scale the mean to temperature 
        
        # convert to cuda Tensors for broadcast
        centroids = torch.Tensor(centroids).cuda()
        centroids = nn.functional.normalize(centroids, p=2, dim=1)    

        im2cluster = torch.LongTensor(im2cluster).cuda()               
        density = torch.Tensor(density).cuda()
        
        results['centroids'].append(centroids)
        results['density'].append(density)
        results['im2cluster'].append(im2cluster)    
        
    return results

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
            
    def update(self, embeds, images, sub_boxs=None, obj_boxs=None):
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
                # add to the back
                self.embeds.append(embed)
                self.images.append(image)
        
                self.sub_box.append(sub_box)
                self.obj_box.append(obj_box)
            

        

def store_to_pool(pool_e, pool_g, feed_dict_q, feed_dict_k, metadata, model, args):
    
#     st()
    #print('Storing to pool...')
    model.eval()
    with torch.no_grad():
        feat_q = model(feed_dict_q, None, metadata, is_eval=True)
        feat_k = model(feed_dict_k, None, metadata, is_eval=True)
        
        dim1 = feat_q.shape[0]
        img_q = torch.zeros([dim1, 3, 256, 256])
        img_k = torch.zeros([dim1, 3, 256, 256])
        
        
        if args.mode=='node':
            cnt = 0
            
            for b in range(feed_dict_q["objects_boxes"].shape[0]//args.hyp_N):
                for s in range(args.hyp_N):
                    img_q[cnt] = feed_dict_q["images"][b]
                    img_k[cnt] = feed_dict_k["images"][b]
                    cnt += 1
                    
            pool_e.update(feat_q, img_q, feed_dict_q["objects_boxes"], None)
            pool_g.update(feat_k, img_k, feed_dict_k["objects_boxes"], None)
            
        else:
            dim1 = feat_q.shape[0]
            subj_q = torch.zeros([dim1, 4])
            subj_k = torch.zeros([dim1, 4])
            obj_q = torch.zeros([dim1, 4])
            obj_k = torch.zeros([dim1, 4])
            
            cnt = 0
            for b in range(feed_dict_q["objects_boxes"].shape[0]//args.hyp_N):
                for s in range(args.hyp_N):
                    for o in range(args.hyp_N):
                        img_q[cnt] = feed_dict_q["images"][b]
                        img_k[cnt] = feed_dict_k["images"][b]
                        cnt += 1
            
            cnt = 0
            for b in range(feed_dict_q["objects_boxes"].shape[0]//args.hyp_N):
                for s in range(args.hyp_N):
                    for o in range(args.hyp_N):
                        start_idx = b*args.hyp_N
                        subj_q[cnt] = feed_dict_q["objects_boxes"][start_idx + s]
                        obj_q[cnt] = feed_dict_q["objects_boxes"][start_idx + o]
                        cnt += 1
                
                
            cnt = 0
            for b in range(feed_dict_q["objects_boxes"].shape[0]//args.hyp_N):
                for s in range(args.hyp_N):
                    for o in range(args.hyp_N):
                        start_idx = b*args.hyp_N
                        subj_k[cnt] = feed_dict_k["objects_boxes"][start_idx + s]
                        obj_k[cnt] = feed_dict_k["objects_boxes"][start_idx + o]
                        cnt += 1
                    
            pool_e.update(feat_q, img_q, subj_q, obj_q)
            pool_g.update(feat_k, img_k, subj_k, obj_k)
            
                
    return


from sklearn.neighbors import NearestNeighbors
def random_retrieve_topk(args, pool_e=None, pool_g=None, imgs_to_view=3):
    
#     st()
    
    print("==> Fitting k-nearest-neighbour model on pool g...")
    knn = NearestNeighbors(n_neighbors=10, metric="cosine")
    knn.fit(torch.stack(pool_g.embeds).cpu())
    
    # select imgs_to_view images from pool q randomly
    query_indices_to_use = random.sample(range(0, pool_e.num), imgs_to_view)
    assert pool_e.num==pool_g.num
    
    
    figures_gen = []
    print(query_indices_to_use)

    # Perform image retrieval on test images
    print("==> Performing image retrieval on test images...")
    for i, index in enumerate(query_indices_to_use):
        temp = []
        _, indices = knn.kneighbors(torch.reshape(pool_e.embeds[index], (1,-1)).cpu()) # find k nearest train neighbours
   
        img_query = pool_e.images[index].permute(1,2,0).cpu().numpy() # query image
        temp.append(img_query)
        
        imgs_retrieval = [pool_g.images[idx].permute(1,2,0).cpu().numpy() for idx in indices.flatten()]# retrieval images
        temp.extend(imgs_retrieval)
        
        imgs_sub_boxes = [pool_e.sub_box[index]]
        imgs_sub_boxes.extend([pool_g.sub_box[idx] for idx in indices.flatten()])
        
        
        imgs_obj_boxes = None
        
        if args.mode=='spatial':
            imgs_obj_boxes = [pool_e.obj_box[index]]
            imgs_obj_boxes.extend([pool_g.obj_box[idx] for idx in indices.flatten()])
        

        figures_gen.append([temp, imgs_sub_boxes, imgs_obj_boxes])

    fig = plot_query_retrieval(figures_gen, None, args)
        
    return fig

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
    fig = plt.figure(figsize=(20, 4))
    for idx in range(n_retrieval):
        for im in range(0, 11):
            ax = fig.add_subplot(n_retrieval, 11, 11*idx+im+1,xticks=[], yticks=[])
            if args.mode=='node':
                im_to_plot = draw_bounding_box(imgs_retrieval[idx][0][im], imgs_retrieval[idx][1][im], None)
            else:
                im_to_plot = draw_bounding_box(imgs_retrieval[idx][0][im], imgs_retrieval[idx][1][im], imgs_retrieval[idx][2][im])
            if im==0: 
                ax.set_title('Query')
            else:
                ax.set_title('Top_'+str(im))
            ax.imshow(im_to_plot)
    plt.close(fig)
    return fig
        