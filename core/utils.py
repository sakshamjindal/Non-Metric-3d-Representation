import torch
import torch.nn as nn
import numpy as np
import faiss
from tqdm import tqdm
import random
import matplotlib.pyplot as plt

def compute_features(eval_loader, model, args):
    print('Computing features...')
    model.eval()
    features = torch.zeros(len(eval_loader.dataset),256).cuda()
    for i, (feed_dict_q, metadata) in enumerate(eval_loader):
        with torch.no_grad():
            feat = model(feed_dict_q, None, metadata, is_eval=True)
            index = metadata['index']
            features[index] = feat    
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
    def __init__(self, pool_size):
        self.pool_size = pool_size
        random.seed(125)
        if self.pool_size > 0:
            self.num = 0
            self.embeds = []
            self.scene_nums = []
            self.images = []
            #self.classes = []
            
    def fetch(self):
        return self.embeds, self.images, self.scene_nums #, self.classes,None, self.visual2D
            
    def is_full(self):
        full = self.num==self.pool_size
        # print 'num = %d; full = %s' % (self.num, full)
        return full
            
    def update(self, embeds, images, scene_nums, classes=None, vis2Ds=None):
        # embeds is B x ... x C
        # images is B x ... x 3

        #for embed, image, class_val,vis2D in zip(embeds, images,classes, vis2Ds):
        for embed, image, scene_num in zip(embeds, images, scene_nums):
            if self.num < self.pool_size:
                # the pool is not full, so let's add this in
                self.num = self.num + 1
            else:
                # the pool is full
                # pop from the front
                self.embeds.pop(0)
                self.images.pop(0)
                self.scene_nums.pop(0)
                #self.classes.pop(0)
                #self.visual2D.pop(0)
            # add to the back
            self.embeds.append(embed)
            self.images.append(image)
            self.scene_nums.append(scene_num)
            
            #self.classes.append(class_val)
            #self.visual2D.append(vis2D)
        # return self.embeds, self.images
        

def store_to_pool(pool_e, pool_g, feed_dict_q, feed_dict_k, metadata, model, args):
    #print('Storing to pool...')
    model.eval()
    with torch.no_grad():
        feat_q = model(feed_dict_q, None, metadata, is_eval=True)
        feat_k = model(feed_dict_k, None, metadata, is_eval=True)

        pool_e.update(feat_q, feed_dict_q["images"], metadata["scene_number"],None, None)
        pool_g.update(feat_k, feed_dict_k["images"], metadata["scene_number"],None, None)
                
    return


from sklearn.neighbors import NearestNeighbors
def random_retrieve_topk(pool_e, pool_g, imgs_to_view=3):
    print("==> Fitting k-nearest-neighbour model on pool g...")
    knn = NearestNeighbors(n_neighbors=10, metric="cosine")
    knn.fit(torch.stack(pool_g.embeds).cpu())
    
    # select imgs_to_view images from pool q randomly
    query_indices_to_use = random.sample(range(0, pool_e.num), imgs_to_view)
    
    
    figures_gen = []
    print(query_indices_to_use)

    # Perform image retrieval on test images
    print("==> Performing image retrieval on test images...")
    for i, index in enumerate(query_indices_to_use):
        temp = []
        _, indices = knn.kneighbors(torch.reshape(pool_e.embeds[index], (1,-1)).cpu()) # find k nearest train neighbours
        #print(pool_e.scene_nums[index])
        img_query = pool_e.images[index].permute(1,2,0).cpu() # query image
        temp.append(img_query)
        imgs_retrieval = [pool_g.images[idx].permute(1,2,0).cpu() for idx in indices.flatten()]# retrieval images
        temp.extend(imgs_retrieval)
        #print([pool_g.scene_nums[idx].cpu() for idx in indices.flatten()])
        figures_gen.append(temp)
    fig = plot_query_retrieval(figures_gen, None)
        
    return fig
        

def plot_query_retrieval(imgs_retrieval, outFile):
    n_retrieval = len(imgs_retrieval)
    fig = plt.figure(figsize=(20, 4))
    for idx in range(3):
        for im in range(0, 11):
            ax = fig.add_subplot(3, 11, 11*idx+im+1,xticks=[], yticks=[])
            ax.imshow(imgs_retrieval[idx][im])
            if im==0:
                ax.set_title('Query')
            else:
                ax.set_title('Top_'+str(im))
    plt.close(fig)
    return fig
        