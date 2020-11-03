from sklearn.neighbors import NearestNeighbors
import torch
import torch.nn as nn

def pair_embeddings(output_k, output_q, mode = "node"):
    
    if mode=="node":
        mode = 0
    elif mode=="spatial":
        mode = 1
    else:
        raise ValueError("Mode should be either node or spatial")
    
    num_batch = len(output_k)
    assert num_batch==len(output_q)   
    
    output_q_rearrange = []
    
    for batch_ind in range(num_batch):
        
        num_obj_in_batch = output_k[batch_ind].shape[0]
        assert num_obj_in_batch==output_q[batch_ind].shape[0]
        
        #flatten the features - 
        output_k[batch_ind] = output_k[batch_ind].view(-1,256)
        output_q[batch_ind] = output_q[batch_ind].view(-1,256)
        
        #form two pool for nearest neighbour search
        pool_e = output_k[batch_ind].clone().detach().cpu()
        pool_g = output_q[batch_ind].clone().detach().cpu()

        with torch.no_grad():

            knn_e = NearestNeighbors(n_neighbors= num_obj_in_batch, metric="euclidean")
            knn_g = NearestNeighbors(n_neighbors= num_obj_in_batch, metric="euclidean")

            knn_g.fit(pool_g)
            knn_e.fit(pool_e)
            
            paired = []
            pairs = []
            for index in range(num_obj_in_batch):  

                #fit knn on each of the object 
                _, indices_e = knn_g.kneighbors(torch.reshape(pool_e[index], (1,-1)).detach().cpu())
                indices_e = list(indices_e.flatten())
                for e in indices_e:
                    if e not in paired:
                        paired.append(e)
                        pairs.append([index,e])
                        break
        
        print(pairs)
        #rearranging the matched in output_q based on pair formed
        pool_g_rearranged = torch.zeros(pool_e.shape[0], 256)
        for pair in pairs:
            pool_g_rearranged[pair[0]] = output_q[batch_ind][pair[1]]
        
        output_q[batch_ind] = pool_g_rearranged.cuda()
        
    return output_k, output_q

def stack_features_across_batch(feature_list):

    num_batch = len(feature_list)
    features = feature_list[0].view(-1,256)
    
    for num in range(1,num_batch):
        features = torch.cat([features, feature_list[num]], dim =0)
        
    return features

