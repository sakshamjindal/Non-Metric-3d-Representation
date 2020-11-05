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
        
        num_obj_in_batch = output_k[batch_ind][0].shape[0]
        assert num_obj_in_batch==output_q[batch_ind][0].shape[0]
        
        if mode=="spatial":
            assert num_obj_in_batch==output_q[batch_ind][1].shape[0]
            assert num_obj_in_batch==output_q[batch_ind][1].shape[1]
            assert output_k[batch_ind][1].shape[0]==output_k[batch_ind][1].shape[0]
            assert output_k[batch_ind][1].shape[1]==output_k[batch_ind][1].shape[1]
            assert output_k[batch_ind][1].shape[0]==output_k[batch_ind][1].shape[1]
            assert output_k[batch_ind][1].shape[1]==output_k[batch_ind][1].shape[0]
            
        #flatten the node features only - 
        output_k[batch_ind][0] = output_k[batch_ind][0].view(-1,256)
        output_q[batch_ind][0] = output_q[batch_ind][0].view(-1,256)
        
        
        #form two pool from node features for nearest neighbour search
        pool_e = output_k[batch_ind][0].clone().detach().cpu()
        pool_g = output_q[batch_ind][0].clone().detach().cpu()

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
                        pairs.append(e)
                        break
        
        print(pairs)
        #rearranging the matched in output_q based on pair formed
        
    
        #Rearranging the node_features in output_q based on pair formed
        assert num_obj_in_batch == len(pairs)
        
        node_pool_rearranged = torch.zeros(pool_e.shape[0], 256)
        for index_node in range(num_obj_in_batch):
            pair_mapping_obj = pairs[index_node]
            node_pool_rearranged[index_node] = output_q[batch_ind][0][pair_mapping_obj].clone()
        
        output_q[batch_ind][0] = node_pool_rearranged.cuda()
        
        #If mode is spatial : also repair the spatial embeddings
        if mode=="spatial":
            spatial_pool_rearranged = torch.zeros(pool_e.shape[0], pool_e.shape[0], 256)
            for index_subj in range(num_obj_in_batch):
                for index_obj in range(num_obj_in_batch):
                    pair_mapping_subj = pairs[index_subj]
                    pair_mapping_obj = pairs[index_obj]
                    spatial_pool_rearranged[index_subj][index_obj] = output_q[batch_ind][1][pair_mapping_subj][pair_mapping_obj].clone()
                    
            output_q[batch_ind][1] = spatial_pool_rearranged
        
    return output_k, output_q

def stack_features_across_batch(output_feature_list, mode="node"):

    num_batch = len(output_feature_list)
    if mode=="node":  
        node_features = output_feature_list[0][0].view(-1,256)

        for num in range(1,num_batch):
            node_features = torch.cat([node_features, output_feature_list[num][0]], dim =0)
        
        return node_features
    
    if mode=="spatial":
        spatial_features = output_feature_list[0][1].view(-1,256)

        for num in range(1, num_batch):
            spatial_features = torch.cat([spatial_features, output_feature_list[num][1].view(-1,256)], dim =0)
            
        return spatial_features
    
    raise ValueError("Training mode not defined properly. It should be either 'node' or 'spatial'." )      

