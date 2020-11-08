
python train.py --batch-size 5 \
                --seed 0 \
                --exp-dir single_obj_exp1 \
                --warmup-epoch 10 \
                --num-cluster 120 \
                --moco_r 40 \
                --hyp_N 1 \
                --mode "node" \
                --data "/home/mprabhud/dataset/clevr_lang/npys/aa_5t.txt"
-----------------------------------------------------------------------------------------------------

python train.py --batch-size 5 \
                --seed 0 \
                --exp-dir two_obj_spatial_without_pretrained \
                --epochs 350 \
                --warmup-epoch 150 \
                --num-cluster 200 \
                --moco_r 100 \
                --hyp_N 2 \
                --mode "spatial" \
                --data "/home/mprabhud/dataset/clevr_lang/npys/ab_5t.txt"
                --use_pretrained
------------------------------------------------------------------------------------------------------                
                
python train.py --batch-size 5 \
                --seed 0 \
                --exp-dir two_obj_spatial_with_node_pretrained \
                --epochs 350 \
                --warmup-epoch 150 \
                --num-cluster 200 \
                --moco_r 100 \
                --hyp_N 2 \
                --mode "spatial" \
                --data "/home/mprabhud/dataset/clevr_lang/npys/ab_5t.txt" \
                --use_pretrained "tb_logs/single_obj_exp1/checkpoint.pth.tar"
                

---------------------------------------------------------------------------------------------------------

Exp Name: two_obj_spatial_with_scene_and_view_loss_exp1
Mode: Spatial
Pretrained : Nodes
Losses : Scene + View
Number of Scene = 10

python train.py --batch-size 1 \
                --seed 0 \
                --exp-dir two_obj_spatial_with_scene_and_view_loss_exp1 \
                --epochs 350 \
                --warmup-epoch 120 \
                --num-cluster 200 \
                --scene_r 20 \
                --view_r 65 \
                --scene_wt 0.4 \
                --view_wt 0.6 \
                --hyp_N 2 \
                --mode "spatial" \
                --data "/home/mprabhud/dataset/clevr_lang/npys/ab_5t.txt" \
                --use_pretrained "tb_logs/single_obj_exp1/checkpoint.pth.tar"
                
---------------------------------------------------------------------------------------------------------

Exp Name: two_obj_spatial_with_scene_and_view_loss_sans_node_exp2
Mode: Spatial
Pretrained : Nodes
Losses : Scene + View
Number of Scene = 10
Removed Node features from the scene embeddings

python train.py --batch-size 1 \
                --seed 0 \
                --exp-dir two_obj_spatial_with_scene_and_view_loss_exp2 \
                --epochs 350 \
                --warmup-epoch 120 \
                --num-cluster 200 \
                --scene_r 20 \
                --view_r 65 \
                --scene_wt 0.4 \
                --view_wt 0.6 \
                --hyp_N 2 \
                --mode "spatial" \
                --data "/home/mprabhud/dataset/clevr_lang/npys/ab_5t.txt" \
#                 --use_pretrained "tb_logs/single_obj_exp1/checkpoint.pth.tar"






--------------------------------------------------------------------------------------------------------------


Exp Name: two_obj_spatial_with_scene_and_view_loss_sans_node_exp3
Mode: Spatial
Pretrained : Nodes
Losses : Scene + View
Number of Scene = 10
Removed Node features from the scene embeddings
Weight of the view loss decreased

python train.py --batch-size 1 \
                --seed 0 \
                --exp-dir two_obj_spatial_with_scene_and_view_loss_exp3 \
                --epochs 350 \
                --warmup-epoch 120 \
                --num-cluster 200 \
                --scene_r 20 \
                --view_r 65 \
                --scene_wt 0.8 \
                --view_wt 0.2 \
                --hyp_N 2 \
                --mode "spatial" \
                --data "/home/mprabhud/dataset/clevr_lang/npys/ab_5t.txt" \
#                 --use_pretrained "tb_logs/single_obj_exp1/checkpoint.pth.tar"

--------------------------------------------------------------------------------------------------------------


Exp Name: two_obj_spatial_with_scene_and_view_loss_sans_node_exp4
Mode: Spatial
Pretrained : Nodes
Losses : Scene + View
Number of Scene = 10
Weight of the view loss decreased
Pretrained Node features used

python train.py --batch-size 1 \
                --seed 0 \
                --exp-dir two_obj_spatial_with_scene_and_view_loss_exp4 \
                --epochs 350 \
                --warmup-epoch 120 \
                --num-cluster 200 \
                --scene_r 20 \
                --view_r 65 \
                --scene_wt 0.8 \
                --view_wt 0.2 \
                --hyp_N 2 \
                --mode "spatial" \
                --data "/home/mprabhud/dataset/clevr_lang/npys/ab_5t.txt" \
                --use_pretrained "tb_logs/single_obj_exp1/checkpoint.pth.tar"

