
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
                --warmup-epoch 150 \
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
                
                
----------------------------------------------------------------------------------------------------------------
Exp 5

python train.py --batch-size 1 \
                --seed 0 \
                --exp-dir two_obj_spatial_with_scene_and_view_loss_exp5 \
                --epochs 350 \
                --warmup-epoch 120 \
                --num-cluster 200 \
                --scene_r 20 \
                --view_r 65 \
                --scene_wt 0.7 \
                --view_wt 0.3 \
                --hyp_N 2 \
                --mode "spatial" \
                --data "/home/mprabhud/dataset/clevr_lang/npys/ab_5t.txt" \
                --use_pretrained "tb_logs/single_obj_exp1/checkpoint.pth.tar"

----------------------------------------------------------------------------------------------------------------
Exp 6

Warmig up the scene model for 70 epochs and then adding the view loss

python train.py --batch-size 1 \
                --seed 0 \
                --exp-dir two_obj_spatial_with_scene_and_view_loss_exp6 \
                --epochs 350 \
                --warmup-epoch 200 \
                --num-cluster 200 \
                --scene_r 20 \
                --view_r 65 \
                --scene_wt 0.2 \
                --view_wt 0.8 \
                --hyp_N 2 \
                --mode "spatial" \
                --data "/home/mprabhud/dataset/clevr_lang/npys/ab_5t.txt" \
                --use_pretrained "tb_logs/single_obj_exp1/checkpoint.pth.tar"
                
                
                
-------------------------------------------------------------------------------------------------------------------

Exp 7
python train.py --batch-size 1 \
                --seed 0 \
                --exp-dir two_obj_spatial_with_scene_and_view_loss_exp7 \
                --epochs 350 \
                --warmup-epoch 120 \
                --num-cluster 200 \
                --scene_r 20 \
                --view_r 65 \
                --scene_wt 0.15 \
                --view_wt 0.85 \
                --hyp_N 2 \
                --mode "spatial" \
                --data "/home/mprabhud/dataset/clevr_lang/npys/ab_5t.txt" \
                --use_pretrained "tb_logs/single_obj_exp1/checkpoint.pth.tar"
                
------------------------------------------------------------------------------------------------------------
Exp 8 -- train for longer epochs without introducing PCL loss

python train.py --batch-size 1 \
                --seed 0 \
                --exp-dir two_obj_spatial_with_scene_and_view_loss_exp8 \
                --epochs 350 \
                --warmup-epoch 350 \
                --num-cluster 200 \
                --scene_r 20 \
                --view_r 65 \
                --scene_wt 0.7 \
                --view_wt 0.3 \
                --hyp_N 2 \
                --mode "spatial" \
                --data "/home/mprabhud/dataset/clevr_lang/npys/ab_5t.txt" \
                --use_pretrained "tb_logs/single_obj_exp1/checkpoint.pth.tar"
                
                
-------------------------------------------------------------------------------------------------------------------
Exp 9 -- train

python train.py --batch-size 1 \
                --seed 0 \
                --exp-dir two_obj_spatial_with_scene_and_view_loss_exp9 \
                --epochs 350 \
                --warmup-epoch 350 \
                --num-cluster 200 \
                --scene_r 20 \
                --view_r 65 \
                --scene_wt 0.15 \
                --view_wt 0.85 \
                --hyp_N 2 \
                --mode "spatial" \
                --data "/home/mprabhud/dataset/clevr_lang/npys/ab_5t.txt" \
                --use_pretrained "tb_logs/single_obj_exp1/checkpoint.pth.tar"
                
                
                
                
-----------------------------------------------------------------------------------
Exp 11
Fresh training after debugging/ Negative embeddings = 2


python train.py --batch-size 1 \
                --seed 0 \
                --exp-dir two_obj_spatial_with_scene_and_view_loss_exp11 \
                --epochs 350 \
                --warmup-epoch 350 \
                --num-cluster 200 \
                --scene_r 20 \
                --view_r 12 \
                --scene_wt 0.5 \
                --view_wt 0.5 \
                --hyp_N 2 \
                --mode "spatial" \
                --data "/home/mprabhud/dataset/clevr_lang/npys/ab_5t.txt" \
                --use_pretrained "tb_logs/single_obj_exp1/checkpoint.pth.tar"
                
                
--------------------------------------------------------------------------------------
Exp 12
added tb weight histograms
Negative embeddings = 2


python train.py --batch-size 1 \
                --seed 0 \
                --exp-dir two_obj_spatial_with_scene_and_view_loss_exp12 \
                --epochs 350 \
                --warmup-epoch 350 \
                --num-cluster 200 \
                --scene_r 20 \
                --view_r 12 \
                --scene_wt 0.5 \
                --view_wt 0.5 \
                --hyp_N 2 \
                --mode "spatial" \
                --data "/home/mprabhud/dataset/clevr_lang/npys/ab_5t.txt" \
                --use_pretrained "tb_logs/single_obj_exp1/checkpoint.pth.tar"
                
                
                
------------------------------------------------------------------------------------
Exp 13
added tb weight histograms
Negative embeddings = 2
Step LR -- decrease LR after every 20 epcochs . Initial LR = 0.03

python train.py --batch-size 1 \
                --seed 0 \
                --exp-dir two_obj_spatial_with_scene_and_view_loss_exp13 \
                --epochs 350 \
                --warmup-epoch 350 \
                --lr 0.03 \
                --num-cluster 200 \
                --scene_r 20 \
                --view_r 12 \
                --scene_wt 0.5 \
                --view_wt 0.5 \
                --hyp_N 2 \
                --mode "spatial" \
                --data "/home/mprabhud/dataset/clevr_lang/npys/ab_5t.txt" \
                --use_pretrained "tb_logs/single_obj_exp1/checkpoint.pth.tar" \
                --schedule 20 40 60 80 100 120 140 \
                
------------------------------------------------------------------------------------
Exp 14
added tb weight histograms
Negative embeddings = 2
Step LR -- decrease LR after every 20 epcochs . Initial LR = 0.03
Momentum decreased  to 0.1 from 0.9

python train.py --batch-size 1 \
                --seed 0 \
                --exp-dir two_obj_spatial_with_scene_and_view_loss_exp14 \
                --epochs 350 \
                --warmup-epoch 350 \
                --lr 0.03 \
                --num-cluster 200 \
                --momentum 0.1 \
                --scene_r 20 \
                --view_r 12 \
                --scene_wt 0.5 \
                --view_wt 0.5 \
                --hyp_N 2 \
                --mode "spatial" \
                --data "/home/mprabhud/dataset/clevr_lang/npys/ab_5t.txt" \
                --use_pretrained "tb_logs/single_obj_exp1/checkpoint.pth.tar" \
                --schedule 20 40 60 80 100 120 140 \
                
------------------------------------------------------------------------------------
Exp 15
added tb weight histograms
Negative embeddings = 2
Step LR -- decrease LR after 200 epchocs
Initial LR - 0.003

python train.py --batch-size 1 \
                --seed 0 \
                --exp-dir two_obj_spatial_with_scene_and_view_loss_exp15 \
                --epochs 350 \
                --warmup-epoch 350 \
                --lr 0.003 \
                --num-cluster 200 \
                --momentum 0.1 \
                --scene_r 20 \
                --view_r 12 \
                --scene_wt 0.5 \
                --view_wt 0.5 \
                --hyp_N 2 \
                --mode "spatial" \
                --data "/home/mprabhud/dataset/clevr_lang/npys/ab_5t.txt" \
                --use_pretrained "tb_logs/single_obj_exp1/checkpoint.pth.tar" \
                --schedule 200 300 \