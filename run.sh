
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
               
------------------------------------------------------------------------------------
Exp 16:

Nullified the negative scene embeddings. Now the Negative: untransformed spatial embeddings
and the Postive: transformed spatial embedding

python train.py --batch-size 1 \
                --seed 0 \
                --exp-dir two_obj_spatial_with_scene_and_view_loss_exp16 \
                --epochs 350 \
                --warmup-epoch 350 \
                --lr 0.003 \
                --num-cluster 200 \
                --momentum 0.1 \
                --scene_r 20 \
                --view_r 6 \
                --scene_wt 0.5 \
                --view_wt 0.5 \
                --hyp_N 2 \
                --mode "spatial" \
                --data "/home/mprabhud/dataset/clevr_lang/npys/ab_5t.txt" \
                --use_pretrained "tb_logs/single_obj_exp1/checkpoint.pth.tar" \
                --schedule 250 300
                
=========experiment failed>>>>>>>>>>>

Still the transformation is not working 

https://pasteboard.co/JzZwOea.png

---------------------------------------------------------------------------------------

Exp17:
Check if experiment 15 still works

python train.py --batch-size 1 \
                --seed 0 \
                --exp-dir two_obj_spatial_with_scene_and_view_loss_exp17 \
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
                --schedule 200 300
                
                
https://pasteboard.co/JzZxEK8.png


----------------------------------------------------------------------------------------------------

Exp 18 : (replica of exp9). Negative scene embeddings=16

python train.py --batch-size 1 \
                --seed 0 \
                --exp-dir two_obj_spatial_with_scene_and_view_loss_exp18 \
                --epochs 350 \
                --warmup-epoch 350 \
                --lr 0.003 \
                --num-cluster 200 \
                --scene_r 20 \
                --view_r 70 \
                --scene_wt 0.75 \
                --view_wt 0.25 \
                --hyp_N 2 \
                --mode "spatial" \
                --data "/home/mprabhud/dataset/clevr_lang/npys/ab_5t.txt" \
                --use_pretrained "tb_logs/single_obj_exp1/checkpoint.pth.tar" \
                --schedule 250 300

---------------------------------------------------------------------------------------

Exp 19 : Trying to replicate exp 12 and check if it works. Here I am try to check if the code reformatting from exp 12 to exp 19 still works and it is able to replicate exacts results

negative embeds = 2
python train.py --batch-size 1 \
                --seed 0 \
                --exp-dir two_obj_spatial_with_scene_and_view_loss_exp19 \
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
                
                
-------------------------------------------------------------------------------

Exp 20: CHangeing exp 18 . learning rate and weights. Negative embeds =16. Solution of exp 18 is really unstable checking if it works 
Negative Embeds : 16

python train.py --batch-size 1 \
                --seed 0 \
                --exp-dir two_obj_spatial_with_scene_and_view_loss_exp20 \
                --epochs 350 \
                --warmup-epoch 350 \
                --lr 0.03 \
                --num-cluster 200 \
                --scene_r 20 \
                --view_r 70 \
                --scene_wt 0.85 \
                --view_wt 0.15 \
                --hyp_N 2 \
                --mode "spatial" \
                --data "/home/mprabhud/dataset/clevr_lang/npys/ab_5t.txt" \
                --use_pretrained "tb_logs/single_obj_exp1/checkpoint.pth.tar" \
                --schedule 250 300
                
--------------------------------------------------------------------------------------

Exp 21: Changeing the LR from exp 20 to 0.003 and see how it affects the model
Negative Embeds : 16

python train.py --batch-size 1 \
                --seed 0 \
                --exp-dir two_obj_spatial_with_scene_and_view_loss_exp21 \
                --epochs 350 \
                --warmup-epoch 350 \
                --lr 0.003 \
                --num-cluster 200 \
                --scene_r 20 \
                --view_r 70 \
                --scene_wt 0.85 \
                --view_wt 0.15 \
                --hyp_N 2 \
                --mode "spatial" \
                --data "/home/mprabhud/dataset/clevr_lang/npys/ab_5t.txt" \
                --use_pretrained "tb_logs/single_obj_exp1/checkpoint.pth.tar" \
                --schedule 250 300
                
                
For exp 20 and 21, the solution is really unstable. These experiments are in continuatio of exp 9 
where there was no gradient flow in viewpoitn transformation layers

---------------------------------------------------------------------------
Exp 22:

Decreasing the LR to 0.00003 from exp21 and see how it affects the model
Negative Embeds : 16

python train.py --batch-size 1 \
                --seed 0 \
                --exp-dir two_obj_spatial_with_scene_and_view_loss_exp22 \
                --epochs 350 \
                --warmup-epoch 350 \
                --lr 0.00003 \
                --num-cluster 200 \
                --scene_r 20 \
                --view_r 70 \
                --scene_wt 0.85 \
                --view_wt 0.15 \
                --hyp_N 2 \
                --mode "spatial" \
                --data "/home/mprabhud/dataset/clevr_lang/npys/ab_5t.txt" \
                --use_pretrained "tb_logs/single_obj_exp1/checkpoint.pth.tar" \
                --schedule 250 300
                
------------------------------------------------------------------------------------------------

Exp 23 Only view loss experiment. Trying to judge if I can build up a scenario where I try for the viewpoint transformation
layer purely from the view stand point

Scene Embeddings = 16 + 1 (untransformed one)

python train.py --batch-size 1 \
                --seed 0 \
                --exp-dir two_obj_spatial_with_scene_and_view_loss_exp23 \
                --epochs 350 \
                --warmup-epoch 350 \
                --lr 0.003 \
                --num-cluster 200 \
                --scene_r 20 \
                --view_r 72 \
                --hyp_N 2 \
                --mode "spatial" \
                --data "/home/mprabhud/dataset/clevr_lang/npys/ab_5t.txt" \
                --use_pretrained "tb_logs/single_obj_exp1/checkpoint.pth.tar" \
                --schedule 250 300
                
                
---------------------------------------------------------------------------------------------

Exp 24 Only view loss experiment. Trying to judge if I can build up a scenario where I try for the viewpoint transformation
layer purely from the view stand point

Scene Embeddings =  1 (untransformed one)

python train.py --batch-size 1 \
                --seed 0 \
                --exp-dir two_obj_spatial_with_scene_and_view_loss_exp24 \
                --epochs 350 \
                --warmup-epoch 350 \
                --lr 0.003 \
                --num-cluster 200 \
                --scene_r 20 \
                --view_r 4 \
                --hyp_N 2 \
                --mode "spatial" \
                --data "/home/mprabhud/dataset/clevr_lang/npys/ab_5t.txt" \
                --use_pretrained "tb_logs/single_obj_exp1/checkpoint.pth.tar" \
                --schedule 250 300
                
                
Observation



---------------------------------------------------------------------------------------------------------------
Exp 25 Only view loss experiment. Trying to judge if I can build up a scenario where I try for the viewpoint transformation
layer purely from the view stand point

Nega Scene Embeddings =  1 (transformed one)

python train.py --batch-size 1 \
                --seed 0 \
                --exp-dir two_obj_spatial_with_scene_and_view_loss_exp25 \
                --epochs 350 \
                --warmup-epoch 350 \
                --lr 0.003 \
                --num-cluster 200 \
                --scene_r 20 \
                --view_r 4 \
                --hyp_N 2 \
                --mode "spatial" \
                --data "/home/mprabhud/dataset/clevr_lang/npys/ab_5t.txt" \
                --use_pretrained "tb_logs/single_obj_exp1/checkpoint.pth.tar" \
                --schedule 250 300
                
----------------------------------------------------------------------------------------------------------------------------
Exp 26:

Revamped the loss calculation code. No key encoder now for view loss
removed the torch no grad and detach from the queue
Negative Scene Embeddings =  1 (transformed one)

python train.py --batch-size 1 \
                --seed 0 \
                --exp-dir two_obj_spatial_with_scene_and_view_loss_exp26 \
                --epochs 350 \
                --warmup-epoch 350 \
                --lr 0.003 \
                --num-cluster 200 \
                --scene_r 20 \
                --view_r 4 \
                --hyp_N 2 \
                --mode "spatial" \
                --data "/home/mprabhud/dataset/clevr_lang/npys/ab_5t.txt" \
                --use_pretrained "tb_logs/single_obj_exp1/checkpoint.pth.tar" \
                --schedule 250 300
                
>>>>>>>>>>>>Experiment succesful>>>>>>>>>>>>>>>>>>>
-----------------------------------------------------------------------------------------                
Exp 27:

Revamped the loss calculation code. No key encoder now for view loss
removed the torch no grad and detach from the queue
Negative Scene Embeddings =  1 + 16 (transformed one)

python train.py --batch-size 1 \
                --seed 0 \
                --exp-dir two_obj_spatial_with_scene_and_view_loss_exp27 \
                --epochs 350 \
                --warmup-epoch 350 \
                --lr 0.003 \
                --num-cluster 200 \
                --scene_r 20 \
                --view_r 72 \
                --hyp_N 2 \
                --K 16 \
                --mode "spatial" \
                --data "/home/mprabhud/dataset/clevr_lang/npys/ab_5t.txt" \
                --use_pretrained "tb_logs/single_obj_exp1/checkpoint.pth.tar" \
                --schedule 250 300
                
!!!!Experiment failed : you shooud not start with that much set of negative embeddings

---------------------------------------------------------------------------------------------
Exp28


Revamped the loss calculation code. No key encoder now for view loss
removed the torch no grad and detach from the queue
Negative Scene Embeddings =  1 + 4 (untransformed one)


python train.py --batch-size 1 \
                --seed 0 \
                --exp-dir two_obj_spatial_with_scene_and_view_loss_exp28 \
                --epochs 350 \
                --warmup-epoch 350 \
                --lr 0.003 \
                --num-cluster 200 \
                --scene_r 20 \
                --view_r 24 \
                --hyp_N 2 \
                --K 4 \
                --mode "spatial" \
                --data "/home/mprabhud/dataset/clevr_lang/npys/ab_5t.txt" \
                --use_pretrained "tb_logs/single_obj_exp1/checkpoint.pth.tar" \
                --schedule 250 300
                
                
A lot of False Negative are there 


---------------------------------------------------------------------------------------------

Exp29

Revamped the loss calculation code. No key encoder now for view loss
removed the torch no grad and detach from the queue
Negative Scene Embeddings =  1 + 8 (untransformed one)

python train.py --batch-size 1 \
                --seed 0 \
                --exp-dir two_obj_spatial_with_scene_and_view_loss_exp29 \
                --epochs 350 \
                --warmup-epoch 350 \
                --lr 0.003 \
                --num-cluster 200 \
                --scene_r 20 \
                --view_r 40 \
                --hyp_N 2 \
                --K 8 \
                --mode "spatial" \
                --data "/home/mprabhud/dataset/clevr_lang/npys/ab_5t.txt" \
                --use_pretrained "tb_logs/single_obj_exp1/checkpoint.pth.tar" \
                --schedule 250 300

--------------------------------------------------------------------------------------------------

Exp30

Revamped the loss calculation code. No key encoder now for view loss
removed the torch no grad and detach from the queue
Negative Scene Embeddings =  1 + 8 (untransformed one)

python train.py --batch-size 1 \
                --seed 0 \
                --exp-dir two_obj_spatial_with_scene_and_view_loss_exp30 \
                --epochs 350 \
                --warmup-epoch 350 \
                --lr 0.003 \
                --num-cluster 200 \
                --scene_r 20 \
                --view_r 40 \
                --hyp_N 2 \
                --K 8 \
                --mode "spatial" \
                --data "/home/mprabhud/dataset/clevr_lang/npys/ab_5t.txt" \
                --use_pretrained "tb_logs/single_obj_exp1/checkpoint.pth.tar" \
                --schedule 250 300

-------------------------------------------------------------------------------------------

Exp 31

Revamped the loss calculation code. No key encoder now for view loss
removed the torch no grad and detach from the queue_view
Negative Scene Embeddings =  1 + 8 (untransformed one)
Increase the learning rate to 0.03


python train.py --batch-size 1 \
                --seed 0 \
                --exp-dir two_obj_spatial_with_scene_and_view_loss_exp31 \
                --epochs 350 \
                --warmup-epoch 350 \
                --lr 0.03 \
                --num-cluster 200 \
                --scene_r 20 \
                --view_r 40 \
                --hyp_N 2 \
                --K 8 \
                --mode "spatial" \
                --data "/home/mprabhud/dataset/clevr_lang/npys/ab_5t.txt" \
                --use_pretrained "tb_logs/single_obj_exp1/checkpoint.pth.tar" \
                --schedule 201 350
                
                
>>>>>>>>>>>>>>>>>>> This LR aint working >>> CHoose 0.003

-------------------------------------------------------------------------------------------

Exp 32:

Riding on the success of the exp 30 which was about only the view loss, now adding the scene loss also
removed the torch no grad and detach from the queue_view
Negative Scene Embeddings =  1 + 8 (untransformed one)

l_neg = torch.einsum('nc,ck->nk', [q, self.queue_scene.clone().detach()])
self._dequeue_and_enqueue_scene(k_t.clone().detach())
self._dequeue_and_enqueue_scene(k_o.clone().detach())


python train.py --batch-size 1 \
                --seed 0 \
                --exp-dir two_obj_spatial_with_scene_and_view_loss_exp32 \
                --epochs 500 \
                --warmup-epoch 350 \
                --lr 0.003 \
                --num-cluster 200 \
                --scene_r 50 \
                --view_r 40 \
                --hyp_N 2 \
                --K 8 \
                --mode "spatial" \
                --data "/home/mprabhud/dataset/clevr_lang/npys/ab_5t.txt" \
                --use_pretrained "tb_logs/single_obj_exp1/checkpoint.pth.tar" \
                --scene_wt 0.5 \
                --view_wt 0.5 \
                --schedule 250 400



-------------------------------------------------------------------------------------------------------

Exp 33:

Riding on the success of the exp 30 which was about only the view loss, now adding the scene loss also
removed the torch no grad and detach from the queue_view
Negative Scene Embeddings =  1 + 8 (untransformed one)

Just a variation of Exp32.

self._dequeue_and_enqueue_scene(k_t)
self._dequeue_and_enqueue_scene(k_o)
l_neg = torch.einsum('nc,ck->nk', [q, self.queue_scene.clone().detach()])
@torch.no_grad to queue_scene


python train.py --batch-size 1 \
                --seed 0 \
                --exp-dir two_obj_spatial_with_scene_and_view_loss_exp33 \
                --epochs 500 \
                --warmup-epoch 350 \
                --lr 0.003 \
                --num-cluster 200 \
                --scene_r 50 \
                --view_r 40 \
                --hyp_N 2 \
                --K 8 \
                --mode "spatial" \
                --data "/home/mprabhud/dataset/clevr_lang/npys/ab_5t.txt" \
                --use_pretrained "tb_logs/single_obj_exp1/checkpoint.pth.tar" \
                --scene_wt 0.5 \
                --view_wt 0.5 \
                --schedule 250 400

-------------------------------------------------------------------------------------------------------------------

Exp 34

Riding on the success of the exp 30 which was about only the view loss, now adding the scene loss also
removed the torch no grad and detach from the queue_view
Negative Scene Embeddings =  1 + 8 (untransformed one)

l_neg = torch.einsum('nc,ck->nk', [q, self.queue_scene.clone().detach()])
self._dequeue_and_enqueue_scene(k_t.clone().detach())
self._dequeue_and_enqueue_scene(k_o.clone().detach())

full retraining of the encoder and scene graph -- torch no grad removed


python train.py --batch-size 1 \
                --seed 0 \
                --exp-dir two_obj_spatial_with_scene_and_view_loss_exp34 \
                --epochs 500 \
                --warmup-epoch 350 \
                --lr 0.003 \
                --num-cluster 200 \
                --scene_r 50 \
                --view_r 40 \
                --hyp_N 2 \
                --K 8 \
                --mode "spatial" \
                --data "/home/mprabhud/dataset/clevr_lang/npys/ab_5t.txt" \
                --use_pretrained "tb_logs/single_obj_exp1/checkpoint.pth.tar" \
                --scene_wt 0.5 \
                --view_wt 0.5 \
                --schedule 250 400
                

-------------------------------------------------------------------------------------------------------------------

Exp 35

Riding on the success of the exp 30 which was about only the view loss, now adding the scene loss also
removed the torch no grad and detach from the queue_view
Negative Scene Embeddings =  1 + 8 (untransformed one)

l_neg = torch.einsum('nc,ck->nk', [q, self.queue_scene.clone().detach()])
self._dequeue_and_enqueue_scene(k_t.clone().detach())
self._dequeue_and_enqueue_scene(k_o.clone().detach())

full retraining of the encoder and scene graph -- torch no grad removed

self._dequeue_and_enqueue_scene(k_t.clone().detach()) ---removed



python train.py --batch-size 1 \
                --seed 0 \
                --exp-dir two_obj_spatial_with_scene_and_view_loss_exp35 \
                --epochs 500 \
                --warmup-epoch 350 \
                --lr 0.003 \
                --num-cluster 200 \
                --scene_r 26 \
                --view_r 40 \
                --hyp_N 2 \
                --K 8 \
                --mode "spatial" \
                --data "/home/mprabhud/dataset/clevr_lang/npys/ab_5t.txt" \
                --use_pretrained "tb_logs/single_obj_exp1/checkpoint.pth.tar" \
                --scene_wt 0.5 \
                --view_wt 0.5 \
                --schedule 250 400
                
-------------------------------------------------------------------------------------------

Exp 35-2: Training on 10 scenes

Riding on the success of the exp 30 which was about only the view loss, now adding the scene loss also
removed the torch no grad and detach from the queue_view
Negative Scene Embeddings =  1 + 16 (untransformed one)

Just a variation of Exp33.

removed ----> self._dequeue_and_enqueue_scene(k_t)
configure ---->
self._dequeue_and_enqueue_scene(k_o)
l_neg = torch.einsum('nc,ck->nk', [q, self.queue_scene.clone().detach()])
@torch.no_grad to queue_scene



python train.py --batch-size 1 \
                --seed 0 \
                --exp-dir two_obj_spatial_with_scene_and_view_loss_exp35_2 \
                --epochs 350 \
                --warmup-epoch 350 \
                --lr 0.003 \
                --num-cluster 200 \
                --scene_r 26 \
                --view_r 40 \
                --hyp_N 2 \
                --K 8 \
                --mode "spatial" \
                --data "/home/mprabhud/dataset/clevr_lang/npys/ab_5t.txt" \
                --use_pretrained "tb_logs/single_obj_exp1/checkpoint.pth.tar" \
                --scene_wt 0.5 \
                --view_wt 0.5 \
                --schedule 250 400
                
-------------------------------------------------------------------------------------------------------------------

Exp 36 - Full dataset training 

Riding on the success of the exp 30 which was about only the view loss, now adding the scene loss also
removed the torch no grad and detach from the queue_view
Negative Scene Embeddings =  1 + 16 (untransformed one)

l_neg = torch.einsum('nc,ck->nk', [q, self.queue_scene.clone().detach()])
self._dequeue_and_enqueue_scene(k_t.clone().detach())
self._dequeue_and_enqueue_scene(k_o.clone().detach())

full retraining of the encoder and scene graph -- torch no grad removed

self._dequeue_and_enqueue_scene(k_t.clone().detach()) ---removed

Full dataset training with scene_r = 60 (15)

--> screen 3

python train.py --batch-size 1 \
                --seed 0 \
                --exp-dir two_obj_spatial_with_scene_and_view_loss_exp36 \
                --epochs 500 \
                --warmup-epoch 500 \
                --lr 0.003 \
                --num-cluster 200 \
                --scene_r 60 \
                --view_r 40 \
                --hyp_N 2 \
                --K 8 \
                --mode "spatial" \
                --data "/home/mprabhud/dataset/clevr_lang/npys/ab_5t.txt" \
                --use_pretrained "tb_logs/single_obj_exp1/checkpoint.pth.tar" \
                --scene_wt 0.5 \
                --view_wt 0.5 \
                --schedule 300 450

-----------------------------------------------------------------------------------------------------------------------------------------------
-----------------------------------------------------------------------------------------------------------------------------------------------
Bug spotted .. experiment 35, 35_2 and 36 needs to be repeated now
----------------------------------------------------------------------------------------------------------------------------------------------
--------------------------------------------------------------------------------------------------------------------------------------------------


Exp 37

--> extension of exp 34 with more views_r and checking if training stablizes

--> model file of exp34

--> screen 4

python train.py --batch-size 1 \
                --seed 0 \
                --exp-dir two_obj_spatial_with_scene_and_view_loss_exp37 \
                --epochs 500 \
                --warmup-epoch 350 \
                --lr 0.003 \
                --num-cluster 200 \
                --scene_r 50 \
                --view_r 72 \
                --hyp_N 2 \
                --K 8 \
                --mode "spatial" \
                --data "/home/mprabhud/dataset/clevr_lang/npys/ab_5t.txt" \
                --use_pretrained "tb_logs/single_obj_exp1/checkpoint.pth.tar" \
                --scene_wt 0.5 \
                --view_wt 0.5 \
                --schedule 250 400

------------------------------------------------------------------------------------------------------------------------------------------------------------------

Exp 38

Negative Scene Embeddings =  1 + 16 (untransformed one)

Just a variation of Exp35



-->full retraining of the encoder and scene graph -- torch no grad removed

---> model file of exp35

---> screen 2

python train.py --batch-size 1 \
                --seed 0 \
                --exp-dir two_obj_spatial_with_scene_and_view_loss_exp38 \
                --epochs 500 \
                --warmup-epoch 350 \
                --lr 0.003 \
                --num-cluster 200 \
                --scene_r 26 \
                --view_r 72 \
                --hyp_N 2 \
                --K 8 \
                --mode "spatial" \
                --data "/home/mprabhud/dataset/clevr_lang/npys/ab_5t.txt" \
                --use_pretrained "tb_logs/single_obj_exp1/checkpoint.pth.tar" \
                --scene_wt 0.5 \
                --view_wt 0.5 \
                --schedule 250 400
                
--------------------------------------------------------------------------------------------------------------------------------------------------
Exp 37 and 38 arefailure, the scene loss curves do not converge. Training does not stabilise. Adding all the views as negatives destablises the training 
--------------------------------------------------------------------------------------------------------------------------------------------------

-------------------------------------------------------------------------------------------------------------------------------------------------
Exp 39 and exp 40

--> extension of exp 34 with more views_r and checking if training stablizes

--> model file of exp34

--> screen 2

--> negatives = 7x2

--> balance out scn

---> in exp 34 view_r is 40 and scene_r is 50, now balancing out them --> 

----> we have both the k_t and k_o in negatives

----> in view forward, k_o is enqued towards the end 

Extenstion of exp 34-->
python train.py --batch-size 1 \
                --seed 0 \
                --exp-dir two_obj_spatial_with_scene_and_view_loss_exp39 \
                --epochs 500 \
                --warmup-epoch 350 \
                --lr 0.003 \
                --num-cluster 200 \
                --scene_r 56 \
                --view_r 48 \
                --hyp_N 2 \
                --K 8 \
                --mode "spatial" \
                --data "/home/mprabhud/dataset/clevr_lang/npys/ab_5t.txt" \
                --use_pretrained "tb_logs/single_obj_exp1/checkpoint.pth.tar" \
                --scene_wt 0.5 \
                --view_wt 0.5 \
                --schedule 250 400


Extension of exp 35-->

---> negatives = 7

---> in exp 35 view_r is 40 and scene_r is 50, 

----> we have  only k_o in scene negatives

----> in view forward, k_o is enqued towards the end

--> screen 4

python train.py --batch-size 1 \
                --seed 0 \
                --exp-dir two_obj_spatial_with_scene_and_view_loss_exp40 \
                --epochs 500 \
                --warmup-epoch 350 \
                --lr 0.003 \
                --num-cluster 200 \
                --scene_r 28 \
                --view_r 48 \
                --hyp_N 2 \
                --K 8 \
                --mode "spatial" \
                --data "/home/mprabhud/dataset/clevr_lang/npys/ab_5t.txt" \
                --use_pretrained "tb_logs/single_obj_exp1/checkpoint.pth.tar" \
                --scene_wt 0.5 \
                --view_wt 0.5 \
                --schedule 250 400
                
--------------------------------------------------------------------------------------------------------------------------------------------

xxxxxxxxxxxx---- failed experiment -------------------xxxxxxxxxxxxxx

Exp 41 -- taking model and experiment configs from exp34 and exp 39

screen 5

python train.py --batch-size 1 \
                --seed 0 \
                --exp-dir two_obj_spatial_with_scene_and_view_loss_exp41 \
                --epochs 500 \
                --warmup-epoch 350 \
                --lr 0.003 \
                --num-cluster 200 \
                --scene_r 56 \
                --view_r 48 \
                --hyp_N 2 \
                --K 8 \
                --mode "spatial" \
                --data "/home/mprabhud/dataset/clevr_lang/npys/ab_5t.txt" \
                --use_pretrained "tb_logs/single_obj_exp1/checkpoint.pth.tar" \
                --scene_wt 0.5 \
                --view_wt 0.5 \
                --schedule 250 400
     
xxxxxxxxxxxx---- failed experiment -------------------xxxxxxxxxxxxxx

-------------------------------------------------------------------------------------------------------
Exp 42 - model from exp39

Extension of experiment 39,decreasing scene_r -- a feasible value looks to be 40% of the dataset.. also this training needs to run a lot longer

screen 1

python train.py --batch-size 1 \
                --seed 0 \
                --exp-dir two_obj_spatial_with_scene_and_view_loss_exp42 \
                --epochs 600 \
                --warmup-epoch 601 \
                --lr 0.003 \
                --num-cluster 200 \
                --scene_r 32 \
                --view_r 48 \
                --hyp_N 2 \
                --K 11 \
                --mode "spatial" \
                --data "/home/mprabhud/dataset/clevr_lang/npys/ab_5t.txt" \
                --use_pretrained "tb_logs/single_obj_exp1/checkpoint.pth.tar" \
                --scene_wt 0.5 \
                --view_wt 0.5 \
                --schedule 450
                
                
----------------------------------------------------------------------------------------------------------------------
exp43_continued

python train.py --batch-size 1 \
                --seed 0 \
                --exp-dir two_obj_spatial_with_scene_and_view_loss_exp43 \
                --epochs 900 \
                --warmup-epoch 901 \
                --lr 0.003 \
                --num-cluster 200 \
                --scene_r 552 \
                --view_r 48 \
                --hyp_N 2 \
                --K 11 \
                --mode "spatial" \
                --data "/home/mprabhud/dataset/clevr_lang/npys/ab_5t.txt" \
                --resume "tb_logs/two_obj_spatial_with_scene_and_view_loss_exp43/checkpoint_599.pth.tar" \
                --scene_wt 0.5 \
                --view_wt 0.5 \
                --schedule 450



--------------------------------------------------------------------------------------------------------------

Exp 43 -- model from exp39 full training of the dataset with scenes negatives as 30% of the dataset (230*0.3*8)

Extension of experiment 36 with model configs from exp 42 and 39

python train.py --batch-size 1 \
                --seed 0 \
                --exp-dir two_obj_spatial_with_scene_and_view_loss_exp43 \
                --epochs 600 \
                --warmup-epoch 601 \
                --lr 0.003 \
                --num-cluster 200 \
                --scene_r 552 \
                --view_r 48 \
                --hyp_N 2 \
                --K 11 \
                --mode "spatial" \
                --data "/home/mprabhud/dataset/clevr_lang/npys/ab_5t.txt" \
                --use_pretrained "tb_logs/single_obj_exp1/checkpoint.pth.tar" \
                --scene_wt 0.5 \
                --view_wt 0.5 \
                --schedule 450
---------------------------------------------------------------------------
Exp 44

batched_training_test for first 20 scenes. Extension of exp42 (and exp 39)


python train.py --batch_size 4 \
                --seed 0 \
                --exp-dir two_obj_spatial_with_scene_and_view_loss_exp44 \
                --epochs 600 \
                --warmup-epoch 601 \
                --lr 0.003 \
                --num-cluster 200 \
                --scene_r 32 \
                --view_r 48 \
                --hyp_N 2 \
                --K 11 \
                --mode "spatial" \
                --data "/home/mprabhud/dataset/clevr_lang/npys/ab_5t.txt" \
                --use_pretrained "tb_logs/single_obj_exp1/checkpoint.pth.tar" \
                --scene_wt 0.5 \
                --view_wt 0.5 \
                --schedule 450
                
------------------------------------------------------------------------------
Exp 45

Batched training of the full dataset. Extension of exp 39, 42 and 43 with reduced scene_r at 20% of the dataset


python train.py --batch_size 4 \
                --seed 0 \
                --exp-dir two_obj_spatial_with_scene_and_view_loss_exp45 \
                --epochs 800 \
                --warmup-epoch 801 \
                --lr 0.003 \
                --num-cluster 200 \
                --scene_r 368 \
                --view_r 48 \
                --hyp_N 2 \
                --K 11 \
                --mode "spatial" \
                --data "/home/mprabhud/dataset/clevr_lang/npys/ab_5t.txt" \
                --use_pretrained "tb_logs/single_obj_exp1/checkpoint.pth.tar" \
                --scene_wt 0.5 \
                --view_wt 0.5 \
                --schedule 600
                
>>>> Even though the loss curve are comparable to the training routines with batch_size, it was not able to attain the same levels of accuracy

----------------------------------------------------------------------------------------------------------------------------------

Exp 46:

Batched training with bs=4 of full dataset. Learning rate is increased by factor of 4.

python train.py --batch_size 4 \
                --seed 0 \
                --exp-dir two_obj_spatial_with_scene_and_view_loss_exp46 \
                --epochs 800 \
                --warmup-epoch 801 \
                --lr 0.012 \
                --num-cluster 200 \
                --scene_r 368 \
                --view_r 48 \
                --hyp_N 2 \
                --K 11 \
                --mode "spatial" \
                --data "/home/mprabhud/dataset/clevr_lang/npys/ab_5t.txt" \
                --use_pretrained "tb_logs/single_obj_exp1/checkpoint.pth.tar" \
                --scene_wt 0.5 \
                --view_wt 0.5 \
                --schedule 600
                
-------------------------------------------------------------------------------------------------------------------------------------                
Exp 47:

Continuation of exp 46, lr=0.012. Scene_r decreased to 120 (6% of the dataset)

python train.py --batch_size 4 \
                --seed 0 \
                --exp-dir two_obj_spatial_with_scene_and_view_loss_exp47 \
                --epochs 800 \
                --warmup-epoch 801 \
                --lr 0.012 \
                --num-cluster 200 \
                --scene_r 120 \
                --view_r 48 \
                --hyp_N 2 \
                --K 11 \
                --mode "spatial" \
                --data "/home/mprabhud/dataset/clevr_lang/npys/ab_5t.txt" \
                --use_pretrained "tb_logs/single_obj_exp1/checkpoint.pth.tar" \
                --scene_wt 0.5 \
                --view_wt 0.5 \
                --schedule 600

-------------------------------------------------------------------------------------------------------------------------------------


Exp 48 (exp similar to exp 35 and 36 (good results))

Continuation of exp 47, lr=0.012. Scene_r decreased to 60 (6% of the dataset)
self._dequeue_and_enqueue_scene(k_t.clone().detach()) --removed

Verifying the fact that having batched training has same almost similar effects with single batch. (Exp similar to ). If it does not, I think having batched training I will remove


---- training running on screen 1

python train.py --batch_size 4 \
                --seed 0 \
                --exp-dir two_obj_spatial_with_scene_and_view_loss_exp48 \
                --epochs 800 \
                --warmup-epoch 801 \
                --lr 0.012 \
                --num-cluster 200 \
                --scene_r 60 \
                --view_r 48 \
                --hyp_N 2 \
                --K 11 \
                --mode "spatial" \
                --data "/home/mprabhud/dataset/clevr_lang/npys/ab_5t.txt" \
                --use_pretrained "tb_logs/single_obj_exp1/checkpoint.pth.tar" \
                --scene_wt 0.5 \
                --view_wt 0.5 \
                --schedule 800
                
                
------------------------------------------------------------------------------------------------------------------------

Exp 47_continued

python train.py --batch_size 4 \
                --seed 0 \
                --exp-dir two_obj_spatial_with_scene_and_view_loss_exp47_continued \
                --epochs 800 \
                --warmup-epoch 801 \
                --lr 0.012 \
                --num-cluster 200 \
                --scene_r 120 \
                --view_r 48 \
                --hyp_N 2 \
                --K 11 \
                --mode "spatial" \
                --data "/home/mprabhud/dataset/clevr_lang/npys/ab_5t.txt" \
                --resume "tb_logs/two_obj_spatial_with_scene_and_view_loss_exp47/checkpoint_599.pth.tar" \
                --scene_wt 0.5 \
                --view_wt 0.5 \
                --schedule 801
                
                
--------------------------------------------------------------------------------------

Node Exp 1: 

python train.py --batch_size 4 \
                --seed 0 \
                --exp-dir two_obj_node_exp1 \
                --epochs 300 \
                --warmup-epoch 101 \
                --num-cluster 120 \
                --lr 0.003 \
                --moco_r 16 \
                --hyp_N 2 \
                --mode "node" \
                --data "/home/mprabhud/dataset/clevr_lang/npys/ab_5t.txt"