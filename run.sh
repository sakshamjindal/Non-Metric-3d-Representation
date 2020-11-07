python train.py --batch-size 5 \
                --seed 0 \
                --exp-dir single_obj_exp1 \
                --warmup-epoch 10 \
                --num-cluster 120 \
                --moco_r 40 \
                --hyp_N 1 \
                --mode "node" \
                --data "/home/mprabhud/dataset/clevr_lang/npys/aa_5t.txt"
                
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