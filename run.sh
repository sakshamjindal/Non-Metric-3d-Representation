python train.py --batch-size 5 \
                --seed 0 \
                --exp-dir single_obj_exp1 \
                --warmup-epoch 1 \
                --num-cluster 120 \
                --moco_r 40 \
                --hyp_N 2 \
                --mode "node" \
                --data "/home/mprabhud/dataset/clevr_lang/npys/ab_5t.txt"
                
# python train.py --batch-size 5 \
#                 --seed 0 \
#                 --exp-dir two_obj_exp1 \
#                 --warmup-epoch 100 \
#                 --num-cluster 200 \
#                 --moco_r 100 \
#                 --hyp_N 2 \
#                 --mode "spatial" \
#                 --data "/home/mprabhud/dataset/clevr_lang/npys/ab_5t.txt"
                
                
# python train.py --batch-size 5 \
#                 --seed 0 \
#                 --exp-dir two_obj_exp2_pretrained_on_node \
#                 --warmup-epoch 100 \
#                 --num-cluster 200 \
#                 --moco_r 100 \
#                 --hyp_N 2 \
#                 --mode "spatial" \
#                 --data "/home/mprabhud/dataset/clevr_lang/npys/ab_5t.txt" \
#                 --use_pretrained "tb_logs/single_obj_exp1/checkpoint.pth.tar"