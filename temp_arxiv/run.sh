
# ogb-arxiv

python main.py --gnn sage --lr 0.01 --dataset ogb-arxiv

python main.py --method policy --gnn sage --lr 0.005 --K 5 --T 5 --num_sample 1 --beta 0.5 --lr_a 0.01 --dataset ogb-arxiv --device 1

python main.py --gnn gpr --lr 0.01 --dataset ogb-arxiv

python main.py --method policy --gnn gpr --lr 0.01 --K 3 --T 5 --num_sample 1 --beta 1.0 --lr_a 0.001 --dataset ogb-arxiv --device 0

