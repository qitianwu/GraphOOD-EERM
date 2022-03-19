
# elliptic

python main.py --method base --gnn sage --lr 0.01 --weight_decay 0. --num_layers 5 --dataset elliptic

python main.py --method policy --gnn sage --lr 0.0002 --weight_decay 0. --num_layers 5 --K 3 --T 1 --num_sample 5 --beta 1.0 --lr_a 0.0001 --dataset elliptic --device 1

python main.py --method base --gnn gpr --lr 0.01 --weight_decay 0. --num_layers 5 --dataset elliptic

python main.py --method policy --gnn gpr --lr 0.01 --weight_decay 0. --num_layers 5 --K 3 --T 1 --num_sample 5 --beta 1.0 --lr_a 0.005 --dataset elliptic --device 1