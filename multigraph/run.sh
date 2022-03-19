
# Twitch-e
python main.py --method base --gnn gcn --lr 0.01 --weight_decay 1e-3 --num_layers 2 --device 6

python main.py --method policy --gnn gcn --lr 0.01 --weight_decay 1e-3 --num_layers 2 --K 3 --T 1 --num_sample 5 --beta 3.0 --lr_a 0.001 --device 6

python main.py --method base --gnn gat --lr 0.01 --weight_decay 1e-3 --num_layers 2 --device 6

python main.py --method policy --gnn gat --lr 0.01 --weight_decay 1e-3 --num_layers 2 --K 3 --T 1 --num_sample 5 --beta 1.0 --lr_a 0.005 --device 6

python main.py --gnn gcnii --lr 0.01 --weight_decay 1e-3 --num_layers 10 --device 6

python main.py --method policy --gnn gcnii --lr 0.001 --weight_decay 1e-3 --num_layers 10 --K 3 --T 1 --num_sample 5 --beta 1.0 --lr_a 0.01 --device 6

# fb-100
python main.py --method base --gnn gcn --lr 0.01 --weight_decay 1e-3 --num_layers 2 --dataset fb100 --device 0

python main.py --method policy --gnn gcn --lr 0.01 --weight_decay 1e-3 --num_layers 2 --K 3 --T 1 --num_sample 5 --beta 1.0 --lr_a 0.005 --dataset fb100 --device 1