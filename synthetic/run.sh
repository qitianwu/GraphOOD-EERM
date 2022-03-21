
# cora
python main.py --method erm --dataset cora --gnn gcn --run 20 --lr 0.001 --device 1

python main.py --method eerm --gnn gcn --lr 0.01 --weight_decay 1e-3 --num_layers 2 --K 3 --T 1 --num_sample 5 --beta 2.0 --lr_a 0.005 --device 1 --dataset cora --run 20 --device 1

# amazon-photo
python main.py --method erm --dataset amazon-photo --gnn gcn --run 20 --lr 0.001 --device 1

python main.py --method eerm --gnn gcn --lr 0.01 --weight_decay 1e-3 --num_layers 2 --K 3 --T 1 --num_sample 5 --beta 2.0 --lr_a 0.005 --device 1 --dataset amazon-photo --run 20 --device 1
