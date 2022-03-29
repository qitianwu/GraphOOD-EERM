# cora
python main.py --method erm --dataset cora --gnn_gen gcn --gnn gcn --run 20 --lr 0.001 --device 0

python main.py --method eerm --dataset cora --gnn_gen gcn --gnn gcn --lr 0.005 --K 10 --T 1 --num_sample 1 --beta 1.0 --lr_a 0.001 --run 20 --device 0

python main.py --method erm --dataset cora --gnn_gen gat --gnn gcn --run 20 --lr 0.001 --device 0

python main.py --method eerm --dataset cora --gnn_gen gat --gnn gcn --lr 0.005 --K 10 --T 1 --num_sample 1 --beta 1.0 --lr_a 0.001 --run 20 --device 0

python main.py --method erm --dataset cora --gnn_gen sgc --gnn gcn --run 20 --lr 0.001 --device 0

python main.py --method eerm --dataset cora --gnn_gen sgc --gnn gcn --lr 0.005 --K 5 --T 1 --num_sample 1 --beta 1.0 --lr_a 0.005 --run 20 --device 0

# amazon-photo
python main.py --method erm --dataset amazon-photo --gnn_gen gcn --gnn gcn --run 20 --lr 0.001 --device 0

python main.py --method eerm --dataset amazon-photo --gnn_gen gcn --gnn gcn --lr 0.01 --K 5 --T 1 --num_sample 1 --beta 1.0 --lr_a 0.005 --run 20 --device 0

python main.py --method erm --dataset amazon-photo --gnn_gen gat --gnn gcn --run 20 --lr 0.001 --device 0

python main.py --method eerm --dataset amazon-photo --gnn_gen gat --gnn gcn --lr 0.01 --K 5 --T 1 --num_sample 1 --beta 1.0 --lr_a 0.005 --run 20 --device 0

python main.py --method erm --dataset amazon-photo --gnn_gen sgc --gnn gcn --run 20 --lr 0.001 --device 0

python main.py --method eerm --dataset amazon-photo --gnn_gen sgc --gnn gcn --lr 0.01 --K 5 --T 1 --num_sample 1 --beta 1.0 --lr_a 0.005 --run 20 --device 0
