device="cuda:0"
epoch=100
batch=64
lr=0.1
wd=0.0001
optim="SGD"
scheduler="cosine"
pretrained=0
num_workers=4

dataset="CIFAR10"
seed=42
model="R18"



python3 main.py --seed=$seed --model=$model --num_workers=$num_workers --epoch=$epoch --batch=$batch --lr=$lr --wd=$wd --optimizer=$optim --scheduler=$scheduler --pretrained=$pretrained --device=$device --dataset=$dataset 
