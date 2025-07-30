device="cuda:0"
epoch=100
batch=64
lr=0.1
wd=0.0001
optim="SGD"
#optim="Adam"
#lr=0.001
#wd=0.0001
scheduler="cosine"
pretrained=0
num_workers=4

dataset="CIFAR10"
#dataset="Tinyimagenet"
seed=42
model="R18"



python3 main.py --seed=$seed --model=$model --num_workers=$num_workers --epoch=$epoch --batch=$batch --lr=$lr --wd=$wd --optimizer=$optim --scheduler=$scheduler --pretrained=$pretrained --device=$device --dataset=$dataset 
python3 exp_merge.py --exp_num=$exp_num

