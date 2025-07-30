device="cuda:1"
epoch=100
batch=64
lr=0.1
wd=0.0001
optim="SGD"
#optim="Adam"
#lr=0.0001
#wd=0.0
scheduler="cosine"
pretrained=0
num_workers=4



for dataset in "CIFAR10" "CIFAR100" "Tinyimagenet"
do
	for seed in 42 #72 448
	do
		for pretrained in 0 1 2
		do
			for model in "R50" "vit" 
			do
				exp_num=$dataset"_"$model
				case_name=$model"_"$pretrained"_"$optim"_"$lr
				python3 exp.py --seed=$seed --model=$model --num_workers=$num_workers --epoch=$epoch --batch=$batch --lr=$lr --wd=$wd --optimizer=$optim --scheduler=$scheduler --pretrained=$pretrained --device=$device --dataset=$dataset --exp_num=$exp_num --case_name=$case_name 
				python3 exp_merge.py --exp_num=$exp_num
			done
		done
	done
done

optim="Adam"
lr=0.001
wd=0.0001

for dataset in "CIFAR10" "CIFAR100" "Tinyimagenet"
do
	for seed in 42 #72 448
	do

		for pretrained in 0 1 2
		do
			for model in "R50" "vit" 
			do
				exp_num=$dataset"_"$model
				case_name=$model"_"$pretrained"_"$optim"_"$lr
				python3 exp.py --seed=$seed --model=$model --num_workers=$num_workers --epoch=$epoch --batch=$batch --lr=$lr --wd=$wd --optimizer=$optim --scheduler=$scheduler --pretrained=$pretrained --device=$device --dataset=$dataset --exp_num=$exp_num --case_name=$case_name
				python3 exp_merge.py --exp_num=$exp_num
			done
		done
	done
done

