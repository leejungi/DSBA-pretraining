
#for accum_step in 1 8 32 128
#do
#	python3 main.py --model='modernbert' --accum_step=$accum_step 
##	accelerate launch main.py --model='modernbert' --accum_step=$accum_step
#done

for accum_step in 128
do
	python3 main.py --model='modernbert' --accum_step=$accum_step --epoch=5 --wd=0.001
done
