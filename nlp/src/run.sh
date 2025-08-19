
for accum_step in 2 #8 32
do
	python3 main.py --model='modernbert' --accum_step=$accum_step
#	accelerate launch main.py --model='modernbert' --accum_step=$accum_step
done

#for accum_step in 2 8 32
#do
#	accelerate launch main.py --model='modernbert' --accum_step=$accum_step --epoch=20
#done
