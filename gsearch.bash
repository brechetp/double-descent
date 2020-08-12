#for arg in 'softmax' 'no-softmax'; do srun -p cuda -x cuda01,cuda02 python train_mnist.py  --vary_name softmax --width 1000 --gd_mode 'full' "--$arg" & done
dir='slurm/scripts'
template='template.sbatch'
name=${@}
name=${name/ /_}
fname="$dir/$name.sbatch"
cp "$dir/$template" $fname

sed -i "s/^\(#SBATCH -J\) test_slurm/\1 $name/" $fname

#for n in 36 37 38 39 40 41 42 43 44;
for n in 59 60 61;
do
    np=`echo "$n*10^4" | bc` &&
    echo "srun python train_mnist.py --nepochs 6000  --num_parameters $np --vary_name size_max num_parameters  --no-softmax --size_max None --learning_rate 0.01" >> $fname; 
    
done;

sbatch $fname

