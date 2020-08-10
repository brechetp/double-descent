#for arg in 'softmax' 'no-softmax'; do srun -p cuda -x cuda01,cuda02 python train_mnist.py  --vary_name softmax --width 1000 --gd_mode 'full' "--$arg" & done
dir='slurm/scripts'
template='template.sbatch'
name=${@}
name=${name/ /_}
fname="$dir/$name.sbatch"
cp "$dir/$template" $fname

sed -i "s/--job-name=.*$/--job-name=$name/" $fname

for arg in 'softmax' 'no-softmax'; do 
    echo "sbatch python train_mnist.py --vary_name softmax --width 1000 --gd_mode 'full' --$arg"; done

sbatch $fname
