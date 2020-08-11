#for arg in 'softmax' 'no-softmax'; do srun -p cuda -x cuda01,cuda02 python train_mnist.py  --vary_name softmax --width 1000 --gd_mode 'full' "--$arg" & done
dir='slurm/scripts'
template='template.sbatch'
name=${@}
name=${name/ /_}
fname="$dir/$name.sbatch"
cp "$dir/$template" $fname

sed -i "s/--job-name=.*$/--job-name=$name/" $fname

for w in 6000 7000 7300 7400 7500 7547 7600 7700 8000;
do
    echo "python train_mnist.py --nepochs 1000  --vary_name size_max width --width $w --softmax" >> $fname; 
    
done;

sbatch $fname

