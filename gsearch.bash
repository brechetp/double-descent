#for arg in 'softmax' 'no-softmax'; do srun -p cuda -x cuda01,cuda02 python train_mnist.py  --vary_name softmax --width 1000 --gd_mode 'full' "--$arg" & done
dir='slurm/scripts'
template='template_mpcdf.sbatch'
name=${@}
name=${name/ /_}
fname="$dir/$name.sbatch"
cp "$dir/$template" $fname

sed -i "s/^\(#SBATCH -J\) test_slurm/\1 $name/" $fname

#for n in 36 37 38 39 40 41 42 43 44;
for n in 30 35 39 40 41 45 50 60;
do
    np=`echo "$n*10^3" | bc` &&
    echo "#srun python train_mnist.py --nepochs 6000  --num_parameters $np --vary_name size_max num_parameters  --no-softmax --size_max 4000 --learning_rate 0.01" >> $fname; 
    
done;

nexp=`grep srun $fname  | wc -l`  # the number of experiments in the file
total=`wc -l < $fname`  # the total number of lines

max_run=1

let beg=$total-$nexp+1
let i=$beg

let blocks=($nexp - 1)/$max_run+1

for bcnt in `seq 1 $blocks`; do
    sed -i "s/^\(#SBATCH -J\) .*$/\1 $name-$bcnt/" $fname
    sed -i "$i,`expr $i + $max_run - 1`s/^#*//" $fname
    sbatch $fname
    sed -i "$i,`expr $i + $max_run - 1`s/^/#/" $fname
    let i=$i+$max_run
done;




