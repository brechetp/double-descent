#for arg in 'softmax' 'no-softmax'; do srun -p cuda -x cuda01,cuda02 python train_mnist.py  --vary_name softmax --width 1000 --gd_mode 'full' "--$arg" & done
dir='slurm/scripts'
template='template.sbatch'
max_run=3;
name='resume'
fname="$dir/$name.sbatch"
cp "$dir/$template" $fname

sed -i "s/^\(#SBATCH -J\) test_slurm/\1 $name/" $fname
lst=`find $1 -iname checkpoint.pth`
echo $lst

for ckpt in lst;
#for n in 60;
do
    echo "#srun python train_mnist.py --checkpoint $ckpt" >> $fname; 
    
done;

nexp=`grep srun $fname  | wc -l`  # the number of experiments in the file
total=`wc -l < $fname`  # the total number of lines


let beg=$total-$nexp+1
let i=$beg

let blocks=($nexp - 1)/$max_run+1

for bcnt in `seq 1 $blocks | shuf`; do
    sed -i "s/^\(#SBATCH -J\) .*$/\1 $name-$bcnt/" $fname
    sed -i "$i,`expr $i + $max_run - 1`s/^#*//" $fname
    sbatch $fname
    sed -i "$i,`expr $i + $max_run - 1`s/^/#/" $fname
    let i=$i+$max_run
done;

