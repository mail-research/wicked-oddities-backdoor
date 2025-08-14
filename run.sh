POISON_RATE=0.1
export DATASET='CIFAR10'
export MODEL='myresnet18'
export STRAG='knn'
export SURR_MODEL='vicreg'
export LABEL=0
export PERCENTILE=-1
export K=50
ATTACK='badnet'
export ELABEL="new/${DATASET}_${MODEL}_${STRAG}_${SURR_MODEL}_${POISON_RATE}"
export base_dir="logs/${ATTACK}/${ELABEL}/"
mkdir -p $base_dir
base_log="${base_dir}log"   
echo $base_dir

sh etc/run_${ATTACK}.sh ${POISON_RATE} 
