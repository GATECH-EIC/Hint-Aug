
#!/usr/bin/env bash         \

currenttime=`date "+%Y%m%d_%H%M%S"`

CONFIG=./experiments/LoRA/ViT-B_prompt_lora_8_patch.yaml
CKPT=/home/shang/ViT-B_16.npz
WEIGHT_DECAY=0.0001


mkdir -p LoRA
PYTHONPATH="$(dirname $0)/..":$PYTHONPATH \

# food-101 oxford_pets stanford_cars oxford_flowers fgvc_aircraft

for LR in 0.001
do 
    for DATASET in food-101 oxford_pets stanford_cars oxford_flowers fgvc_aircraft
    do
        for SEED in 2
        do
            for SHOT in 4 8 12 16
            do 
                CUDA_VISIBLE_DEVICES=0,1,2,3 python -m torch.distributed.launch --nproc_per_node=4 --master_addr="127.0.0.1" --master_port=1676 --use_env train_patch.py --switch_bn True --use_pre_soft --mild_l_inf 0.001 --patch_fool --teacher_model 'vit_base_patch16_224_in21k' --data-path=./data/dataset/${DATASET} --data-set=${DATASET}-FS --cfg=${CONFIG} --resume=${CKPT} --output_dir=./saves/few-shot_${DATASET}_shot-${SHOT}_seed-${SEED}_lr-${LR}_wd-${WEIGHT_DECAY}_adapter --batch-size=16 --lr=${LR} --epochs=100 --is_LoRA --weight-decay=${WEIGHT_DECAY} --few-shot-seed=${SEED} --few-shot-shot=${SHOT} --launcher="pytorch"\
                    2>&1 | tee -a LoRA/${currenttime}-${DATASET}-${LR}-seed-${SEED}-shot-${SHOT}-adapter.log
            done
        done
    done
done