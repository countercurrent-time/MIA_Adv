LANG=python
DATADIR=../dataset/APPS
LITFILE=./literals.json
OUTPUTDIR=../save/
PRETRAINDIR=codellama/CodeLlama-7b-hf
LOGFILE=completion_javaCorpus.log
PER_NODE_GPU=2
export CUDA_VISIBLE_DEVICES=0,1
python3 run_lm.py \
        --data_dir=$DATADIR \
        --lit_file=$LITFILE \
        --langs=$LANG \
        --output_dir=$OUTPUTDIR \
        --pretrain_dir=$PRETRAINDIR \
        --log_file=$LOGFILE \
        --model_type=codellama \
        --block_size=1024 \
        --do_train \
        --gpu_per_node $PER_NODE_GPU \
        --learning_rate=8e-5 \
        --weight_decay=0.01 \
        --evaluate_during_training \
        --per_gpu_train_batch_size=2 \
        --per_gpu_eval_batch_size=4 \
        --gradient_accumulation_steps=4 \
        --num_train_epochs=5 \
        --logging_steps=100 \
        --save_steps=1000 \
        --seed=42 \
        --overwrite_output_dir \
        --fp16 \
        --not_pretrain \
        --use_lora \
        --lora_r=8 \
        --lora_alpha=32 \
