export CUDA_VISIBLE_DEVICES=0,1
export MODE=victim
LANG=python
SAMPLE_RATIO=20
Percentage=0.01
DATADIR=../dataset/APPS
LITFILE=./literals.json
# OUTPUTDIR should be same as DATADIR for the requirement of dataset.py in classifier
PRETRAINDIR=../save/codellama/CodeLlama-7b-hf/100/checkpoint-epoch-4
LOGFILE=completion_javaCorpus_eval.log
python -u run_lm.py \
        --mode=$MODE \
        --data_dir=$DATADIR \
        --lit_file=$LITFILE \
        --langs=$LANG \
        --output_dir=$DATADIR \
        --pretrain_dir=$PRETRAINDIR \
        --log_file=$LOGFILE \
        --model_type=codellama \
        --block_size=1024 \
        --eval_line \
        --logging_steps=100 \
        --seed=42 \
        --generate_method=top-k \
        --topk=50 \
        --temperature=0.8