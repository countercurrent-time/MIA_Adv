PER_NODE_GPU=2
export CUDA_VISIBLE_DEVICES=0,1
# MODEL=codellama/CodeLlama-7b-hf
# SURROGATE_MODEL=codellama/CodeLlama-7b-hf
MODEL=microsoft/codebert-base
SURROGATE_MODEL=deepseek-ai/deepseek-coder-6.7b-base
MASTER_PORT=95497 # modify
Percentage=0.01


for SAMPLE_RATIO in {20..20..10}
do

LANG=python
CLASSIFIER_SAVE_DICT=classifier_save/${SURROGATE_MODEL##*/}/${SAMPLE_RATIO}/
PREDICTION_DATA_FOLDER_PATH=../../dataset/APPS
LITFILE=../../llm_finetuning/literals.json

python run.py \
    --do_lower_case \
    --lang ${LANG} \
    --surrogate_model ${SURROGATE_MODEL} \
    --sample_ratio ${SAMPLE_RATIO} \
    --batch_size 16 \
    --learning_rate 5e-5 \
    --num_train_epochs 4 \
    --classifier_save_dir ${CLASSIFIER_SAVE_DICT} \
    --prediction_data_folder_path ${PREDICTION_DATA_FOLDER_PATH} \
    --lit_file ${LITFILE} \
    --classifier_model_path ${MODEL} \
    --weight_decay=0.01 \
    --seed 43 \
    --mode surrogate \
    --use_tree_component
    # --mode checkpoint-epoch-5_surrogate \

done

