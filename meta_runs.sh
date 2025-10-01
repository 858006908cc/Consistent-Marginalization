# Please run meta_memory_bank_estimation.py first to estimate the discrepancies.
# /usr/bin/python3.9 meta_memory_bank_estimation.py

dataset_names=('banking77' 'clinc' 'massive_scenario' 'mtop_intent' 'stackexchange')
for dataset in "${dataset_names[@]}"
do
    echo "=== Running dataset: $dataset ==="
    /usr/bin/python3.9 meta_train_CM.py --dataset_name "$dataset" \
        > ".../unlearning_meta_memorisation_${dataset}.log" 2>&1
done