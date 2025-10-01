

dataset_names=('banking77' 'clinc' 'massive_scenario' 'mtop_intent' 'stackexchange')

for dataset in "${dataset_names[@]}"
do
    echo "=== Running dataset: $dataset ==="
    /usr/bin/python3.9 chatgpt_train.py.py --dataset_name "$dataset" \
        > ".../unlearning_memorisation_${dataset}_chatgpt4o_mini.log" 2>&1
done
