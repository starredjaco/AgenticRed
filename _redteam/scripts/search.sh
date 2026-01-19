#!/bin/bash
# Set HF_HOME if needed
# export HF_HOME=''

# Set model endpoints
LLAMA2_ENDPOINT=''
LLAMA3_ENDPOINT=''
ATTACKER_ENDPOINTS=''
VICUNA_ENDPOINTS=''
CLASSIFIER_ENDPOINT=''


# Parse --expr and --seed as named arguments
INDEX=1
SEED=42

while [[ $# -gt 0 ]]; do
    key="$1"
    case $key in
        --expr)
            INDEX="$2"
            shift # past argument
            shift # past value
            ;;
        --seed)
            SEED="$2"
            shift
            shift
            ;;
        *)
            echo "Unknown argument: $1"
            exit 1
            ;;
    esac
done

echo "Experiment index: ${INDEX}"
echo "Shuffle seed: ${SEED}"


case "$INDEX" in
    1)
        echo "1) baseline"
        env ATTACKER_ENDPOINT="$ATTACKER_ENDPOINT" LLAMA2_ENDPOINT="$LLAMA2_ENDPOINT" CLASSIFIER_ENDPOINT="$CLASSIFIER_ENDPOINT" SEED="$SEED" \
        envsubst < ../configs/exp1_baseline.yaml > ../configs/exp1_baseline_sub.yaml
        python -u search.py --config ../configs/exp1_baseline_sub.yaml
        ;;
    2)
        echo "2) switch target model -> LLAMA3"
        env ATTACKER_ENDPOINTS="$ATTACKER_ENDPOINTS" LLAMA3_ENDPOINT="$LLAMA3_ENDPOINT" CLASSIFIER_ENDPOINT="$CLASSIFIER_ENDPOINT" SEED="$SEED" \
        envsubst < ../configs/exp2_llama3.yaml > ../configs/exp2_llama3_sub.yaml
        python -u search.py --config ../configs/exp2_llama3_sub.yaml
        ;;
    3)
        echo "3) switch attacker model -> VICUNA"
        env VICUNA_ENDPOINTS="$VICUNA_ENDPOINTS" LLAMA2_ENDPOINT="$LLAMA2_ENDPOINT" CLASSIFIER_ENDPOINT="$CLASSIFIER_ENDPOINT" SEED="$SEED" \
        envsubst < ../configs/exp3_vicuna.yaml > ../configs/exp3_vicuna_sub.yaml
        python -u search.py --config ../configs/exp3_vicuna_sub.yaml
        ;;
    4)
        echo "4) switch meta agent model -> deepseek-reasoner"
        env ATTACKER_ENDPOINTS="$ATTACKER_ENDPOINTS" LLAMA2_ENDPOINT="$LLAMA2_ENDPOINT" CLASSIFIER_ENDPOINT="$CLASSIFIER_ENDPOINT" SEED="$SEED" \
        envsubst < ../configs/exp4_deepseek.yaml > ../configs/exp4_deepseek_sub.yaml
        python -u search.py --config ../configs/exp4_deepseek_sub.yaml
        ;;
    5)
        echo "5) w/o evolutionary pressure"
        env ATTACKER_ENDPOINTS="$ATTACKER_ENDPOINTS" LLAMA2_ENDPOINT="$LLAMA2_ENDPOINT" CLASSIFIER_ENDPOINT="$CLASSIFIER_ENDPOINT" SEED="$SEED" \
        envsubst < ../configs/exp5_no_evo.yaml > ../configs/exp5_no_evo_sub.yaml
        python -u search.py --config ../configs/exp5_no_evo_sub.yaml
        ;;
    6)
        echo "6) w/ weak archive"
        env ATTACKER_ENDPOINTS="$ATTACKER_ENDPOINTS" LLAMA2_ENDPOINT="$LLAMA2_ENDPOINT" CLASSIFIER_ENDPOINT="$CLASSIFIER_ENDPOINT" SEED="$SEED" \
        envsubst < ../configs/exp6_weak_archive.yaml > ../configs/exp6_weak_archive_sub.yaml
        python -u search.py --config ../configs/exp6_weak_archive_sub.yaml
        ;;
    7)
        echo "7) w/ diversity incentive"
        env ATTACKER_ENDPOINTS="$ATTACKER_ENDPOINTS" LLAMA2_ENDPOINT="$LLAMA2_ENDPOINT" CLASSIFIER_ENDPOINT="$CLASSIFIER_ENDPOINT" SEED="$SEED" \
        envsubst < ../configs/exp7_diversity_incentive.yaml > ../configs/exp7_diversity_incentive_sub.yaml
        python -u search.py --config ../configs/exp7_diversity_incentive_sub.yaml
        ;;
    8)
        echo "8) w/ diversity threshold"
        env ATTACKER_ENDPOINTS="$ATTACKER_ENDPOINTS" LLAMA2_ENDPOINT="$LLAMA2_ENDPOINT" CLASSIFIER_ENDPOINT="$CLASSIFIER_ENDPOINT" SEED="$SEED" \
        envsubst < ../configs/exp8_diversity_threshold.yaml > ../configs/exp8_diversity_threshold_sub.yaml
        python -u search.py --config ../configs/exp8_diversity_threshold_sub.yaml
        ;;
    *)
        echo "Unknown experiment index: ${INDEX}"
        exit 1
        ;;
esac