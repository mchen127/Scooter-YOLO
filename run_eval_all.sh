#!/bin/bash

# Evaluate all models in YOLO-Taiwan-Traffic directory
# Skips models that are already in results/all_results.json

RESULT_FILE="results/all_results.json"
BASE_DIR="YOLO-Taiwan-Traffic"

# Create results directory if it doesn't exist
mkdir -p results

# Initialize results file if it doesn't exist
if [ ! -f "$RESULT_FILE" ]; then
    echo "[]" > "$RESULT_FILE"
fi

echo "=========================================="
echo "Evaluating all models in $BASE_DIR"
echo "Results will be saved to: $RESULT_FILE"
echo "=========================================="

# Loop through all subdirectories
for exp_dir in "$BASE_DIR"/*/; do
    exp_name=$(basename "$exp_dir")
    model_path="${exp_dir}weights/best.pt"
    
    # Check if best.pt exists
    if [ ! -f "$model_path" ]; then
        echo "[SKIP] No best.pt found in: $exp_name"
        continue
    fi
    
    # Check if model is already in results file
    if grep -q "\"model_path\": \"$model_path\"" "$RESULT_FILE" 2>/dev/null; then
        echo "[SKIP] Already evaluated: $exp_name"
        continue
    fi
    
    echo ""
    echo "[EVAL] Evaluating: $exp_name"
    echo "       Model: $model_path"
    python scripts/evaluate.py --model "$model_path"
done

echo ""
echo "=========================================="
echo "All evaluations complete!"
echo "Results saved to: $RESULT_FILE"
echo "=========================================="