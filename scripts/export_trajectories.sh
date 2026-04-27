#!/bin/bash
# Export Nexus-Alpha RFT Trajectories for Google Colab
# Extracts only 'rewarded' trajectories to ensure high-quality training data.

echo "📦 Preparing Reinforcement Learning Data..."

DATA_DIR="../data/rft"
EXPORT_FILE="${DATA_DIR}/colab_export.zip"
JSONL_FILE="${DATA_DIR}/trajectories.jsonl"
TEMP_FILE="${DATA_DIR}/filtered_trajectories.jsonl"

if [ ! -f "$JSONL_FILE" ]; then
    echo "❌ No trajectories found at $JSONL_FILE"
    exit 1
fi

# Extract only the trajectory records that have received a final PnL reward
grep '"status": "rewarded"' "$JSONL_FILE" > "$TEMP_FILE"

TOTAL_TRADES=$(wc -l < "$TEMP_FILE" | tr -d ' ')

if [ "$TOTAL_TRADES" -lt 10 ]; then
    echo "⚠️ Warning: You only have $TOTAL_TRADES rewarded trades."
    echo "You usually want at least 50-100 before training a new LoRA."
fi

# Zip the filtered dataset
cd "$DATA_DIR" || exit
zip -q colab_export.zip $(basename "$TEMP_FILE")

# Cleanup temp
rm "$(basename "$TEMP_FILE")"

echo "✅ Success!"
echo "- Exported $TOTAL_TRADES completed trades."
echo "- File saved to: $EXPORT_FILE"
echo ""
echo "Next Steps:"
echo "1. Upload $EXPORT_FILE to your Google Colab instance."
echo "2. Run the nexus_art_trainer.ipynb notebook."
