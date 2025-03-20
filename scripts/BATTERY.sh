export CUDA_VISIBLE_DEVICES=0

# Train phase only
python3 run_battery.py --anormly_ratio 0.5 \
    --num_epochs 20 \
    --batch_size 256 \
    --mode train \
    --dataset BATTERY \
    --data_path /Users/orient/Documents/Projects/GitHub/Anomaly-Transformer/data_factory/raw/DanDLN_Data.csv \
    --input_c 12 \
    --output_c 12 \
    --win_size 15 \
    --model_save_path checkpoints/battery_new
