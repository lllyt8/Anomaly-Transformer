export CUDA_VISIBLE_DEVICES=0

python main.py --anormly_ratio 0.5 --num_epochs 10 --batch_size 256 \
    --mode train --dataset BATTERY --data_path dataset/battery_data.csv \
    --input_c 12 --output_c 12 --win_size 100

python main.py --anormly_ratio 0.5 --num_epochs 10 --batch_size 256 \
    --mode test --dataset BATTERY --data_path dataset/battery_data.csv \
    --input_c 12 --output_c 12 --win_size 100 --pretrained_model 10
