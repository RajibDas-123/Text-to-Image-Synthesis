export MODEL_NAME="runwayml/stable-diffusion-v1-5"
export DATASET_DIR="../data/flowers"

python train_text_to_image_updated.py \
  --pretrained_model_name_or_path=$MODEL_NAME \
  --train_data_dir=$DATASET_DIR \
  --resolution=64 --center_crop --random_flip \
  --save_loss \
  --loss_save_path='base.csv'\
  --train_batch_size=16 \
  --max_grad_norm=100000 \
  --mixed_precision="fp16" \
  --max_train_steps=1500 \
  --num_train_epochs=50 \
  --learning_rate=1e-05 \
  --output_dir="sd-flower-model"

