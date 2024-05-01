export MODEL_NAME="runwayml/stable-diffusion-v1-5"
export DATASET_DIR="../data/flowers"

python train_text_to_image_updated.py \
  --pretrained_model_name_or_path=$MODEL_NAME \
  --train_data_dir=$DATASET_DIR \
  --save_loss \
  --loss_save_path='with_clip_and_swa.csv'\
  --resolution=64 --center_crop --random_flip \
  --train_batch_size=16 \
  --max_grad_norm=1 \
  --use_swa\
  --mixed_precision="fp16" \
  --max_train_steps=150 \
  --num_train_epochs=50 \
  --learning_rate=1e-05 \
  --output_dir="sd-flower-model"

