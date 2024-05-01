export DATASET_DIR="../../data/cubs/cubs"
export GLOVE_PATH="../../data/glove.6B.300d.txt"

/usr/bin/python3 train_gan.py \
  --train_data_dir=$DATASET_DIR \
  --glove_path=$GLOVE_PATH \
  --resolution=2 \
  --train_batch_size=64 \
  --num_train_epochs=500 \
  --output_dir="gan-cub-model"