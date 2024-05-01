import argparse
import time
from glob import glob
import os
from tqdm import tqdm
from PIL import Image
import numpy as np
import pandas as pd
from utils import loadGloveModel
import tensorflow as tf
from models import build_discriminator_func, build_generator_func
from utils import generator_loss, discriminator_loss, hms_string, save_images


def parse_args():
    parser = argparse.ArgumentParser(description="Simple example of a training script.")
    parser.add_argument(
        "--train_data_dir",
        type=str,
        default=None,
        help=(
            "A folder containing the training data. "
       ),
    )
    parser.add_argument(
        "--glove_path",
        type=str,
        default=None,
        help=(
            "Path to the glove file "
       ),
    )
    parser.add_argument(
        "--resolution",
        type=int,
        default=2,
        help=(
            "The resolution for input images, all the images in the train/validation dataset will be resized to this"
            " resolution"
        ),
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="sd-model-finetuned",
        help="The output directory where the model predictions and checkpoints will be written.",
    )
    parser.add_argument(
        "--train_batch_size", type=int, default=16, help="Batch size (per device) for the training dataloader."
    )
    parser.add_argument("--num_train_epochs", type=int, default=100)

    args = parser.parse_args()

    return args

args = parse_args()

print(args)

GENERATE_RES = args.resolution # Generation resolution factor 
GENERATE_SQUARE = 32 * GENERATE_RES # rows/cols (should be square)
IMAGE_CHANNELS = 3


PREVIEW_ROWS = 4
PREVIEW_COLS = 7
PREVIEW_MARGIN = 16

# Size vector to generate images from
SEED_SIZE = 100
EMBEDDING_SIZE = 300

# Configuration
DATA_PATH = args.train_data_dir
MODEL_PATH = args.output_dir
GLOVE_PATH = args.glove_path
EPOCHS = args.num_train_epochs
BATCH_SIZE = args.train_batch_size
BUFFER_SIZE = 4000

print(f"Will generate {GENERATE_SQUARE}px square images.")

glove_embeddings = loadGloveModel(GLOVE_PATH)

training_binary_path = os.path.join(DATA_PATH, "image_npy",
        f'training_data_{GENERATE_SQUARE}_{GENERATE_SQUARE}_')

start = time.time()

training_data = []
flower_paths = sorted(glob(DATA_PATH + "/*.jpg"))

for filename, file_path in tqdm(enumerate(flower_paths)):
    try:
        image = Image.open(file_path).resize((GENERATE_SQUARE,
            GENERATE_SQUARE),Image.LANCZOS)
        channel = np.asarray(image).shape[2]
        if channel == 3:
            training_data.append(np.asarray(image))
    except KeyboardInterrupt:
        break
    except:
        pass
    if len(training_data) == 100:
        training_data = np.reshape(training_data,(-1,GENERATE_SQUARE,
                GENERATE_SQUARE,IMAGE_CHANNELS))
        training_data = training_data.astype(np.float32)
        training_data = training_data / 127.5 - 1.
        np.save(training_binary_path + str(100000 + filename) + ".npy",training_data)
        elapsed = time.time()-start
        training_data = []


caption_df = pd.read_csv(os.path.join(DATA_PATH, "metadata.csv"))

captions = []
caption_embeddings = np.zeros((len(caption_df),300),dtype=np.float32)
for i, row in tqdm(caption_df.iterrows()):
    filename = row['file_name'].rstrip('jpg')
    x = row['caption']
    x = x.replace(" ","")
    captions.append(x)
    count = 0
    for t in x:
        try:
            caption_embeddings[i] += glove_embeddings[t]
            count += 1
        except:
            pass
    caption_embeddings[i] /= count

embedding_binary_path = os.path.join(DATA_PATH,
        f'embedding_data.npy')
np.save(embedding_binary_path,caption_embeddings)

image_binary_path = os.path.join(DATA_PATH, 'image_npy')
images = os.listdir(image_binary_path)

final_images = np.load(os.path.join(image_binary_path, images[0]))
for i in images[1:]:
    try:
        final_images = np.concatenate([final_images,np.load(image_binary_path + i)],axis = 0)
    except:
        pass

save_images_captions = captions[-28:].copy()
save_images_embeddings = np.copy(caption_embeddings[-28:])
save_images_npy = np.copy(final_images[-28:])
save_images_rl = final_images[-28:]

train_images_captions = captions[:-28].copy()
train_images_embeddings = np.copy(caption_embeddings[:-28])
train_images_npy = np.copy(final_images[:-28])
train_images = final_images[:-28]

p = np.random.permutation(len(train_images))
final_images_shuffled = train_images[p]
final_embeddings_shuffled = train_images_embeddings[p]

train_dataset = tf.data.Dataset.from_tensor_slices({'images': final_images_shuffled,
                                                    'embeddings': final_embeddings_shuffled}).shuffle(BUFFER_SIZE).batch(BATCH_SIZE)

cross_entropy = tf.keras.losses.BinaryCrossentropy()
generator = build_generator_func(SEED_SIZE,EMBEDDING_SIZE, IMAGE_CHANNELS)

image_shape = (GENERATE_SQUARE,GENERATE_SQUARE,IMAGE_CHANNELS)
discriminator = build_discriminator_func(image_shape,EMBEDDING_SIZE)

generator_optimizer = tf.keras.optimizers.Adam(learning_rate=2.0e-4,beta_1 = 0.5)
discriminator_optimizer = tf.keras.optimizers.Adam(learning_rate=2.0e-4,beta_1 = 0.5)


@tf.function
def train_step(images,captions,fake_captions):
  seed = tf.random.normal([images.shape[0], SEED_SIZE],dtype=tf.float32)

  with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
    generated_images = generator((seed,captions), training=True)
    real_image_real_text = discriminator((images,captions), training=True)
    real_image_fake_text = discriminator((images,fake_captions), training=True)
    fake_image_real_text = discriminator((generated_images,captions), training=True)

    gen_loss = generator_loss(cross_entropy, fake_image_real_text)
    disc_loss = discriminator_loss(cross_entropy, real_image_real_text, fake_image_real_text, real_image_fake_text)


    gradients_of_generator = gen_tape.gradient(\
        gen_loss, generator.trainable_variables)
    gradients_of_discriminator = disc_tape.gradient(\
        disc_loss, discriminator.trainable_variables)

    generator_optimizer.apply_gradients(zip(
        gradients_of_generator, generator.trainable_variables))
    discriminator_optimizer.apply_gradients(zip(
        gradients_of_discriminator, 
        discriminator.trainable_variables))
  return gen_loss,disc_loss

def train(train_dataset, epochs):
  epochs = 500

  fixed_seed = np.random.normal(0, 1, (PREVIEW_ROWS * PREVIEW_COLS, 
                                    SEED_SIZE))
  fixed_embed = save_images_embeddings

  start = time.time()

  for epoch in range(epochs):
      print("epoch start...")
      epoch_start = time.time()

      gen_loss_list = []
      disc_loss_list = []

      for batch in train_dataset:
        train_batch = batch['images']
        caption_batch = batch['embeddings']

        fake_caption_batch = np.copy(caption_batch)
        np.random.shuffle(fake_caption_batch)

        t = train_step(train_batch,caption_batch,fake_caption_batch)
        gen_loss_list.append(t[0])
        disc_loss_list.append(t[1])
        
      print("now")
      g_loss = sum(gen_loss_list) / len(gen_loss_list)
      d_loss = sum(disc_loss_list) / len(disc_loss_list)

      epoch_elapsed = time.time()-epoch_start
      print(f'Epoch {epoch+1}, gen loss={g_loss},disc loss={d_loss}, {hms_string(epoch_elapsed)}')
      save_images(generator, epoch,fixed_seed,fixed_embed, PREVIEW_MARGIN, PREVIEW_ROWS, PREVIEW_COLS, GENERATE_SQUARE)

      generator.save(os.path.join(MODEL_PATH,"text_to_image_generator_character.h5"))
      discriminator.save(os.path.join(MODEL_PATH,"text_to_image_disc_character.h5"))
      print("model saved")

      elapsed = time.time()-start
      print ('Training time:', hms_string(elapsed))

if __name__ == "__main__":
    train(train_dataset, EPOCHS)

     

