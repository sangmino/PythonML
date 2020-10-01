import tensorflow as tf
import numpy as np
import pathlib
import matplotlib.pyplot as plt
import os
import tensorflow_datasets as tfds

#--------------------------------------------------------------------#
# region : Creating tensors in TensorFlow

## Create a tensor from a list or a numpy array
a = np.array([1, 2, 3], dtype=np.int32)
b = [4, 5, 6]

t_a = tf.convert_to_tensor(a)
t_b = tf.convert_to_tensor(b)

print(t_a)
print(t_b)

## Accessing values that a tensor refers to
t_ones.numpy()

## Tensor of constant values
const_tensor = tf.constant([1.2, 5, np.pi], dtype=tf.float32)
print(const_tensor)

# endregion

#--------------------------------------------------------------------#
# region : Manipulating the data type and shape of a tensor

## Changing the data type to a deisred type
t_a_new = tf.cast(t_a, tf.int64)
print(t_a_new.dtype)

t = tf.random.uniform(shape=(3, 5))

## Transposing a tensor
t_tr = tf.transpose(t)
print(t.shape, ' --> ', t_tr.shape)

## Reshaping a tensor from a 1D vector to a 2D array
t = tf.zeros((30,))
t_reshape = tf.reshape(t, shape=(5, 6))
print(t_reshape.shape)

# endregion

#--------------------------------------------------------------------#
# region : Applying mathematical operations to tensors

## Instantiate two random tensors
tf.random.set_seed(1)
t1 = tf.random.uniform(shape=(5, 2), 
                       minval=-1.0,
                       maxval=1.0)
t2 = tf.random.normal(shape=(5, 2), 
                      mean=0.0,
                      stddev=1.0)

t3 = tf.multiply(t1, t2).numpy()
print(t3)

## Mathematical operations
# Mean
t4 = tf.math.reduce_mean(t1, axis=0)
print(t4)

# Matrix-matrix product
t5 = tf.linalg.matmul(t1, t2, transpose_b=True)
print(t5.numpy())

t6 = tf.linalg.matmul(t1, t2, transpose_a=True)
print(t6.numpy())

## Computing the L-p norm of a tensor
norm_t1 = tf.norm(t1, ord=2, axis=1).numpy()
print(norm_t1)

# endregion

#--------------------------------------------------------------------#
# region : Split, stack, and concatenate tensors

## Dividing the tensor into a list of three tensors
tf.random.set_seed(1)
t = tf.random.uniform((6,))
print(t.numpy())

t_splits = tf.split(t, 3)
[item.numpy() for item in t_splits]

## Providing the sizes of different splits (size 5 into size 3 and size 2)
tf.random.set_seed(1)
t = tf.random.uniform((5,))
print(t.numpy())

t_splits = tf.split(t, num_or_size_splits=[3, 2])
[item.numpy() for item in t_splits]

## Concatenating tensors
A = tf.ones((3,))
B = tf.zeros((2,))
C = tf.concat([A, B], axis=0)
print(C.numpy())

## Stacking the tensors
A = tf.ones((3,))
B = tf.zeros((3,))

S = tf.stack([A, B], axis=1)
print(S.numpy())

# endregion


#--------------------------------------------------------------------#
# region : Building input pipelines using tf.data: The TensorFlow Dataset API

## Creating a TensorFlow Dataset from existing tensors 
a = [1.2, 3.4, 7.5, 4.1, 5.0, 1.0]
ds = tf.data.Dataset.from_tensor_slices(a)
print(ds)

## Creating batches from the dataset
ds_batch = ds.batch(3)
for i, elem in enumerate(ds_batch, 1):
    print('batch {}:'.format(i), elem.numpy())


## Combining two tensors into a joint dataset
tf.random.set_seed(1)
t_x = tf.random.uniform([4, 3], dtype=tf.float32)   # Feature values
t_y = tf.range(4)   # Class labels

# First create two separate datasets
ds_x = tf.data.Dataset.from_tensor_slices(t_x)
ds_y = tf.data.Dataset.from_tensor_slices(t_y)

# Use the zip function to form a joint dataset
ds_joint = tf.data.Dataset.zip((ds_x, ds_y))

for example in ds_joint:
    print('  x: ', example[0].numpy(), 
          '  y: ', example[1].numpy())

## Alternately, it can be done in one step:
ds_joint = tf.data.Dataset.from_tensor_slices((t_x, t_y))
for example in ds_joint:
    print('  x: ', example[0].numpy(), 
          '  y: ', example[1].numpy())


## Transformations to the individual element of a dataset
ds_trans = ds_joint.map(lambda x, y: (x*2-1.0, y))
for example in ds_trans:
    print('  x: ', example[0].numpy(), 
          '  y: ', example[1].numpy())

# endregion

#--------------------------------------------------------------------#
# region : Shuffle, batch, and repeat

## Shuffled version of the ds_joint dataset
tf.random.set_seed(1)
ds = ds_joint.shuffle(buffer_size=len(t_x))

## Divide the dataset into batches for model training
ds = ds_joint.batch(batch_size=3,
                    drop_remainder=False)

batch_x, batch_y = next(iter(ds))
print('Batch-x: \n', batch_x.numpy())
print('Batch-y:   ', batch_y.numpy())

## Repeat the batched dataset twice
## (We get four batches)
ds = ds_joint.batch(3).repeat(count=2)
for i,(batch_x, batch_y) in enumerate(ds):
    print(i, batch_x.shape, batch_y.numpy())

## On the other hand, repeat first and then batch
## (We get three batches)
ds = ds_joint.repeat(count=2).batch(3)
for i,(batch_x, batch_y) in enumerate(ds):
    print(i, batch_x.shape, batch_y.numpy())

# endregion

#--------------------------------------------------------------------#
# region : Creating a dataset from files on your local storage disk

imgdir_path = pathlib.Path('cat_dog_images')
file_list = sorted([str(path) for path in imgdir_path.glob('*.jpg')])
print(file_list)


fig = plt.figure(figsize=(10, 5))
for i,file in enumerate(file_list):
    img_raw = tf.io.read_file(file)
    img = tf.image.decode_image(img_raw)
    print('Image shape: ', img.shape)
    ax = fig.add_subplot(2, 3, i+1)
    ax.set_xticks([]); ax.set_yticks([])
    ax.imshow(img)
    ax.set_title(os.path.basename(file), size=15)

plt.tight_layout()
plt.show()


labels = [1 if 'dog' in os.path.basename(file) else 0
          for file in file_list]
print(labels)


ds_files_labels = tf.data.Dataset.from_tensor_slices(
    (file_list, labels))

for item in ds_files_labels:
    print(item[0].numpy(), item[1].numpy())


def load_and_preprocess(path, label):
    image = tf.io.read_file(path)
    image = tf.image.decode_jpeg(image, channels=3)
    image = tf.image.resize(image, [img_height, img_width])
    image /= 255.0

    return image, label

img_width, img_height = 120, 80

ds_images_labels = ds_files_labels.map(load_and_preprocess)

fig = plt.figure(figsize=(10, 5))
for i,example in enumerate(ds_images_labels):
    print(example[0].shape, example[1].numpy())
    ax = fig.add_subplot(2, 3, i+1)
    ax.set_xticks([]); ax.set_yticks([])
    ax.imshow(example[0])
    ax.set_title('{}'.format(example[1].numpy()), 
                 size=15)
    
plt.tight_layout()
plt.show()

# endregion

#--------------------------------------------------------------------#
# region : Fetching available datasets from the tensorflow_datasets library

print(len(tfds.list_builders()))
print(tfds.list_builders()[:5])

## Run this to see the full list:
tfds.list_builders()

## Fetching CelebA dataset
celeba_bldr = tfds.builder('celeb_a')
print(celeba_bldr.info.features)
print('\n', 30*"=", '\n')
print(celeba_bldr.info.features.keys())
print('\n', 30*"=", '\n')
print(celeba_bldr.info.features['image'])
print('\n', 30*"=", '\n')
print(celeba_bldr.info.features['attributes'].keys())
print('\n', 30*"=", '\n')
print(celeba_bldr.info.citation)

# 1. Download the data, prepare it, and write it to disk
celeba_bldr.download_and_prepare()

# 2. Load data from disk as tf.data.Datasets
datasets = celeba_bldr.as_dataset(shuffle_files=False)
datasets.keys()

# 3. Check out what the image examples look like
ds_train = datasets['train']
assert isinstance(ds_train, tf.data.Dataset)

example = next(iter(ds_train))
print(type(example))
print(example.keys())

# 4, Reformat the dataset into (features, labels)
ds_train = ds_train.map(lambda item: 
     (item['image'], tf.cast(item['attributes']['Male'], tf.int32)))


# 5. Batch the dataset and take a batch of 18 examples
#    to visualize with labels
ds_train = ds_train.batch(18)
images, labels = next(iter(ds_train))

print(images.shape, labels)

fig = plt.figure(figsize=(12, 8))
for i,(image,label) in enumerate(zip(images, labels)):
    ax = fig.add_subplot(3, 6, i+1)
    ax.set_xticks([]); ax.set_yticks([])
    ax.imshow(image)
    ax.set_title('{}'.format(label), size=15)
    

plt.show()






