# Tensorflow tutorial on sentiment analysis
Contains the Dockerfiles and code for the Tensorflow tutorial on sentiment analysis.

## Docker images
```bash
docker pull datagovsg/tensorflow-fold:v1.12.0-rc2  # For CPU
docker pull datagovsg/tensorflow-fold:v1.12.0-rc2-gpu  # For GPU
```
To run the jupyter notebook.
```bash
cd tensorflow-sentiment-tutorial

# For CPU
docker run -d \
       -p 8888:8888 \
       -p 6006:6006 \
       --volume=`pwd`:/workdir \
       --workdir=/workdir \
       datagovsg/tensorflow-fold:v1.12.0-rc2

# For GPU
nvidia-docker run -d \
       -p 8888:8888 \
       -p 6006:6006 \
       --volume=`pwd`:/workdir \
       --workdir=/workdir \
       datagovsg/tensorflow-fold:v1.12.0-rc2-gpu
```
To find out the token for the jupyter notebook, run the following.
```bash
docker logs CONTAINER_ID
```

## Tutorials
The notebook `cnn-sentiment.ipynb` implements [Convolutional Neural Networks for Sentence Classification](https://arxiv.org/abs/1408.5882). This notebook is unable to run on GPU if the driver version is too low. For NVIDIA GTX 1070 GPU, the minimum NVIDIA driver version is 410.XX.

The notebook `tree-lstm-sentiment.ipynb` implements [TreeLSTMS](https://arxiv.org/abs/1503.00075) and was modified from the original [tensorflow/fold](https://github.com/tensorflow/fold)
