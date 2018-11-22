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

# Tutorials
## 1. CNN
The notebook `cnn-sentiment.ipynb` implements [Convolutional Neural Networks for Sentence Classification](https://arxiv.org/abs/1408.5882). This notebook is unable to run on GPU if the driver version is too low. For NVIDIA GTX 1070 GPU, the minimum NVIDIA driver version is 410.XX.

### CNN Model Modifcations
Processing of SST:
  * Replaced u"-", u"\\/" and u"\xa0" in a word with u" " if the word is not found in word2vec
  * If a lower cased word is not found in word2vec, we use the embeddings of a variant casing for the word if it exists in word2vec

A few modifications were made to the CNN model:
  * we increased the filter size to 300
  * we added a dropout layer to the input
  * included l2 regularization for final linear layer weights
  * reset the word embedding for the unknown token to zero after each run
  * included a filter padding at the front and back of a sentence (that is trained) before padding with zeros to make all sentences of the same length (filter padding is of length 4 if sentence is longer than 15 tokens and is of length 2i if sentence is shorter than 15 tokens)

### Results
The following results were obtained for ten trials.

### Test Scores
Dataset | `mean` | `standard deviation` | `minimum` | `maximum`
--- | --- | --- | --- | ---
SST1 | 50.1448 | 0.9097 | 48.914 | 51.672


## 2. Tree-LSTM
The notebook `tree-lstm-sentiment.ipynb` implements [TreeLSTMS](https://arxiv.org/abs/1503.00075) and was modified from the original [tensorflow/fold](https://github.com/tensorflow/fold)
