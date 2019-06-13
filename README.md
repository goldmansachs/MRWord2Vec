# MRWord2Vec

## Introduction

MRWord2Vec is a [MapReduce](https://en.wikipedia.org/wiki/MapReduce) implementation of 
[Word2Vec](https://en.wikipedia.org/wiki/Word2vec).

It's a Java library, and can be used to train Word2Vec models in two ways:
- in the MapReduce framework, and
- on a single machine.

The novelty of this library is the MapReduce implementation and the extremely 
low memory footprint.
The amount of memory needed is approximately twice the amount of memory needed 
to store the Word2Vec vectors.
This is achieved by implementing incremental training - training on one sentence at a time.


## Quick-start
This section will demonstrate how to quickly get this library up and running.
For a more comprehensive guide, read the rest of the sections.

The following shell script will train a Word2Vec model and compute the nearest neighbours:
```
# The location of the MRWord2Vec's jar file.
mrword2vec_jar=[path_to_mrword2vec.jar]
# The location of the directory containing all the dependency jars. 
# This will, for example, include the jar file for jblas.
dependency_jars=[path_to_directory_containing_all_dependency_jars]
# Setting up the HADOOP_CLASSPATH by adding all the dependencies and the jar file of this library.
HADOOP_CLASSPATH=$HADOOP_CLASSPATH:$(echo $dependency_jars/*.jar | tr ' ' ':'):$mrword2vec_jar
# The text file used for training. See the section on Input to learn how this file is formatted.
inputFile=[path_to_training_text_file]
# The directory where the model is to be saved.
modelPath=[path_where_the_model_is_to_be_saved]

# Running the class Word2VecDriver. This trains the model.
hadoop jar $mrword2vec_jar com.gs.mrword2vec.mapreduce.train.Word2VecDriver -libjars=$(echo $dependency_jars/*.jar | tr ' ' ',') -queue_name=my_queue_name -input_path=$inputFile -output_dir=$modelPath -min_count=100 -max_vocab_size=100000 -num_parts=100 -num_reducers=5 -iterations=10 -mapper_memory_mb=6144
# Computing the nearest neighbours.
hadoop jar $mrword2vec_jar com.gs.mrword2vec.NearestNeighbours -D modelDir=$modelPath -D k=20
```

The command above will run the MapReduce jobs to train a Word2Vec model on the specified
training file and will save the learned model parameters.
It will run 10 MapReduce jobs (since `iterations` is set to 10), with each
job using 100 mappers (since `num_parts` is set to 100) and 5 reducers.
Since there are 100 mappers, 100 independent Word2Vec models are trained, one in each mapper.
Each of these models is trained on 1/100th of the data.

The last command is then used to get the top k nearest neighbours for
every word in the vocabulary of the trained model.
When it completes, a file will be created storing the nearest neighbours for every
vocabulary word in the same directory as the saved model (inside `modelPath`).


## Background

#### Word2Vec
Word2Vec is a model to learn real-vector representations of words from text data.
These are low-dimensional (i.e., of dimension much smaller than the vocabulary size) vectors with
semantic meaning, unlike one-hot vectors.
Similar words will have similar vectors, where similarity of two vectors is 
computed using [cosine similarity](https://en.wikipedia.org/wiki/Cosine_similarity).

To learn these vector representations Word2Vec uses text data.
Word2Vec is an unsupervised algorithm, and needs text data in the form of sentences.

The model this library implements is called the skip-gram model. 
The training objective of skip-gram is to find word vectors that maximize the 
probability of accurately predicting the context given the word.

Hierarchical softmax is used to speed up the computation. 
(Another technique to do that is negative sampling, which isn't implemented here.)

#### Distributed Word2Vec
The idea, used here, of sharding the data and averaging the resulting vectors, appears
in Spark's implementation of Word2Vec.
The algorithm is as follows:

1. Start with a model whose weights are randomly initialized.
2. Partition the data into N parts.
3. Repeat K times:
    1. Spawn N mappers, with each mapper training a Word2Vec model on its shard, 
    starting from the previous model, doing gradient descent.
    2. Combine all the N models, by taking an _average_ of all vectors for each word.
    3. Save the model.

There is a trade-off here between N and K. 
The larger the N, the faster the training, since each mapper trains on 1/N of the data.
However, too large an N and the quality will suffer.
Similarly, the larger the K, the better the model. However, larger K necessitates 
greater training time. If K = N, then there is no
saving in time by distributing the training.

## Input
The input to this library is a single text file containing pre-processed data.
Each line of this file must contain exactly one sentence,
the words of which are separated by spaces.
For example, consider the following small text file:
```text
Word2vec is a group of related models that are used to produce word embeddings. These models are shallow, two-layer neural networks that are trained to reconstruct linguistic contexts of words. Word2vec takes as its input a large corpus of text and produces a vector space, typically of several hundred dimensions, with each unique word in the corpus being assigned a corresponding vector in the space. Word vectors are positioned in the vector space such that words that share common contexts in the corpus are located in close proximity to one another in the space.
```

After pre-processing it would look like:
```text
word2vec is a group of related models that are used to produce word embeddings
these models are shallow two-layer neural networks that are trained to reconstruct linguistic contexts of words
word2vec takes as its input a large corpus of text and produces a vector space typically of several hundred dimensions with each unique word in the corpus being assigned a corresponding vector in the space
word vectors are positioned in the vector space such that words that share common contexts in the corpus are located in close proximity to one another in the space
```

Here, we have put each sentence on a different line, removed all punctuation, 
and converted everything to lower case since we don't want to learn separate word 
embeddings for lower case and upper case words. 


## Usage

#### MapReduce framework

The class used to run MRWord2Vec is `Word2VecDriver.java`.
This class takes some arguments and runs the MapReduce jobs.

#####Arguments

1. **queue\_name** : A required argument. Example usage: `-queue_name=my_queue_name`. 
The name of the queue on which to run the MapReduce jobs.
2. **input\_path** : A required argument. Example usage: `-input_path=/user/my_name/input_file`. 
The HDFS path of the text file to be used for training. The file must contain one sentence per line. 
A sentence is a sequence of words separated by spaces. 
The text must be pre-processed, i.e., all words should be in lower case and there should be no punctuation. 
This file will also be used to build the vocabulary. 
Hence the input file may contain more distinct words than desired for the vocabulary.
3. **output\_dir** : A required argument. Example usage: `-output_dir=/user/my_name/output_dir`. 
The HDFS path of the directory where the model will be saved. 
During training, a temporary directory, `temp`, is created inside `output_dir`, 
where model parameters are saved during training. 
After the training is complete, the temporary directory is deleted and `output_dir` 
will contain two files, one containing the Huffman tree encoding of the vocabulary words 
and the other containing the model parameters (with edge weights stored as matrices). 
These two files are necessary to compute nearest neighbours, if so desired.
4. **num\_parts** : An optional argument. Default = 10. Example usage: `-num_parts=100`. 
The number of parts into which to split the input corpus. 
This is equal to the number of mappers spawned, since each mapper is responsible to train on one split. 
The higher the `num_parts`, the faster the training will be.
5. **num\_reducers** : An optional argument. Default = 10. Example usage: `-num_reducers=10`. 
The number of reducers to be spawned while training. The reducers are responsible for averaging the vectors. 
Set this number depending on the vocabulary size, with the default value being sufficient for most cases.
6. **mapper\_memory\_mb** : An optional argument. Default = 3072. Example usage: `-mapper_memory_mb=6144`. 
The size of the memory, in megabytes, allocated to each mapper used in training. 
The memory usage of mappers while training is proportional to the product of vocabulary size and vector size. 
Set this number accordingly.
7. **reducer_memory_mb** : An optional argument. Default = 3072. Example usage: `-reducer_memory_mb=6144`. 
The size of the memory, in megabytes, allocated to each reducer 
(used for averaging after the training is complete for that epoch). 
The memory usage of the reducers is proportional to the product of the number of mappers and the vector size. 
The default value should suffice for most cases.
8. **max_vocab_size** : An optional argument. Default = 10,000. Example usage: `-max_vocab_size=50000`. 
This argument limits the vocabulary by choosing the `max_vocab_size` most frequent words from 
the training data file specified by `input_path`. 
This is one of the two parameters one can use to limit the vocabulary size, the other being `min_count`. 
9. **min_count** : An optional argument. Default = 10. Example usage: `-min_count=100`. 
All words with frequency less than `min_count` are discarded. 
This is one of the two parameters one can use to limit the vocabulary size, the other being `max_vocab_size`.
10. **iterations**: An optional argument. Default = 1. Example usage: `-iterations=10`. 
This specifies the number of times you want to train your model on the training data in a distributed manner. 
This corresponds to K in the algorithm described in the section Distributed Word2Vec above. 
You will need to experiment to find the best value of this hyperparameter. 
In our tests, 10 iterations work fine when `num_parts` equals 100 for a data set of 1 TB.
11. **vector_size** : An optional argument. Default = 300. Example usage: `-vector_size=200`. 
The dimensionality of the word2vec vectors. 
Larger vectors will usually give better results, but around 300 the marginal gain 
in quality is offset by the increase in the training time. 
For most cases the default value of 300 will suffice.     

After adding all the dependencies to `HADOOP_CLASSPATH`, run the driver class like this:
```text
hadoop jar [path_to_mrword2vec.jar] com.gs.mrword2vec.mapreduce.train.Word2VecDriver -libjars=$(echo [path_of_directory_containing_all_dependency_jars]/*.jar | tr ' ' ',') -queue_name=my_queue_name -input_path=[path_to_training_text_file] -output_dir=[path_to_save_model] -min_count=100 -max_vocab_size=100000 -num_parts=100 -num_reducers=5 -iterations=10 -mapper_memory_mb=6144
```

#### Single machine framework

It's also possible to train a Word2Vec model on a single machine (without Hadoop).
The class whose API is relevant for this is `Word2Vec.java`. For example:
```
// Creating a Word2Vec object with vector size = 300, min_count = 15, 
// max_vocab_size = 2000, and 3 epochs. 
Word2Vec word2vec = new Word2Vec(300, 15, 2000, 3);
// sentences is of type List<String[]>.
word2vec.train(sentences);
word2vec.save("path_to_save_the_model");
```
It's also possible to use the function `trainSentence` to train one sentence at a 
time to save memory by reading one sentence at a time from a file and calling `trainSentence`
on the sentence.
```
for(int i = 0; i < N; i++) {
  String[] sentence = getNextSentence();
  word2vec.trainSentence(sentence);
}
```

A saved model can be read using the `read` function.
```
String locationOfSavedModelDir = "/user/my_user/modelDir";
Word2Vec word2vec = new Word2Vec();
word2vec.read(locationOfSavedModelDir);
```

## Dependencies

1. Apache HBase Client
```
<groupId>org.apache.hbase</groupId>
<artifactId>hbase-client</artifactId>
<version>1.1.2</version>
```
2. Apache Hadoop Common
```
<groupId>org.apache.hadoop</groupId>
<artifactId>hadoop-common</artifactId>
<version>2.7.3</version>
```
3. Apache Hadoop MapReduce Core
```
<groupId>org.apache.hadoop</groupId>
<artifactId>hadoop-mapreduce-client-core</artifactId>
<version>2.7.3</version>
```
4. jblas
```
<groupId>org.jblas</groupId>
<artifactId>jblas</artifactId>
<version>1.2.4</version>
```