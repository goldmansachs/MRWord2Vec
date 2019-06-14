/*
 Licensed to the Apache Software Foundation (ASF) under one
 or more contributor license agreements.  See the NOTICE file
 distributed with this work for additional information
 regarding copyright ownership.  The ASF licenses this file
 to you under the Apache License, Version 2.0 (the
 "License"); you may not use this file except in compliance
 with the License.  You may obtain a copy of the License at

 http://www.apache.org/licenses/LICENSE-2.0

 Unless required by applicable law or agreed to in writing,
 software distributed under the License is distributed on an
 "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
 KIND, either express or implied.  See the License for the
 specific language governing permissions and limitations
 under the License.
 */
package com.gs.mrword2vec.mapreduce.train;

import com.gs.mrword2vec.Word2Vec;
import com.gs.mrword2vec.mapreduce.utils.Settings;
import com.gs.mrword2vec.mapreduce.utils.Vector;
import com.gs.mrword2vec.utils.Constants;
import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.fs.FileSystem;
import org.apache.hadoop.fs.Path;
import org.apache.hadoop.io.LongWritable;
import org.apache.hadoop.io.Text;
import org.apache.hadoop.mapreduce.Mapper;
import org.apache.log4j.Logger;

import java.io.BufferedReader;
import java.io.IOException;
import java.io.InputStreamReader;

/**
 * Word2VecMapper.java is the mapper class for the MR job which trains the
 * word2vec model.
 * Each mapper trains a separate model on all the data it gets. Thus, the number
 * of models trained is equal to the number of mappers spawned. This number can
 * be controlled by the user.
 * A word2vec object is initialized in the setup. The map function iterates over
 * the data set sentence by sentence. The model is updated incrementally at a
 * sentence level. The cleanup function writes the model to context.
 */
public class Word2VecMapper extends Mapper<LongWritable, Text, Text, Vector> {

  public enum WORD2VEC_MAPPER_COUNTERS {
    SENTENCES,
    WORDS
  }

  private static Word2Vec word2Vec;
  private static int sentenceCount = 0;
  private static Runtime runtime = Runtime.getRuntime();
  private static final Logger log = Logger.getLogger(Word2VecMapper.class);

  /**
   * Called once at the beginning of the task.
   * A Word2Vec object is initialized here which map function trains.
   */
  @Override
  protected void setup(Context context) throws IOException {
    log.info("Entered setup.");
    log.info("Runtime max memory = " + runtime.maxMemory());
    log.info("Runtime free memory = " + runtime.freeMemory());
    log.info("Runtime total memory = " + runtime.totalMemory());
    log.info("Runtime used memory = " + (runtime.totalMemory() - runtime.freeMemory()));
    log.info("Runtime used memory = Runtime total memory - Runtime free memory");
    // ### Reading configuration. ###
    Configuration conf = context.getConfiguration();
    String huffmanTreeFile = conf.get(Settings.HUFFMAN_TREE);
    String previousModelFile = conf.get(Settings.PREVIOUS_MODEL);
    // Reading vector size from configuration using Settings.VECTOR_SIZE.
    // If it isn't specified in the driver class, then Settings.VECTOR_SIZE_DEFAULT
    // is used.
    int vectorSize = conf.getInt(Settings.VECTOR_SIZE, Settings.VECTOR_SIZE_DEFAULT);
    // The number of epochs is hard-coded to 1.
    // It can be made a variable, for example, by saving the data to some
    // temporary file and then reading it line by line multiple times.
    int epochs = 1;
    // The current iteration number. This will equal the index of the current job.
    // Default value is 1, if nothing is specified in the driver class.
    int iter = conf.getInt(Settings.ITERATION_NUMBER, 1);
    // Total number of iterations we need to run. This will equal number of
    // MapReduce jobs needed for training.
    // Default value is 1, if nothing is specified in the driver class.
    int iterations = conf.getInt(Settings.ITERATIONS, 1);

    // ### Initializing Word2Vec object. ###
    // Linearly decreasing learning rate.
    // [learningRateStart, learningRateEnd] will be used as the learning rate
    // for the current iteration. Within this iteration learning rate will start
    // at learningRateStart, and end at learningRateEnd.
    double learningRateStart = Settings.LEARNING_RATE_START *
      (1 - (iter - 1.0) / iterations);
    double learningRateEnd = Settings.LEARNING_RATE_START *
      (1 - (iter * 1.0) / iterations);
    learningRateStart = Math.max(learningRateStart, Settings.LEARNING_RATE_END);
    learningRateEnd = Math.max(learningRateEnd, Settings.LEARNING_RATE_END);
    word2Vec = new Word2Vec(vectorSize, epochs, learningRateStart, learningRateEnd);

    // ### Reading Huffman Tree. ###
    FileSystem fs = FileSystem.get(conf);
    BufferedReader br1 = new BufferedReader(new InputStreamReader(fs.open(
      new Path(huffmanTreeFile))));
    word2Vec.readHuffmanTree(br1);
    br1.close();

    // ### Reading model from previous iteration. ###
    BufferedReader br2 = new BufferedReader(new InputStreamReader(fs.open(
      new Path(previousModelFile))));
    word2Vec.readMatrices(br2);
    br2.close();
    log.info("Setup complete.");
  }

  @Override
  protected void map(LongWritable key, Text value, Context context) {
    // Read a line, and train on it.
    String[] sentence = value.toString().split(Constants.WHITESPACE_REGEX);
    // All the heavy lifting happens in the next line.
    word2Vec.trainSentence(sentence);
    context.getCounter(WORD2VEC_MAPPER_COUNTERS.SENTENCES).increment(1);
    context.getCounter(WORD2VEC_MAPPER_COUNTERS.WORDS).increment(sentence.length);
    if (sentenceCount % 100000 == 0) {
      log.info("Sentences processed = " + sentenceCount);
      log.info("Runtime used memory = " +
        (runtime.totalMemory() - runtime.freeMemory()));
    }
    sentenceCount++;
  }

  /**
   * Called once at the end of the task.
   * (word, vector) for all words in the vocabulary is written to context here.
   */
  @Override
  protected void cleanup(Context context) throws IOException, InterruptedException {
    log.info("Entered cleanup.");
    // ### Sending vectors to reducers. ###
    Text key;
    // syn0 matrix.
    for (int i = 0; i < word2Vec.vocabSize; i++) {
      // syn0's vectors have their key ending with "0".
      key = new Text(i + " 0");
      // Create a Vector object containing values syn0[x, x+1, x+2, ..., x+n-1],
      // where x = i * word2Vec.vectorSize and n = word2Vec.vectorSize.
      Vector vec = new Vector(word2Vec.syn0, i * word2Vec.vectorSize,
        (i + 1) * word2Vec.vectorSize);
      context.write(key, vec);
    }
    // syn1 matrix.
    for (int i = 0; i < word2Vec.vocabSize; i++) {
      // syn1's vectors have their key ending with "1".
      key = new Text(i + " 1");
      // Create a Vector object containing values syn1[x, x+1, x+2, ..., x+n-1],
      // where x = i * word2Vec.vectorSize and n = word2Vec.vectorSize.
      Vector vec = new Vector(word2Vec.syn1, i * word2Vec.vectorSize,
        (i + 1) * word2Vec.vectorSize);
      context.write(key, vec);
    }
    log.info("Exiting cleanup.");
  }
}