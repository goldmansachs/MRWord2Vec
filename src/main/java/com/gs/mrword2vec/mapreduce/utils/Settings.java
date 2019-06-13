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
package com.gs.mrword2vec.mapreduce.utils;

/**
 * This interface contains various constants used in MapReduce jobs.
 */
public interface Settings {

  String PATH_INDICATOR = "path";
  String STRING_INDICATOR = "string";
  String INTEGER_INDICATOR = "int";

  String QUEUE_NAME = "queue_name";

  // Path for input file which contains training data.
  // Each line in this file is a sentence. A sentence is a space separated list
  // of words.
  String INPUT_PATH = "input_path";
  // Directory where the model will be saved.
  String OUTPUT_DIR = "output_dir";

  // Number of parts to split input corpus into.
  String NUM_PARTS = "num_parts";
  // Number of mappers = num of parts.
  String NUM_MAPPERS = NUM_PARTS;

  // Number of reducers.
  String NUM_REDUCERS = "num_reducers";

  // Default number of mappers and reducers.
  int NUM_MAPPERS_DEFAULT = 10;
  int NUM_REDUCERS_DEFAULT = 10;

  // Size of mapper.
  String MAPPER_MEMORY_MB = "mapper_memory_mb";
  String REDUCER_MEMORY_MB = "reducer_memory_mb";
  int MAPPER_MEMORY_MB_DEFAULT = 3072;
  int REDUCER_MEMORY_MB_DEFAULT = 3072;

  // Vocabulary size.
  String MAX_VOCAB_SIZE = "max_vocab_size";
  int MAX_VOCAB_SIZE_DEFAULT = 10000;

  // Minimum count of a word for it to be considered part of vocabulary.
  String MIN_COUNT = "min_count";
  int MIN_COUNT_DEFAULT = 10;

  // Number of iterations.
  String ITERATIONS = "iterations";
  int ITERATIONS_DEFAULT = 1;

  // Number of epochs in each iterations.
  String EPOCHS = "epochs";
  int EPOCHS_DEFAULT = 1;

  // Vector size for embeddings.
  String VECTOR_SIZE = "vector_size";
  int VECTOR_SIZE_DEFAULT = 300;

  // Path for Huffman Tree.
  String HUFFMAN_TREE = "huffman";

  // Path for previous model.
  String PREVIOUS_MODEL = "previous_model";

  // Current iteration number.
  String ITERATION_NUMBER = "iteration_number";

  // Initial value of learning rate.
  double LEARNING_RATE_START = 0.025;
  // Minimum value for learning rate.
  double LEARNING_RATE_END = 0.0001;
}