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
package com.gs.mrword2vec;

import com.gs.mrword2vec.utils.Constants;
import com.gs.mrword2vec.utils.Tuple;
import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.conf.Configured;
import org.apache.hadoop.fs.FSDataOutputStream;
import org.apache.hadoop.fs.FileSystem;
import org.apache.hadoop.fs.Path;
import org.apache.hadoop.hbase.HBaseConfiguration;
import org.apache.hadoop.util.Tool;
import org.apache.hadoop.util.ToolRunner;
import org.apache.log4j.Logger;

import java.io.*;
import java.util.List;
import java.util.Set;

/**
 * This class is used to compute the k nearest neighbours for all words in the
 * vocabulary of a trained model. It needs two input parameters:
 * 1. modelDir - Path of the model directory. This directory should contain the
 * Huffman tree (which is needed for hierarchical softmax) and the matrices.
 * They allow us to reconstruct the word2vec model--all words and their
 * vectors--which is needed to compute neighbours.
 * The matrices file contains values of syn0 and syn1 matrices.
 * 2. k - Number of nearest neighbours to be computed for each word.
 * The output file is saved in the modelDir with name "nearestNeighbours".
 *
 * Example usage:
 * hadoop jar /path_to_jar/word2vec.jar NearestNeighbours
 * -D modelDir=/path_to_model/model_file -D k=20
 */
public class NearestNeighbours extends Configured implements Tool {

  private static final Logger log = Logger.getLogger(NearestNeighbours.class);

  public static void createFile(String modelDir, Configuration conf, int k)
    throws Exception {
    log.info("modelDir = " + modelDir);
    log.info("k = " + k);
    String modelFile = modelDir + Constants.MODEL_PATH;
    String huffmanFile = modelDir + Constants.HUFFMAN_TREE_PATH;
    String outputFile = modelDir + Constants.NEAREST_NEIGHBOURS_PATH;
    // Initializing word2vec object.
    Word2Vec word2Vec = new Word2Vec();
    // Reading Huffman tree.
    FileSystem fs = FileSystem.get(conf);
    BufferedReader br1 = new BufferedReader(new InputStreamReader(fs.open(
      new Path(huffmanFile))));
    word2Vec.readHuffmanTree(br1);
    br1.close();
    // Reading matrices syn0 and syn1.
    BufferedReader br2 = new BufferedReader(new InputStreamReader(fs.open(
      new Path(modelFile))));
    word2Vec.readMatrices(br2);
    br2.close();
    // Populate norms. For every word in the vocabulary, compute the l2 norm of
    // its vector, and store it in an array.
    word2Vec.populateNorms();
    // Write the nearest neighbours to output file.
    Path outputPath = new Path(outputFile);
    FSDataOutputStream outputStream = FileSystem.get(conf).create(outputPath, true);
    BufferedWriter bw = new BufferedWriter(new OutputStreamWriter(outputStream));
    Set<String> vocabSet = word2Vec.getVocab();
    for (String w: vocabSet) {
      // Find the k nearest neighbours of word w.
      List<Tuple> nearestNeighbours = word2Vec.mostSimilar(w, k);
      bw.write(w + "\t" + nearestNeighbours + "\n");
    }
    bw.close();
    outputStream.close();
  }

  @Override
  public int run(String[] args) throws Exception {
    Configuration conf = HBaseConfiguration.create(getConf());
    // Read the parameters.
    // Path of the directory where a model is saved.
    String modelDir = conf.get("modelDir");
    // Number of nearest neighbours required for each vocabulary word.
    // A default value of 20 is used if it isn't specified.
    int k = conf.getInt("k", 20);
    createFile(modelDir, conf, k);
    log.info("Nearest neighbours file stored at " + modelDir +
      Constants.NEAREST_NEIGHBOURS_PATH);
    return 1;
  }

  public static void main(String[] args) throws Exception {
    System.exit(ToolRunner.run(new NearestNeighbours(), args));
  }
}