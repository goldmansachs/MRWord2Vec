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

import com.google.common.base.Preconditions;
import com.gs.mrword2vec.VocabWord;
import com.gs.mrword2vec.Word2Vec;
import com.gs.mrword2vec.mapreduce.utils.Vector;
import com.gs.mrword2vec.mapreduce.wordcount.WordCountCombiner;
import com.gs.mrword2vec.mapreduce.wordcount.WordCountMapper;
import com.gs.mrword2vec.mapreduce.wordcount.WordCountReducer;
import com.gs.mrword2vec.utils.Constants;
import com.gs.mrword2vec.utils.Utils;
import com.gs.mrword2vec.mapreduce.utils.Settings;
import com.gs.mrword2vec.mapreduce.utils.Word2VecOptions;
import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.conf.Configured;
import org.apache.hadoop.fs.FSDataOutputStream;
import org.apache.hadoop.fs.FileSystem;
import org.apache.hadoop.fs.FileUtil;
import org.apache.hadoop.fs.Path;
import org.apache.hadoop.io.LongWritable;
import org.apache.hadoop.io.Text;
import org.apache.hadoop.mapreduce.Job;
import org.apache.hadoop.mapreduce.lib.input.TextInputFormat;
import org.apache.hadoop.mapreduce.lib.output.TextOutputFormat;
import org.apache.hadoop.util.Tool;
import org.apache.hadoop.util.ToolRunner;
import org.apache.log4j.Logger;

import java.io.*;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

/**
 * This class trains a word2vec model.
 * As input you need a data set, which is a file containing a sentence in each line.
 * A word count MapReduce job constructs a file which contains words and their
 * frequencies. This file is used to construct a Huffman tree, which then gets
 * used in the training.
 * Before starting training, we construct a seed file, which contains
 * initialization for weight matrices.
 * Training happens in one or more iterations, each iteration
 * being a MapReduce job using the results of the previous job.
 * Each iteration trains the neural net by doing one epoch in every mapper on
 * its data. The results from all the mapper are then combined by averaging
 * vectors.
 * The final model parameters are then stored in the specified output directory.
 * <br><br>
 *
 * USAGE:<br>
 * Before using this class to train a word2vec model, you need to collect the
 * training data in a single text file. The format of the text file must be such
 * that it contains one sentence per line. The sentence should be a string of
 * words separated by space.<br>
 * After setting the HADOOP_CLASSPATH, run this class like:<br>
 * hadoop jar /path_to_jar/jarName.jar
 * com.gs.compliance.sag.word2vec.mapreduce.Word2VecDriver
 * -libjars=/path_to_jar1,/path_to_jar2
 * -queue_name=my_queue_name
 * -input_path=/path_to_training_text_file
 * -output_dir=/path_to_store_trained_model
 * -min_count=500 -max_vocab_size=100000 -num_parts=100 -num_reducers=5
 * -iterations=10 -epochs=1 -mapper_memory_mb=6144
 */
public class Word2VecDriver extends Configured implements Tool {

  private static final Logger log = Logger.getLogger(Word2VecDriver.class);

  public String queueName = null;
  public String inputPath = null;
  public String outputDir = null;
  public String tempDir = null;
  public String wordCountFile = null;
  public String huffmanTreeFile = null;
  public String finalHuffmanTreeFile = null;
  public String finalModelFile = null;
  public int numMappers = Settings.NUM_MAPPERS_DEFAULT;
  public int numReducers = Settings.NUM_REDUCERS_DEFAULT;
  public int mapperSize = Settings.MAPPER_MEMORY_MB_DEFAULT;
  public int reducerSize = Settings.REDUCER_MEMORY_MB_DEFAULT;
  public int maxVocabSize = Settings.MAX_VOCAB_SIZE_DEFAULT;
  public int minCount = Settings.MIN_COUNT_DEFAULT;
  public int iterations = Settings.ITERATIONS_DEFAULT;
  public int epochs = Settings.EPOCHS_DEFAULT;
  public int vectorSize = Settings.VECTOR_SIZE_DEFAULT;
  public int vocabSize = 0;
  public long trainWordsCount = 0L;

  /**
   * This function handles all aspects of training a word2vec model.
   * That includes running a word count job to be able to decide on vocabulary,
   * constructing Huffman tree, and then running multiple MapReduce jobs for
   * training, one for each iteration.
   */
  @Override
  public int run(String[] args) throws Exception {
    Configuration conf = new Configuration(getConf());
    // Read command line options.
    Word2VecOptions w2vOptions = new Word2VecOptions(args);
    queueName = w2vOptions.queueName;
    inputPath = w2vOptions.inputPath;
    outputDir = w2vOptions.outputDir;
    tempDir = outputDir + Constants.TEMP_DIR;
    wordCountFile = tempDir + Constants.WORDCOUNT_FILE_PATH;
    huffmanTreeFile = tempDir + Constants.HUFFMAN_TREE_PATH;
    finalHuffmanTreeFile = outputDir + Constants.HUFFMAN_TREE_PATH;
    finalModelFile = outputDir + Constants.MODEL_PATH;
    numMappers = w2vOptions.numMappers;
    numReducers = w2vOptions.numReducers;
    mapperSize = w2vOptions.mapperSize;
    reducerSize = w2vOptions.reducerSize;
    maxVocabSize = w2vOptions.maxVocabSize;
    minCount = w2vOptions.minCount;
    iterations = w2vOptions.iterations;
    epochs = w2vOptions.epochs;
    vectorSize = w2vOptions.vectorSize;

    boolean wcStatus = doWordCount(conf);
    if (!wcStatus) {
      log.error("Failed to do word count.");
      return 1;
    }

    boolean huffStatus = buildHuffmanTree(conf);
    if (!huffStatus) {
      log.error("Failed to build Huffman tree.");
      return 1;
    }

    // This seed file contains the initialization for weight matrices used in
    // training.
    writeSeedFile(conf);

    // Train the model.
    boolean trainStatus = train(conf);
    if (!trainStatus) {
      log.error("Failed to train the model.");
      return 1;
    }

    log.info("Two files created:");
    log.info("Matrices are stored at " + finalModelFile);
    log.info("Huffman tree is stored at " + finalHuffmanTreeFile);

    return 0;
  }

  public void setupMRConfigs(Configuration conf) {
    conf.set("mapreduce.job.queuename", queueName);
  }

  /**
   * This function runs a MapReduce job which reads the input text file and does
   * word count on it. The output file is used in constructing vocabulary for
   * the word2vec model.
   * @param conf Configuration object.
   * @return Returns true if the job ran successfully, false otherwise.
   * @throws IOException
   * @throws ClassNotFoundException
   * @throws InterruptedException
   */
  public boolean doWordCount(Configuration conf) throws IOException,
    ClassNotFoundException, InterruptedException {
    log.info("Doing word count.");
    boolean status = false;
    // ##############################
    // ### Setting configuration. ###
    // ##############################
    setupMRConfigs(conf);
    // Words with frequency less than this won't be printed in the output of the
    // MapReduce job.
    conf.setInt(Settings.MIN_COUNT, minCount);
    // Setting paths.
    Path inputFilePath = new Path(inputPath);
    // Temporary directory to store the output of the MapReduce job.
    String outputWCDir = wordCountFile + "_dir";
    Path outputDirPath = new Path(outputWCDir);
    FileSystem fs = FileSystem.get(conf);
    // Delete global output directory if it already exists.
    if (fs.exists(new Path(outputDir))) {
      boolean bool = fs.delete(new Path(outputDir), true);
      Preconditions.checkArgument(bool, "Failed to delete old output directory.");
    }
    // Merged files will be stored here.
    Path finalOutputPath = new Path(wordCountFile);
    // ###############################
    // ### Setting job parameters. ###
    // ###############################
    Job job = Job.getInstance(conf, "Word2Vec: Word count for building the " +
      "Huffman tree");
    job.setJarByClass(Word2VecDriver.class);
    job.setMapperClass(WordCountMapper.class);
    job.setCombinerClass(WordCountCombiner.class);
    job.setReducerClass(WordCountReducer.class);
    job.setNumReduceTasks(10);
    job.setMapOutputKeyClass(Text.class);
    job.setMapOutputValueClass(LongWritable.class);
    job.setOutputKeyClass(Text.class);
    job.setOutputValueClass(LongWritable.class);
    job.setInputFormatClass(TextInputFormat.class);
    job.setOutputFormatClass(TextOutputFormat.class);
    TextInputFormat.addInputPath(job, inputFilePath);
    TextOutputFormat.setOutputPath(job, outputDirPath);
    // If job completes successfully, merge all files into one.
    if (job.waitForCompletion(true)) {
      // Delete if it already exists.
      fs.delete(finalOutputPath, true);
      status = FileUtil.copyMerge(fs, outputDirPath, fs, finalOutputPath, true,
        conf, null);
      log.info("Word count complete.");
    } else {
      log.error("Word count job failed.");
    }
    return status;
  }

  /**
   * This function runs a MapReduce job to do word count over the input data.
   * Then it builds and saves a Huffman tree, which gets used in training later
   * on.
   * @param conf Configuration object.
   */
  public boolean buildHuffmanTree(Configuration conf)
    throws IOException, ClassNotFoundException, InterruptedException {
    // Word count file.
    Path finalOutputPath = new Path(wordCountFile);
    FileSystem fs = FileSystem.get(conf);
    // ##################################
    // ### Reading the file to a Map. ###
    // ##################################
    Map<String, Long> wordCountMap = new HashMap<>();
    BufferedReader br = new BufferedReader(new InputStreamReader(fs.open(
      finalOutputPath)));
    String line;
    // Each line contains word and its frequency separated by a tab.
    while ((line = br.readLine()) != null) {
      String[] wordFreq = line.split("\\s+");
      // wordFreq[0] is the word, and wordFreq[1] is its frequency.
      wordCountMap.put(wordFreq[0], Long.parseLong(wordFreq[1]));
    }
    br.close();
    // Sort in descending order by count.
    Map<String, Long> sortedWordCountMap = Utils.sortByValue(wordCountMap);
    // ############################
    // ### Create Huffman tree. ###
    // ############################
    Word2Vec word2Vec = new Word2Vec();
    word2Vec.vectorSize = vectorSize;
    word2Vec.vocabSize = Math.min(maxVocabSize, sortedWordCountMap.size());
    List<VocabWord> vocabList = new ArrayList<>();
    // Populate entries in vocabList using sortedWordCountMap.
    int c = 0;
    // Read entries starting with most frequent.
    for (Map.Entry<String, Long> e: sortedWordCountMap.entrySet()) {
      // If the number of entries exceed maxVocabSize, read only maxVocabSize
      // most frequent.
      if (c == maxVocabSize) {
        break;
      }
      vocabList.add(new VocabWord(e.getKey(), e.getValue(),
        new int[word2Vec.maxCodeLength], new int[word2Vec.maxCodeLength], 0));
      trainWordsCount += e.getValue();
      c++;
    }
    word2Vec.vocab = vocabList;
    Preconditions.checkArgument(vocabList.size() == word2Vec.vocabSize,
      "Error in vocab size.");
    vocabSize = word2Vec.vocabSize;
    word2Vec.createHuffmanTree();
    // ###############################
    // ### Write the Huffman tree. ###
    // ###############################
    Path huffmanTreePath = new Path(huffmanTreeFile);
    FSDataOutputStream outputStream = FileSystem.get(conf).create(
      huffmanTreePath, true);
    BufferedWriter bw = new BufferedWriter(new OutputStreamWriter(outputStream));
    word2Vec.writeHuffmanTree(bw);
    bw.close();
    outputStream.close();
    return true;
  }

  /**
   * This function creates a file, called a seed file, which contains
   * initialization for the weight matrices used in word2vec. The format of the
   * file is same as the format of the file produced at the end of each iteration
   * of training.
   * @param conf Configuration object.
   * @throws IOException Throws an IOException if unable to create the seed file.
   */
  public void writeSeedFile(Configuration conf) throws IOException {
    log.info("Writing seed file.");
    Path seedFilePath = new Path(tempDir + "/0");
    Word2Vec word2Vec = new Word2Vec();
    word2Vec.vectorSize = vectorSize;
    word2Vec.vocabSize = vocabSize;
    word2Vec.initializeMatrices();
    FSDataOutputStream outputStream = FileSystem.get(conf).create(seedFilePath, true);
    BufferedWriter bw = new BufferedWriter(new OutputStreamWriter(outputStream));
    word2Vec.writeMatrices(bw);
    bw.close();
    outputStream.close();
    log.info("Seed file written.");
  }

  /**
   * This function trains a word2vec model by running one or more iterations,
   * each iteration using the results of the previous one.
   * @param conf Configuration object.
   */
  public boolean train(Configuration conf)
    throws IOException, ClassNotFoundException, InterruptedException {
    log.info("Starting training.");
    for (int iter = 1; iter <= iterations; iter++) {
      log.info("Starting job " + iter + ".");
      setupMRConfigs(conf);
      conf.setInt("mapreduce.map.memory.mb", mapperSize);
      conf.set("mapreduce.map.java.opts", "-Xmx" + (int) (0.8 * mapperSize) + "m");
      conf.setInt("mapreduce.reduce.memory.mb", reducerSize);
      conf.set("mapreduce.reduce.java.opts", "-Xmx" + (int) (0.8 * reducerSize) + "m");

      FileSystem fs = FileSystem.get(conf);
      Path inputFilePath = new Path(inputPath);
      Path outputDirPath = new Path(tempDir + "/" + iter + "_dir");

      // Calculating split size, to get numMappers mappers.
      long inputFileSize = fs.getFileStatus(inputFilePath).getLen();
      long splitSize = (long) Math.ceil(inputFileSize / (1.0 * numMappers));
      conf.setLong("mapreduce.input.fileinputformat.split.minsize", splitSize);

      // Setting various properties needed during training.
      conf.set(Settings.HUFFMAN_TREE, huffmanTreeFile);
      conf.set(Settings.PREVIOUS_MODEL, tempDir + "/" + (iter-1));
      conf.setInt(Settings.VECTOR_SIZE, vectorSize);
      conf.setInt(Settings.EPOCHS, epochs);
      conf.setInt(Settings.ITERATION_NUMBER, iter);
      conf.setInt(Settings.ITERATIONS, iterations);

      Job job = Job.getInstance(conf,
        "Word2Vec: Iteration " + iter + "/" + iterations);
      job.setJarByClass(Word2VecDriver.class);
      job.setMapperClass(Word2VecMapper.class);
      job.setReducerClass(Word2VecReducer.class);
      job.setNumReduceTasks(numReducers);

      job.setMapOutputKeyClass(Text.class);
      job.setMapOutputValueClass(Vector.class);
      job.setOutputKeyClass(Text.class);
      job.setOutputValueClass(Text.class);

      job.setInputFormatClass(TextInputFormat.class);
      job.setOutputFormatClass(TextOutputFormat.class);
      TextInputFormat.addInputPath(job, inputFilePath);
      TextOutputFormat.setOutputPath(job, outputDirPath);

      // If job completes successfully, merge all files into one.
      Path finalOutputPath = new Path(tempDir + "/" + iter);
      if (job.waitForCompletion(true)) {
        // Delete if it already exists.
        fs.delete(finalOutputPath, true);
        FileUtil.copyMerge(fs, outputDirPath, fs, finalOutputPath, true, conf, null);
      } else {
        log.error("Job " + iter + " failed.");
        return false;
      }
      log.info("Job " + iter + " complete.");
    }
    // Move the final model from temporary directory to final directory.
    FileSystem fs = FileSystem.get(conf);
    // Copy huffman tree.
    fs.rename(new Path(huffmanTreeFile), new Path(finalHuffmanTreeFile));
    // Copy final model.
    fs.rename(new Path(tempDir + "/" + iterations), new Path(finalModelFile));
    // Delete temporary directory.
    fs.delete(new Path(tempDir), true);
    return true;
  }

  public static void main(String[] args) throws Exception {
    System.exit(ToolRunner.run(new Word2VecDriver(), args));
  }
}