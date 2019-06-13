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

import com.google.common.base.Preconditions;
import org.apache.commons.cli.*;

/**
 * This class handles reading options/parameters provided by user through the
 * command line.
 */
public class Word2VecOptions {

  public String queueName = null;
  public String inputPath = null;
  public String outputDir = null;
  public int numMappers = Settings.NUM_MAPPERS_DEFAULT;
  public int numReducers = Settings.NUM_REDUCERS_DEFAULT;
  public int mapperSize = Settings.MAPPER_MEMORY_MB_DEFAULT;
  public int reducerSize = Settings.REDUCER_MEMORY_MB_DEFAULT;
  public int maxVocabSize = Settings.MAX_VOCAB_SIZE_DEFAULT;
  public int minCount = Settings.MIN_COUNT_DEFAULT;
  public int iterations = Settings.ITERATIONS_DEFAULT;
  public int epochs = Settings.EPOCHS_DEFAULT;
  public int vectorSize = Settings.VECTOR_SIZE_DEFAULT;

  /**
   * Constructor for Word2VecOptions object. It will parse the options/parameters
   * provided by the user through the command line.
   */
  public Word2VecOptions(String[] args) throws ParseException {
    // Options object describes the possible options for a command line.
    Options options = new Options();

    // #########################################################
    // ### Adding various command line arguments to options. ###
    // #########################################################
    // Each of these lines is adding an Option object to "options" object defined
    // above. By adding these, we are able to parse the command line and assign
    // values to various parameters like queue name, input data path, etc.
    // For example, the user could specify queue name by:
    // -queue_name=queue17
    options.addOption(OptionBuilder.withArgName(Settings.STRING_INDICATOR)
      .hasArg().withDescription("Input file path").create(Settings.QUEUE_NAME));
    options.addOption(OptionBuilder.withArgName(Settings.PATH_INDICATOR)
      .hasArg().withDescription("Input file path").create(Settings.INPUT_PATH));
    options.addOption(OptionBuilder.withArgName(Settings.PATH_INDICATOR)
      .hasArg().withDescription("Output directory").create(Settings.OUTPUT_DIR));
    options.addOption(OptionBuilder.withArgName(Settings.INTEGER_INDICATOR)
      .hasArg().withDescription("Number of mappers")
      .create(Settings.NUM_MAPPERS));
    options.addOption(OptionBuilder.withArgName(Settings.INTEGER_INDICATOR)
      .hasArg().withDescription("Number of reducers")
      .create(Settings.NUM_REDUCERS));
    options.addOption(OptionBuilder.withArgName(Settings.INTEGER_INDICATOR)
      .hasArg().withDescription("Mapper memory in MB")
      .create(Settings.MAPPER_MEMORY_MB));
    options.addOption(OptionBuilder.withArgName(Settings.INTEGER_INDICATOR)
      .hasArg().withDescription("Reducer memory in MB")
      .create(Settings.REDUCER_MEMORY_MB));
    options.addOption(OptionBuilder.withArgName(Settings.INTEGER_INDICATOR)
      .hasArg().withDescription("Max vocabulary size")
      .create(Settings.MAX_VOCAB_SIZE));
    options.addOption(OptionBuilder.withArgName(Settings.INTEGER_INDICATOR)
      .hasArg().withDescription("Minimum count of a word in vocabulary")
      .create(Settings.MIN_COUNT));
    options.addOption(OptionBuilder.withArgName(Settings.INTEGER_INDICATOR)
      .hasArg().withDescription("Number of iterations")
      .create(Settings.ITERATIONS));
    options.addOption(OptionBuilder.withArgName(Settings.INTEGER_INDICATOR)
      .hasArg().withDescription("Number of epochs")
      .create(Settings.EPOCHS));
    options.addOption(OptionBuilder.withArgName(Settings.INTEGER_INDICATOR)
      .hasArg().withDescription("Vector size")
      .create(Settings.VECTOR_SIZE));

    CommandLineParser parser = new GnuParser();
    CommandLine cmdLine = parser.parse(options, args);

    // ###########################################
    // ### Reading the command line arguments. ###
    // ###########################################
    if (cmdLine.hasOption(Settings.QUEUE_NAME)) {
      queueName = cmdLine.getOptionValue(Settings.QUEUE_NAME);
    } else {
      // If not initialized.
      throw new ParseException("Parsing failed due to \"" + Settings.QUEUE_NAME
        + "\" not initialized.");
    }

    if (cmdLine.hasOption(Settings.INPUT_PATH)) {
      inputPath = cmdLine.getOptionValue(Settings.INPUT_PATH);
    } else {
      // If not initialized.
      throw new ParseException("Parsing failed due to \"" + Settings.INPUT_PATH
        + "\" not initialized.");
    }

    if (cmdLine.hasOption(Settings.OUTPUT_DIR)) {
      outputDir = cmdLine.getOptionValue(Settings.OUTPUT_DIR);
    } else {
      // If not initialized.
      throw new ParseException("Parsing failed due to \"" + Settings.OUTPUT_DIR
        + "\" not initialized.");
    }

    if (cmdLine.hasOption(Settings.NUM_MAPPERS)) {
      numMappers = Integer.parseInt(cmdLine.getOptionValue(Settings.NUM_MAPPERS));
      Preconditions.checkArgument(numMappers > 0, "Illegal settings for \"" +
        Settings.NUM_MAPPERS + "\". Option must be positive.");
    }

    if (cmdLine.hasOption(Settings.NUM_REDUCERS)) {
      numReducers = Integer.parseInt(cmdLine.getOptionValue(Settings.NUM_REDUCERS));
      Preconditions.checkArgument(numReducers > 0, "Illegal settings for \"" +
        Settings.NUM_REDUCERS + "\". Option must be positive.");
    }

    if (cmdLine.hasOption(Settings.MAPPER_MEMORY_MB)) {
      mapperSize = Integer.parseInt(cmdLine.getOptionValue(Settings.MAPPER_MEMORY_MB));
      Preconditions.checkArgument(mapperSize > 0,
        "Illegal settings for \"" + Settings.MAPPER_MEMORY_MB +
          "\". Option must be positive. Got " + mapperSize + ".");
    }

    if (cmdLine.hasOption(Settings.REDUCER_MEMORY_MB)) {
      reducerSize = Integer.parseInt(cmdLine.getOptionValue(Settings.REDUCER_MEMORY_MB));
      Preconditions.checkArgument(reducerSize > 0,
        "Illegal settings for \"" + Settings.REDUCER_MEMORY_MB +
          "\". Option must be positive. Got " + reducerSize + ".");
    }

    if (cmdLine.hasOption(Settings.MAX_VOCAB_SIZE)) {
      maxVocabSize = Integer.parseInt(cmdLine.getOptionValue(Settings.MAX_VOCAB_SIZE));
      Preconditions.checkArgument(maxVocabSize > 0, "Illegal settings for \"" +
        Settings.MAX_VOCAB_SIZE + "\". Option must be positive.");
    }

    if (cmdLine.hasOption(Settings.MIN_COUNT)) {
      minCount = Integer.parseInt(cmdLine.getOptionValue(Settings.MIN_COUNT));
      Preconditions.checkArgument(minCount > 0, "Illegal settings for \"" +
        Settings.MIN_COUNT + "\". Option must be positive.");
    }

    if (cmdLine.hasOption(Settings.ITERATIONS)) {
      iterations = Integer.parseInt(cmdLine.getOptionValue(Settings.ITERATIONS));
      Preconditions.checkArgument(iterations > 0, "Illegal settings for \"" +
        Settings.ITERATIONS + "\". Option must be positive.");
    }

    if (cmdLine.hasOption(Settings.EPOCHS)) {
      epochs = Integer.parseInt(cmdLine.getOptionValue(Settings.EPOCHS));
      Preconditions.checkArgument(epochs > 0, "Illegal settings for \"" +
        Settings.EPOCHS + "\". Option must be positive.");
    }

    if (cmdLine.hasOption(Settings.VECTOR_SIZE)) {
      vectorSize = Integer.parseInt(cmdLine.getOptionValue(Settings.VECTOR_SIZE));
      Preconditions.checkArgument(vectorSize > 0, "Illegal settings for \"" +
        Settings.VECTOR_SIZE + "\". Option must be positive.");
    }
  }
}