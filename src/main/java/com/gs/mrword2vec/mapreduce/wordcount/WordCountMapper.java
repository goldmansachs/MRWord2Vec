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
package com.gs.mrword2vec.mapreduce.wordcount;

import org.apache.hadoop.io.LongWritable;
import org.apache.hadoop.io.Text;
import org.apache.hadoop.mapreduce.Mapper;

import java.io.IOException;

/**
 * Mapper class for word count job.
 */
public class WordCountMapper extends Mapper<LongWritable, Text, Text, LongWritable> {

  public enum WORDCOUNT_MAPPER_COUNTERS {
    LINES
  }

  private Text wordText = new Text();
  private final static LongWritable one = new LongWritable(1L);

  @Override
  public void map(LongWritable key, Text value, Context context)
    throws IOException, InterruptedException {
    context.getCounter(WORDCOUNT_MAPPER_COUNTERS.LINES).increment(1);
    // Each line is a String of words separated by white space.
    String[] words = value.toString().split("\\s+");
    for (String word: words) {
      wordText.set(word);
      context.write(wordText, one);
    }
  }
}