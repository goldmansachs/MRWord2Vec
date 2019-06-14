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

import com.gs.mrword2vec.mapreduce.utils.Settings;
import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.io.LongWritable;
import org.apache.hadoop.io.Text;
import org.apache.hadoop.mapreduce.Reducer;

import java.io.IOException;

/**
 * Reducer class for word count job.
 */
public class WordCountReducer extends Reducer<Text, LongWritable, Text, LongWritable> {

  public enum WORDCOUNT_REDUCER_COUNTERS {
    WORDS_WRITTEN
  }

  private LongWritable result = new LongWritable();
  private int minCount;

  @Override
  public void setup(Context context) {
    Configuration conf = context.getConfiguration();
    // Get the minimum frequency needed for a word to be part of vocabulary.
    // This parameter is passed from the driver class.
    // Default value is 1.
    minCount = conf.getInt(Settings.MIN_COUNT, 1);
  }

  @Override
  public void reduce(Text key, Iterable<LongWritable> values, Context context)
    throws IOException, InterruptedException {
    long sum = 0L;
    for (LongWritable value: values) {
      sum += value.get();
    }
    if (sum >= minCount) {
      result.set(sum);
      context.write(key, result);
      context.getCounter(WORDCOUNT_REDUCER_COUNTERS.WORDS_WRITTEN).increment(1);
    }
  }
}