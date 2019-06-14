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
import com.gs.mrword2vec.mapreduce.utils.Vector;
import org.apache.hadoop.io.Text;
import org.apache.hadoop.mapreduce.Reducer;

import java.io.IOException;

/**
 * Word2VecReducer.java is the reducer class for the MR job which trains the
 * word2vec model.
 * It is responsible for collecting all vectors (from different models trained
 * on each mapper) for a word, and take their average, and then write them to a
 * file so that they can be used by the next job if there is one.
 */
public class Word2VecReducer extends Reducer<Text, Vector, Text, Text> {

  public enum WORD2VEC_REDUCER_COUNTERS {
    KEYS
  }

  @Override
  public void reduce(Text key, Iterable<Vector> values, Context context)
    throws IOException, InterruptedException {
    context.getCounter(WORD2VEC_REDUCER_COUNTERS.KEYS).increment(1);
    // avgVec will store the average vector.
    Vector avgVec = new Vector();
    // Count of vectors received.
    int c = 0;
    for (Vector v: values) {
      avgVec.add(v);
      c += 1;
    }
    Preconditions.checkArgument(c != 0);
    // Divide by c to get average.
    avgVec.divide(c);
    // Write to context.
    StringBuilder sb = new StringBuilder();
    for (int i = 0; i < avgVec.size; i++) {
      sb.append(avgVec.values[i]).append(" ");
    }
    Text val = new Text(sb.toString().trim());
    context.write(key, val);
  }
}