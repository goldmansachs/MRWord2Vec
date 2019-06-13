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
package com.gs.mrword2vec.utils;

/**
 * An object to store a String and a float. Used when computing similar words
 * to a given word.
 */
public class Tuple implements Comparable<Tuple> {
  public String word;
  public float score;

  public Tuple(String word, float score) {
    this.word = word;
    this.score = score;
  }

  /**
   * t1 < t2 iff t1.score < t2.score.
   */
  @Override
  public int compareTo(Tuple t) {
    if (score < t.score) {
      return -1;
    } else if (score > t.score) {
      return 1;
    } else {
      return 0;
    }
  }

  @Override
  public String toString() {
    return "(" + word + ", " + score + ")";
  }
}