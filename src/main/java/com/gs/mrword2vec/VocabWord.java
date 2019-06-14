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

/**
 * This class encapsulates an entry in vocabulary.
 * Other than storing information about the word, like its String representation
 * and its count in the corpus, the object also stores information about its
 * location in the Huffman tree.
 */
public class VocabWord implements Comparable<VocabWord> {

  // The word itself.
  public String word;
  // Number of occurrences of the word in the corpus.
  public Long count;
  // Variable point stores the index of non-leaf nodes which occur on the path
  // to this word.
  // Number of relevant entries is (codeLen + 1).
  // point[0] is root of the Huffman tree. point[i+1] is child of point[i].
  // Last relevant entry is the leaf, which corresponds to the vocabulary word.
  public int[] point;
  // Huffman code for the word.
  // Number of relevant entries is codeLen.
  // code[i] is 1 if we need to take right at ith step in the tree to reach the
  // leaf corresponding to the vocabulary word, 0 if we need to take left.
  public int[] code;
  // Length of the code. codeLen entries in arrays "code" and "point" are
  // meaningful. Rest of the entries don't matter.
  public int codeLen;

  /**
   * Constructor used in testing.
   */
  public VocabWord(String word) {
    this.word = word;
  }

  /**
   * Main constructor to initialize all parameters.
   * @param word String representation of the word.
   * @param count Count in the corpus.
   * @param code Huffman code.
   * @param point Pointers to nodes in the path from root to the leaf denoting
   *              this word.
   * @param codeLen Code length.
   */
  public VocabWord(String word, Long count, int[] code, int[] point, int codeLen) {
    this.word = word;
    this.count = count;
    this.code = code;
    this.point = point;
    this.codeLen = codeLen;
  }

  /**
   * Useful for debugging.
   */
  @Override
  public String toString() {
    return "(" + word + ", " + count + ")";
  }

  /**
   * A VocabWord object with greater count is defined by this compareTo function
   * to be smaller than a VocabWord object with smaller count, so that if a list
   * of VocabWord objects is sorted, it will be sorted in descending order.
   * @param a : Other VocabWord object to compare against.
   */
  @Override
  public int compareTo(VocabWord a) {
    if (count < a.count) {
      return 1;
    } else if (count > a.count) {
      return -1;
    } else {
      return 0;
    }
  }
}