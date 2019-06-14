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

import com.gs.mrword2vec.utils.Tuple;
import org.jblas.FloatMatrix;
import org.junit.Test;

import java.io.*;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Set;

import static junit.framework.TestCase.assertEquals;
import static junit.framework.TestCase.assertTrue;
import static junit.framework.TestCase.assertFalse;

/**
 * Test class for Word2Vec.class.
 */
public class Word2VecTest {

  @Test
  public void testBuildVocab() {
    // Creating a dummy data set of sentences.
    // Three sentences with 4 distinct words - "a", "b", "c" and 'd".
    // "a" occurs 6 times, "b" occurs 2 times, "c" occurs 4 times, and "d"
    // occurs 1 time.
    List<String[]> sentences = new ArrayList<>();
    sentences.add(new String[]{"a", "b", "c", "a"});
    sentences.add(new String[]{"b", "c", "c", "c"});
    sentences.add(new String[]{"a", "a", "a", "a", "d"});
    // Initializing a Word2Vec object.
    Word2Vec word2Vec = new Word2Vec();
    word2Vec.maxVocabSize = 5;
    // This should remove "d" from vocabulary.
    word2Vec.minCount = 2;
    // Run buildVocab function.
    word2Vec.buildVocab(sentences);
    // vocabSize should be 3, since "d"'s frequency is 1.
    assertTrue(word2Vec.vocabSize == 3);
    // "d" shouldn't be part of vocabulary.
    assertFalse(word2Vec.vocabHash.containsKey("d"));
    // buildVocab also sorts the vocabulary by its count in nonincreasing order.
    // So the most frequent word will have the index 0, the second most frequent
    // word will the index 1, and so on.
    // Count of "a" should be 6, and its ID should be 0.
    assertTrue(word2Vec.vocabHash.get("a") == 0);
    assertTrue(word2Vec.vocab.get(0).word.equals("a"));
    assertTrue(word2Vec.vocab.get(0).count == 6L);
    // Count of "c" should be 4, and its ID should be 1.
    assertTrue(word2Vec.vocabHash.get("c") == 1);
    assertTrue(word2Vec.vocab.get(1).word.equals("c"));
    assertTrue(word2Vec.vocab.get(1).count == 4L);
    // Count of "b" should be 2, and its ID should be 2.
    assertTrue(word2Vec.vocabHash.get("b") == 2);
    assertTrue(word2Vec.vocab.get(2).word.equals("b"));
    assertTrue(word2Vec.vocab.get(2).count == 2L);
    // Variable trainWordsCount should equal sum of counts of all vocabulary
    // words.
    assertTrue(word2Vec.trainWordsCount == 6 + 4 + 2);
  }

  @Test
  public void testCreateHuffmanTree() {
    // Creating a dummy data set of sentences.
    // Three sentences with 4 distinct words - "a", "b", "c" and 'd".
    // "a" occurs 6 times, "b" occurs 2 times, "c" occurs 4 times, and "d"
    // occurs 1 time.
    List<String[]> sentences = new ArrayList<>();
    sentences.add(new String[]{"a", "b", "c", "a"});
    sentences.add(new String[]{"b", "c", "c", "c"});
    sentences.add(new String[]{"a", "a", "a", "a", "d"});
    // Initializing a Word2Vec object.
    Word2Vec word2Vec = new Word2Vec();
    // We want all words in the vocabulary so we make maxVocabSize at least as
    // large as the number (four) of distinct words, and minCount 1.
    word2Vec.maxVocabSize = 5;
    word2Vec.minCount = 1;
    // Run buildVocab function.
    word2Vec.buildVocab(sentences);
    // Create the Huffman tree.
    word2Vec.createHuffmanTree();
    // All relevant information pertaining to Huffman tree is stored in vocab list.
    // The tree should have 4 leaves, one for each vocabulary word.
    // The tree will look like:
    //
    //          root
    //           /\
    //          /  \
    //         a   inner node 2
    //                 /\
    //                /  \
    //     inner node 1   c
    //           /\
    //          /  \
    //         d    b
    //
    // Here, left child is given the code 0, and right child is given the code 1.
    // So code of "a" should be 0, code of "c" should be 11, code of "d" should
    // be 100, and code of "b" should be 101.
    // Every node in the tree has a count associated with it. For leaves (which
    // are vocabulary words), the count is their frequency. For inner nodes,
    // it's the sum of the counts of its children.
    // Every node in the tree has an associated ID. IDs for vocabulary words
    // (leaves) will range from 0 to (vocabSize - 1), with more frequent word
    // having lower ID. IDs for inner nodes varies from 0 to (vocabSize - 2),
    // with an inner node having lower count having higher ID. The IDs for inner
    // nodes reflect the order in which they were created. So, for example,
    // in the tree above, inner node 1 is created before inner node 2 which is
    // created before the root. inner node 1 will have ID 0, inner node 2 will
    // have ID 1, and root will have ID 2.
    //
    // Checking for "a".
    // "a"'s ID is 0 since it's the most frequent word.
    assertTrue(word2Vec.vocabHash.get("a") == 0);
    assertTrue(word2Vec.vocab.get(0).word.equals("a"));
    // "a" should have Huffman code "0". So codeLen will be 1.
    assertTrue(word2Vec.vocab.get(0).codeLen == 1);
    assertTrue(word2Vec.vocab.get(0).code[0] == 0);
    // "a"'s parent should be the root of the tree. Root's ID should be 2 since
    // there are 3 non-leaf nodes and it's constructed last.
    assertTrue(word2Vec.vocab.get(0).point[0] == 2);
    // This value should be (ID of "a" - vocabSize) = (0 - 4), which is -4.
    assertTrue(word2Vec.vocab.get(0).point[1] == -4);

    // Checking for "c".
    // "c"'s ID is 1 since it's the second most frequent word.
    assertTrue(word2Vec.vocabHash.get("c") == 1);
    assertTrue(word2Vec.vocab.get(1).word.equals("c"));
    // "c" should have Huffman code "11". So codeLen will be 2.
    assertTrue(word2Vec.vocab.get(1).codeLen == 2);
    assertTrue(word2Vec.vocab.get(1).code[0] == 1);
    assertTrue(word2Vec.vocab.get(1).code[1] == 1);
    // Root's ID should be 2 since there are 3 non-leaf nodes and it's
    // constructed last.
    assertTrue(word2Vec.vocab.get(1).point[0] == 2);
    // "c"'s parent should be inner node 2. Its ID should be 1, since it's
    // constructed after inner node 1.
    assertTrue(word2Vec.vocab.get(1).point[1] == 1);
    // This value should be (ID of "c" - vocabSize) = (1 - 4), which is -3.
    assertTrue(word2Vec.vocab.get(1).point[2] == -3);

    // Checking for "b".
    // "b"'s ID is 2 since it's the third most frequent word.
    assertTrue(word2Vec.vocabHash.get("b") == 2);
    assertTrue(word2Vec.vocab.get(2).word.equals("b"));
    // "b" should have Huffman code "101". So codeLen will be 3.
    assertTrue(word2Vec.vocab.get(2).codeLen == 3);
    assertTrue(word2Vec.vocab.get(2).code[0] == 1);
    assertTrue(word2Vec.vocab.get(2).code[1] == 0);
    assertTrue(word2Vec.vocab.get(2).code[2] == 1);
    // Root's ID should be 2 since there are 3 non-leaf nodes and it's
    // constructed last.
    assertTrue(word2Vec.vocab.get(2).point[0] == 2);
    // Root's child on the way to "b" is inner node 2.
    assertTrue(word2Vec.vocab.get(2).point[1] == 1);
    // "b"'s parent should be inner node 1. Its ID should be 0, since it's
    // constructed first among inner nodes.
    assertTrue(word2Vec.vocab.get(2).point[2] == 0);
    // This value should be (ID of "b" - vocabSize) = (2 - 4), which is -2.
    assertTrue(word2Vec.vocab.get(2).point[3] == -2);

    // Checking for "d".
    // "d"'s ID is 3 since it's the fourth most frequent word.
    assertTrue(word2Vec.vocabHash.get("d") == 3);
    assertTrue(word2Vec.vocab.get(3).word.equals("d"));
    // "d" should have Huffman code "100". So codeLen will be 3.
    assertTrue(word2Vec.vocab.get(3).codeLen == 3);
    assertTrue(word2Vec.vocab.get(3).code[0] == 1);
    assertTrue(word2Vec.vocab.get(3).code[1] == 0);
    assertTrue(word2Vec.vocab.get(3).code[2] == 0);
    // Root's ID should be 2 since there are 3 non-leaf nodes and it's
    // constructed last.
    assertTrue(word2Vec.vocab.get(3).point[0] == 2);
    // Root's child on the way to "d" is inner node 2.
    assertTrue(word2Vec.vocab.get(3).point[1] == 1);
    // "d"'s parent should be inner node 1. Its ID should be 0, since it's
    // constructed first among inner nodes.
    assertTrue(word2Vec.vocab.get(3).point[2] == 0);
    // This value should be (ID of "d" - vocabSize) = (3 - 4), which is -1.
    assertTrue(word2Vec.vocab.get(3).point[3] == -1);
  }

  @Test
  public void testTrainSentence() {
    // ### Forcing the learning rate to be 0. ###
    // Weight matrices syn0 and syn1 should remain unchanged.
    Word2Vec word2Vec = new Word2Vec();
    word2Vec.learningRateStart = 0;
    word2Vec.learningRateEnd = 0;
    word2Vec.alpha = 0;
    word2Vec.vectorSize = 5;
    // Creating a dummy data set of sentences.
    List<String[]> sentences = new ArrayList<>();
    sentences.add(new String[]{"a", "b", "c", "a"});
    sentences.add(new String[]{"b", "c", "c", "c"});
    sentences.add(new String[]{"a", "a", "a", "a", "d"});
    // We want all words in the vocabulary.
    word2Vec.maxVocabSize = 5;
    word2Vec.minCount = 1;
    word2Vec.buildVocab(sentences);
    word2Vec.createHuffmanTree();
    // Initialize the matrices syn0 and syn1.
    word2Vec.initializeMatrices();
    // Create duplicates of syn0 and syn1 to compare against.
    FloatMatrix syn0Copy = word2Vec.syn0.dup();
    FloatMatrix syn1Copy = word2Vec.syn1.dup();
    // Training on a dummy sentence.
    // Note: It may contain words outside vocabulary.
    String[] sentence = {"a", "b", "e", "c"};
    word2Vec.trainSentence(sentence);
    // Compare syn0Copy to syn0. They should be same.
    assertEquals(syn0Copy.length, word2Vec.syn0.length);
    for (int i = 0; i < syn0Copy.data.length; i++) {
      assertEquals(syn0Copy.data[i], word2Vec.syn0.data[i], 1e-6);
    }
    // Compare syn1Copy to syn1. They should be same.
    assertEquals(syn1Copy.length, word2Vec.syn1.length);
    for (int i = 0; i < syn1Copy.data.length; i++) {
      assertEquals(syn1Copy.data[i], word2Vec.syn1.data[i], 1e-6);
    }

    // ### Now, test using learning rate = 1. ###
    word2Vec.learningRateStart = 1;
    word2Vec.learningRateEnd = 1;
    word2Vec.alpha = 1;
    // Initializing matrices syn0 and syn1 with known values.
    for (int i = 0; i < word2Vec.syn0.length; i++) {
      word2Vec.syn0.data[i] = 0.05f;
      word2Vec.syn1.data[i] = 0.05f;
    }
    // Create duplicates of syn0 and syn1 to compare against.
    syn0Copy = word2Vec.syn0.dup();
    syn1Copy = word2Vec.syn1.dup();
    String[] smallSentence = {"a", "c", "e"};
    word2Vec.trainSentence(smallSentence);
    // Vectors for "b" and "d" in syn0 should remain unchanged, because they
    // aren't present in the training sentence.
    // They correspond to indices 10 to 19 in syn0.
    for (int i = 10; i < 20; i++) {
      assertEquals(syn0Copy.data[i], word2Vec.syn0.data[i], 1e-6);
    }
    // In syn1, the first 5 values should remain unchanged since they correspond
    // to vector for inner node 1 which is never visited, and hence never changed.
    for (int i = 0; i < 5; i++) {
      assertEquals(syn1Copy.data[i], word2Vec.syn1.data[i], 1e-6);
    }
    // The last 5 values should also remain unchanged since they are never used.
    // These are redundant values added in syn1 to make its size same as syn0.
    for (int i = 15; i < 20; i++) {
      assertEquals(syn1Copy.data[i], word2Vec.syn1.data[i], 1e-6);
    }

    // First, using "c" we want to predict "a", i.e., using ("c", "a")
    // input-output pair.
    // This will update vector of "c" in syn0, and vector for root in syn1.
    // That is, entries 5,6,7,8,9 in syn0 and entries 10,11,12,13,14 in syn1.
    // At root, we need to take left to reach "a".
    // We take left with probability sigmoid(dot product of vectors
    // v1 = syn0[5,6,7,8,9] and v2 = syn1[10,11,12,13,14]) which equals
    // f = 0.503125.
    // Gradient to update v2 = (f - 1) * v1 =
    // [-0.024843752, -0.024843752, -0.024843752, -0.024843752, -0.024843752]
    // (f - 1) above because we take a left at the root. Had we taken a right,
    // it would have been just f.
    // Learning rate = 1, so v2 should be v2 := v2 - 1 * gradient; which equals:
    // [0.074843752, 0.074843752, 0.074843752, 0.074843752, 0.074843752]
    // Similarly, v1 changes to the same vector as above.
    //
    // Now, using "a" to predict "c", i.e., using ("a", "c") input-output pair.
    // This will first update syn1[10,11,12,13,14], i.e., the vector for root.
    // To reach "c", we need to take right at root, and then right at inner node 2.
    // Probability of taking the left at root is sigmoid(dot product of vectors
    // v1 = syn0[5,6,7,8,9] and v2 = syn1[10,11,12,13,14]) which equals
    // f = 0.504678.
    // Gradient to update v2 = f * v1 =
    // [0.025234, 0.025234, 0.025234, 0.025234, 0.025234]
    // v2 := v2 - 1 * gradient. So v2 becomes:
    // [0.04961, 0.04961, 0.04961, 0.04961, 0.04961]
    // Similarly, syn1[5,6,7,8,9] gets updated to
    // [0.024844, 0.024844, 0.024844, 0.024844, 0.024844].
    // Vector for "a" in syn0, syn0[0,1,2,3,4] gets updated to:
    // [-0.0128, -0.0128, -0.0128, -0.0128, -0.0128].

    // Checking now.
    // Our tolerance is pretty high at 1e-4 because of approximate sigmoid
    // values used from the table.
    // Starting with syn0.
    for (int i = 0; i < 5; i++) {
      assertEquals(-0.0128, word2Vec.syn0.data[i], 1e-4);
    }
    for (int i = 5; i < 10; i++) {
      assertEquals(0.074843752, word2Vec.syn0.data[i], 1e-4);
    }
    // Now for syn1.
    for (int i = 5; i < 10; i++) {
      assertEquals(0.024844, word2Vec.syn1.data[i], 1e-4);
    }
    for (int i = 10; i < 15; i++) {
      assertEquals(0.04961, word2Vec.syn1.data[i], 1e-4);
    }
  }

  @Test
  public void testPopulateNorms() throws Exception {
    Word2Vec word2Vec = new Word2Vec();
    word2Vec.vectorSize = 3;
    word2Vec.vocabSize = 2;
    // Initialize syn0.
    word2Vec.syn0 = new FloatMatrix(2 * 3);
    // First vector will be [0.5, 1.5, -1.0]. Its norm should be 1.8708286934.
    word2Vec.syn0.data[0] = 0.5f;
    word2Vec.syn0.data[1] = 1.5f;
    word2Vec.syn0.data[2] = -1.0f;
    // Second vector will be [2.3, -4.5, -1.7]. Its norm should be 5.3319789947.
    word2Vec.syn0.data[3] = 2.3f;
    word2Vec.syn0.data[4] = -4.5f;
    word2Vec.syn0.data[5] = -1.7f;

    word2Vec.populateNorms();
    // Vocabulary size = 2. Therefore, there should be 2 norm values.
    assertTrue(word2Vec.syn0Norms.length == 2);
    // Tolerance of 1e-6 because float is being used and they have 24 bits of
    // precision, which is equivalent to 7 decimal digits. The test fails at
    // 1e-7.
    assertEquals(1.8708286934, word2Vec.syn0Norms.data[0], 1e-6);
    assertEquals(5.3319789947, word2Vec.syn0Norms.data[1], 1e-6);
  }

  @Test
  public void testMostSimilar() throws Exception {
    // Initialize Word2Vec object with known values.
    Word2Vec word2Vec = new Word2Vec();
    word2Vec.vectorSize = 3;
    // Initializing with 5 words.
    word2Vec.vocabSize = 5;
    word2Vec.vocabHash = new HashMap<>();
    word2Vec.vocabHash.put("a", 0);
    word2Vec.vocabHash.put("b", 1);
    word2Vec.vocabHash.put("c", 2);
    word2Vec.vocabHash.put("d", 3);
    word2Vec.vocabHash.put("e", 4);
    word2Vec.vocab = new ArrayList<>();
    word2Vec.vocab.add(new VocabWord("a"));
    word2Vec.vocab.add(new VocabWord("b"));
    word2Vec.vocab.add(new VocabWord("c"));
    word2Vec.vocab.add(new VocabWord("d"));
    word2Vec.vocab.add(new VocabWord("e"));
    // Initialize syn0.
    word2Vec.syn0 = new FloatMatrix(3 * 5);
    // Vector for "a" will be [0.5, 1.5, 2.5].
    word2Vec.syn0.data[0] = 0.5f;
    word2Vec.syn0.data[1] = 1.5f;
    word2Vec.syn0.data[2] = 2.5f;
    // Vector for "b" will be [0.5, -1.5, 0.5].
    word2Vec.syn0.data[3] = 0.5f;
    word2Vec.syn0.data[4] = -1.5f;
    word2Vec.syn0.data[5] = 0.5f;
    // Vector for "c" will be [5.0, -10.0, 5.0].
    word2Vec.syn0.data[6] = 5.0f;
    word2Vec.syn0.data[7] = -10.0f;
    word2Vec.syn0.data[8] = 5.0f;
    // Vector for "d" will be [1.0, 1.2, 1.0].
    word2Vec.syn0.data[9] = 1.0f;
    word2Vec.syn0.data[10] = 1.2f;
    word2Vec.syn0.data[11] = 1.0f;
    // Vector for "e" will be [-2.3, -1.7, -3.4].
    word2Vec.syn0.data[12] = -2.3f;
    word2Vec.syn0.data[13] = -1.7f;
    word2Vec.syn0.data[14] = -3.4f;

    // Cosine similarity between two vectors is symmetric.
    // The cosine similarity for {a, b} = -0.1528941574
    // The cosine similarity for {a, c} = 0.0
    // The cosine similarity for {a, d} = 0.87489913989
    // The cosine similarity for {a, e} = -0.9282869177
    // The cosine similarity for {b, c} = 0.98473192783
    // The cosine similarity for {b, d} = -0.2601024355
    // The cosine similarity for {b, e} = -0.0407175214
    // The cosine similarity for {c, d} = -0.0880450906
    // The cosine similarity for {c, e} = -0.2113385078
    // The cosine similarity for {d, e} = -0.9392650558

    // floats are precise up to 7 decimal digits. Because of multiple operations
    // needed to compute cosine similarity, the precision gets further reduced.
    // The test cases below fail if eps = 1e-7 is used.
    double eps = 1e-6;

    // Checking for "a".
    List<Tuple> aList = word2Vec.mostSimilar("a", 5);
    // Since we requested top 5 above.
    assertTrue(aList.size() == 5);
    // The first element should be the word itself with score 1.0.
    assertTrue(aList.get(0).word.equals("a"));
    assertEquals(1.0, aList.get(0).score, eps);
    // The second element should be "d" with score 0.87489913989.
    assertTrue(aList.get(1).word.equals("d"));
    assertEquals(0.87489913989, aList.get(1).score, eps);
    // The third element should be "c" with score 0.
    assertTrue(aList.get(2).word.equals("c"));
    assertEquals(0.0, aList.get(2).score, eps);
    // The fourth element should be "b" with score -0.1528941574.
    assertTrue(aList.get(3).word.equals("b"));
    assertEquals(-0.1528941574, aList.get(3).score, eps);
    // The fifth element should be "e" with score -0.9282869177.
    assertTrue(aList.get(4).word.equals("e"));
    assertEquals(-0.9282869177, aList.get(4).score, eps);

    // Checking for "b".
    List<Tuple> bList = word2Vec.mostSimilar("b", 5);
    // Since we requested top 5 above.
    assertTrue(bList.size() == 5);
    // The first element should be the word itself with score 1.0.
    assertTrue(bList.get(0).word.equals("b"));
    assertEquals(1.0, bList.get(0).score, eps);
    // The second element should be "c" with score 0.98473192783.
    assertTrue(bList.get(1).word.equals("c"));
    assertEquals(0.98473192783, bList.get(1).score, eps);
    // The third element should be "e" with score -0.0407175214.
    assertTrue(bList.get(2).word.equals("e"));
    assertEquals(-0.0407175214, bList.get(2).score, eps);
    // The fourth element should be "a" with score -0.1528941574.
    assertTrue(bList.get(3).word.equals("a"));
    assertEquals(-0.1528941574, bList.get(3).score, eps);
    // The fifth element should be "d" with score -0.2601024355.
    assertTrue(bList.get(4).word.equals("d"));
    assertEquals(-0.2601024355, bList.get(4).score, eps);

    // Checking for "c".
    List<Tuple> cList = word2Vec.mostSimilar("c", 5);
    // Since we requested top 5 above.
    assertTrue(cList.size() == 5);
    // The first element should be the word itself with score 1.0.
    assertTrue(cList.get(0).word.equals("c"));
    assertEquals(1.0, cList.get(0).score, eps);
    // The second element should be "b" with score 0.98473192783.
    assertTrue(cList.get(1).word.equals("b"));
    assertEquals(0.98473192783, cList.get(1).score, eps);
    // The third element should be "a" with score 0.
    assertTrue(cList.get(2).word.equals("a"));
    assertEquals(0.0, cList.get(2).score, eps);
    // The fourth element should be "d" with score -0.0880450906.
    assertTrue(cList.get(3).word.equals("d"));
    assertEquals(-0.0880450906, cList.get(3).score, eps);
    // The fifth element should be "e" with score -0.2113385078.
    assertTrue(cList.get(4).word.equals("e"));
    assertEquals(-0.2113385078, cList.get(4).score, eps);

    // Checking for "e".
    List<Tuple> dList = word2Vec.mostSimilar("d", 5);
    // Since we requested top 5 above.
    assertTrue(dList.size() == 5);
    // The first element should be the word itself with score 1.0.
    assertTrue(dList.get(0).word.equals("d"));
    assertEquals(1.0, dList.get(0).score, eps);
    // The second element should be "a" with score 0.87489913989.
    assertTrue(dList.get(1).word.equals("a"));
    assertEquals(0.87489913989, dList.get(1).score, eps);
    // The third element should be "c" with score -0.0880450906.
    assertTrue(dList.get(2).word.equals("c"));
    assertEquals(-0.0880450906, dList.get(2).score, eps);
    // The fourth element should be "b" with score -0.2601024355.
    assertTrue(dList.get(3).word.equals("b"));
    assertEquals(-0.2601024355, dList.get(3).score, eps);
    // The fifth element should be "e" with score -0.9392650558.
    assertTrue(dList.get(4).word.equals("e"));
    assertEquals(-0.9392650558, dList.get(4).score, eps);

    // Checking for "e".
    List<Tuple> eList = word2Vec.mostSimilar("e", 5);
    // Since we requested top 5 above.
    assertTrue(eList.size() == 5);
    // The first element should be the word itself with score 1.0.
    assertTrue(eList.get(0).word.equals("e"));
    assertEquals(1.0, eList.get(0).score, eps);
    // The second element should be "b" with score -0.0407175214.
    assertTrue(eList.get(1).word.equals("b"));
    assertEquals(-0.0407175214, eList.get(1).score, eps);
    // The third element should be "c" with score -0.2113385078.
    assertTrue(eList.get(2).word.equals("c"));
    assertEquals(-0.2113385078, eList.get(2).score, eps);
    // The fourth element should be "a" with score -0.9282869177.
    assertTrue(eList.get(3).word.equals("a"));
    assertEquals(-0.9282869177, eList.get(3).score, eps);
    // The fifth element should be "d" with score -0.9392650558.
    assertTrue(eList.get(4).word.equals("d"));
    assertEquals(-0.9392650558, eList.get(4).score, eps);
  }

  @Test
  public void testGetVocab() {
    // Initialize Word2Vec object with known values.
    Word2Vec word2Vec = new Word2Vec();
    // Populating vocab list with 4 words - "a", "b", "c" and "d".
    word2Vec.vocab = new ArrayList<>();
    word2Vec.vocab.add(new VocabWord("a"));
    word2Vec.vocab.add(new VocabWord("b"));
    word2Vec.vocab.add(new VocabWord("c"));
    word2Vec.vocab.add(new VocabWord("d"));
    // Getting vocabulary set.
    Set<String> vocabSet = word2Vec.getVocab();
    assertTrue(vocabSet.size() == 4);
    assertTrue(vocabSet.contains("a"));
    assertTrue(vocabSet.contains("b"));
    assertTrue(vocabSet.contains("c"));
    assertTrue(vocabSet.contains("d"));
  }

  @Test
  public void testReadAndWriteMatrices() throws IOException {
    // Initializing dummy Word2Vec object.
    Word2Vec word2Vec = new Word2Vec();
    word2Vec.vocabSize = 3;
    word2Vec.vectorSize = 5;
    int matrixLength = word2Vec.vocabSize * word2Vec.vectorSize;
    // Initializing syn0 and syn1 with uniform random values from [0, 1].
    word2Vec.syn0 = FloatMatrix.rand(matrixLength);
    word2Vec.syn1 = FloatMatrix.rand(matrixLength);
    // Create a duplicate of matrices for comparison.
    FloatMatrix syn0Copy = word2Vec.syn0.dup();
    FloatMatrix syn1Copy = word2Vec.syn1.dup();
    // Write the matrices and read them back, and ensure they are same as
    // syn0Copy and syn1Copy. Reading will overwrite the matrices of word2Vec
    // object.
    String path = "src\\test\\resources\\testMatrices";
    BufferedWriter bw = new BufferedWriter(new FileWriter(path));
    word2Vec.writeMatrices(bw);
    bw.close();
    BufferedReader br = new BufferedReader(new FileReader(path));
    word2Vec.readMatrices(br);
    br.close();
    for (int i = 0; i < matrixLength; i++) {
      assertEquals(syn0Copy.data[i], word2Vec.syn0.data[i], 1e-7);
      assertEquals(syn1Copy.data[i], word2Vec.syn1.data[i], 1e-7);
    }
    // Delete the file since it's not needed now.
    File file = new File(path);
    file.delete();
  }

  @Test
  public void testReadAndWriteHuffmanTree() throws IOException {
    // Creating a dummy data set to create a Huffman tree.
    List<String[]> sentences = new ArrayList<>();
    sentences.add(new String[]{"a", "b", "c", "a"});
    sentences.add(new String[]{"b", "c", "c", "c"});
    sentences.add(new String[]{"a", "a", "a", "a", "d"});
    // Initializing a Word2Vec object.
    Word2Vec word2Vec = new Word2Vec();
    // We want all words in the vocabulary.
    word2Vec.maxVocabSize = 5;
    word2Vec.minCount = 1;
    // Run buildVocab function.
    word2Vec.buildVocab(sentences);
    // Create the Huffman tree.
    word2Vec.createHuffmanTree();
    // Create a duplicate of Huffman tree for comparison.
    int vocabSize = word2Vec.vocabSize;
    int vectorSize = word2Vec.vectorSize;
    List<VocabWord> vocab = new ArrayList<>();
    for (int i = 0; i < vocabSize; i++) {
      int n = word2Vec.vocab.get(i).code.length;
      int[] code = new int[n];
      for (int j = 0; j < n; j++) {
        code[j] = word2Vec.vocab.get(i).code[j];
      }
      n = word2Vec.vocab.get(i).point.length;
      int[] point = new int[n];
      for (int j = 0; j < n; j++) {
        point[j] = word2Vec.vocab.get(i).point[j];
      }
      vocab.add(new VocabWord(word2Vec.vocab.get(i).word,
        word2Vec.vocab.get(i).count, code, point, word2Vec.vocab.get(i).codeLen));
    }
    // Write the Huffman tree to a dummy file.
    String path = "src\\test\\resources\\testHuffmanTree";
    BufferedWriter bw = new BufferedWriter(new FileWriter(path));
    word2Vec.writeHuffmanTree(bw);
    bw.close();
    // Read and overwrite the Huffman tree of word2Vec object.
    BufferedReader br = new BufferedReader(new FileReader(path));
    word2Vec.readHuffmanTree(br);
    br.close();
    // Now compare with copy, and ensure they are equal.
    assertEquals(vocabSize, word2Vec.vocabSize);
    assertEquals(vectorSize, word2Vec.vectorSize);
    assertEquals(vocab.size(), word2Vec.vocab.size());
    for (int i = 0; i < vocab.size(); i++) {
      assertEquals(vocab.get(i).codeLen, word2Vec.vocab.get(i).codeLen);
      int[] codeCopy = vocab.get(i).code;
      int[] codeRead = word2Vec.vocab.get(i).code;
      assertEquals(codeCopy.length, codeRead.length);
      for (int j = 0; j < codeCopy.length; j++) {
        assertEquals(codeCopy[j], codeRead[j]);
      }
      int[] pointCopy = vocab.get(i).point;
      int[] pointRead = word2Vec.vocab.get(i).point;
      assertEquals(pointCopy.length, pointRead.length);
      for (int j = 0; j < pointCopy.length; j++) {
        assertEquals(pointCopy[j], pointRead[j]);
      }
    }
    // Delete the file since it's not needed now.
    File file = new File(path);
    file.delete();
  }
}