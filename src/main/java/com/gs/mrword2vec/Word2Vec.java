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

import com.google.common.base.Preconditions;
import com.gs.mrword2vec.utils.BoundedPQ;
import com.gs.mrword2vec.utils.Constants;
import com.gs.mrword2vec.utils.Tuple;
import com.gs.mrword2vec.utils.Utils;
import org.apache.log4j.Logger;
import org.jblas.FloatMatrix;
import org.jblas.JavaBlas;
import org.jblas.NativeBlas;
import org.jblas.util.Random;

import java.io.*;
import java.util.*;

/**
 * The class Word2Vec.java implements word2vec model.
 * It's a skip-gram model, and uses hierarchical softmax for optimization.
 * Training is done incrementally, thus using very less memory, and allows
 * scaling to huge text corpus.
 */
public class Word2Vec {

  private static final Logger log = Logger.getLogger(Word2Vec.class);

  // The dimension of word2vec vectors.
  public int vectorSize = 200;
  // Initial value of learning rate.
  public double learningRateStart = 0.025;
  // Minimum value for learning rate.
  public double learningRateEnd = 0.0001;
  // Minimum count of a word in vocabulary.
  public int minCount = 5;
  // Maximum vocabulary size.
  public int maxVocabSize = 100000;
  // Maximum window size.
  public int window = 5;
  // Number of times we iterate over the data set.
  public int numEpochs = 1;
  // We pre-compute values of sigmoid function on multiple values in a domain to
  // save time, since computing Math.exp(x) is slow.
  // maxVal specifies the range of domain: [-maxVal, maxVal).
  public final int maxVal = 6;
  // Number of parts to split the domain [-maxVal, maxVal) into.
  public final int sigmoidTableSize = 1000;
  // Precomputed values for sigmoid function.
  public float[] sigmoidTable = Utils.getSigmoidTable(maxVal, sigmoidTableSize);
  // Maximum code length of a vocabulary word obtained from Huffman tree.
  // For real data sets, it is really difficult for vocabulary to be so huge that
  // larger codes are required.
  public final int maxCodeLength = 40;
  // Sum of counts of all vocabulary words. Used in updating learning rate.
  public long trainWordsCount = 0L;
  // Learning rate used in gradient descent. We update "alpha" after every
  // 10,000 words by linearly decreasing it from "learningRateStart" to
  // "learningRateEnd".
  public double alpha = learningRateStart;
  // Total number of words we pass through, + 1. Used in updating "alpha".
  public long totalWordCount = numEpochs * trainWordsCount + 1;
  // Last word count update. We update this variable in blocks of 10,000 words.
  public long lastWordCount = 0;
  // Current word count.
  public long currentWordCount = 0;
  // Vocabulary size.
  public int vocabSize = 0;
  // Weight matrix between input and hidden layer.
  // It is stored as a 1-D FloatMatrix of size vocabSize * vectorSize.
  // This matrix stores the word2vec vectors we need for each vocabulary word.
  // We use FloatMatrix as opposed to DoubleMatrix to save memory.
  public FloatMatrix syn0;
  // Weight matrix between hidden and output layer.
  // It is stored as a 1-D FloatMatrix of size vocabSize * vectorSize.
  // This matrix stores the vectorSize dimensional vectors for each non-leaf
  // node in the Huffman tree used for hierarchical softmax.
  // Although there are (vocabSize - 1) non-leaf nodes, and thus a 1-D
  // FloatMatrix of size (vocabSize - 1) * vectorSize would have sufficed, we
  // still make syn1's size vocabSize * vectorSize to keep its size same as that
  // of syn0 because it helps to keep utility functions like reading and writing
  // these matrices simple.
  // We use FloatMatrix as opposed to DoubleMatrix to save memory.
  public FloatMatrix syn1;
  // List of VocabWord objects sorted by their count in descending order.
  public List<VocabWord> vocab = new ArrayList<>();
  // Mapping from vocab word String to its Integer ID.
  // ID varies from 0 to (vocabSize - 1).
  public Map<String, Integer> vocabHash = new HashMap<>();
  // An array of size vectorSize, which will contain l2 (Euclidean) norm for all
  // word2vec vectors in syn0. This is used in computing similarity between a
  // pair of vectors.
  public FloatMatrix syn0Norms = null;

  /**
   * Empty constructor. Used in driver class, Word2VecDriver.java.
   */
  public Word2Vec() {
  }

  /**
   * This constructor is used in mapper class, Word2VecMapper.java.
   */
  public Word2Vec(int vectorSize, int epochs, double learningRateStart,
    double learningRateEnd) {
    this.vectorSize = vectorSize;
    this.numEpochs = epochs;
    this.learningRateStart = learningRateStart;
    this.learningRateEnd = learningRateEnd;
  }

  /**
   * Constructor which reads a saved model from a text file.
   * @param path Path of the saved model.
   * @throws IOException If file not found or error in reading it.
   */
  public Word2Vec(String path) throws IOException {
    read(path);
  }

  /**
   * Constructor used in local training.
   */
  public Word2Vec(int vectorSize, int minCount, int maxVocabSize, int numEpochs) {
    this.vectorSize = vectorSize;
    this.minCount = minCount;
    this.maxVocabSize = maxVocabSize;
    this.numEpochs = numEpochs;
  }

  /**
   * This function learns the vocabulary, which means deciding on vocabulary
   * words and initializing VocabWord object for each.
   * This function should only be used when the dataset is small enough to be
   * stored in memory. For distributed word2vec, there is a MapReduce job that
   * does this.
   * @param sentences A list each of whose elements is a String[] representing
   *                  a sentence.
   */
  public void buildVocab(List<String[]> sentences) {
    log.info("Entered buildVocab.");
    Map<String, Long> wordCount = new HashMap<>();
    // Do a word count over the sentences.
    for (String[] sentence: sentences) {
      for (String word: sentence) {
        Utils.addOneToMap(wordCount, word);
      }
    }
    // Remove words with count lower than "minCount".
    wordCount.entrySet().removeIf(entry -> entry.getValue() < minCount);
    // Convert map to a list of VocabWord objects.
    List<VocabWord> vocabList = new ArrayList<>();
    for (String w: wordCount.keySet()) {
      vocabList.add(new VocabWord(w, wordCount.get(w), new int[maxCodeLength],
        new int[maxCodeLength], 0));
    }
    // Sort the list by count into descending order.
    // VocabWord's default implementation of compareTo function has been
    // overridden to allow this.
    Collections.sort(vocabList);
    // Copy first maxVocabSize elements of vocabList to vocab if vocabList's
    // size is greater than maxVocabSize.
    // Note that since vocabList is sorted in descending order by count, we are
    // copying the maxVocabSize most frequent elements of vocabList to vocab.
    vocab = vocabList.subList(0, Math.min(vocabList.size(), maxVocabSize));
    vocabSize = vocab.size();
    // Populate "vocabHash" and "trainWordsCount".
    for (int i = 0; i < vocab.size(); i++) {
      vocabHash.put(vocab.get(i).word, i);
      trainWordsCount += vocab.get(i).count;
    }
    totalWordCount = numEpochs * trainWordsCount + 1;
    log.info("Built vocabulary of size = " + vocabSize + ".");
  }

  public void writeHuffmanTree(String path) throws IOException {
    BufferedWriter bw = new BufferedWriter(new FileWriter(path));
    writeHuffmanTree(bw);
  }

  /**
   * This function trains a word2vec model over the given data.
   * It also builds the vocabulary if it is not already built.
   * Skip-gram and hierarchical softmax are used for training.
   * This function should only be used when the dataset is small enough to be
   * stored in memory. Storing the data in memory allows running for more than 1
   * epoch easily.
   *
   * @param sentences A list each of whose elements is a String[] representing
   *                  a sentence.
   */
  public void train(List<String[]> sentences) throws Exception {
    log.info("Entered train.");
    // Build vocabulary if not already built.
    if (vocabSize == 0) {
      buildVocab(sentences);
    }
    // Create the Huffman Tree.
    createHuffmanTree();
    initializeMatrices();
    // Train the model.
    for (int i = 0; i < numEpochs; i++) {
      log.info("Starting epoch " + (i + 1) + ".");
      for (String[] sentence: sentences) {
        trainSentence(sentence);
      }
    }
    populateNorms();
    log.info("Training complete.");
  }

  /**
   * This function creates binary Huffman tree using the word counts.
   * This way frequent words will have shorter codes.
   * This code gets stored in the "code" parameter of VocabWord.
   * This function also populates the "vocab" list.
   */
  public void createHuffmanTree() {
    log.info("Entered createHuffmanTree.");
    // ### Defining variables used in construction. ###
    // This array stores the count corresponding to a node in the tree.
    // For leaves (vocabulary words), it's their frequency. For interior nodes,
    // it's the sum of the frequencies of their two children.
    // In a tree with vocabSize leaves, there are (2 * vocabSize - 1) nodes.
    long[] count = new long[2 * vocabSize - 1];
    // This array stores 0 or 1 corresponding to whether the node is a left
    // child of its parent or right child, respectively.
    // The values are 0 by default.
    int[] binary = new int[2 * vocabSize - 1];
    // This array stores the index of the parent node.
    int[] parentNode = new int[2 * vocabSize - 1];
    // Start at the least frequent word.
    // Look at the Invariant 1 below to see how it's updated.
    int pos1 = vocabSize - 1;
    // Start at the first position for interior nodes.
    // Look at the Invariant 2 below to see how it's updated.
    int pos2 = vocabSize;
    // Index of the node with the least frequency. Could be a vocab word or an
    // interior node.
    int min1;
    // Index of the node with the second least frequency. Could be a vocab word
    // or an interior node.
    int min2;

    // ### Initializing "count" array. ###
    // The first vocabSize entries are for the leaves in the tree, and they are
    // initialized with the word count.
    for (int a = 0; a < vocabSize; a++) {
      count[a] = vocab.get(a).count;
    }
    // The next entries are for interior nodes. They are initialized with a very
    // large value.
    for (int a = vocabSize; a < 2 * vocabSize - 1; a++) {
      count[a] = Long.MAX_VALUE;
    }

    // ### Main logic for construction of Huffman tree. ###
    // Runs for (vocabSize - 1) iterations.
    // In each iteration, two nodes are merged. Merging of the last two nodes
    // gives us the root of the tree.
    // Invariant 1: pos1 points to the least frequent vocabulary word which
    // hasn't been merged with any other node yet, i.e. it doesn't have a parent.
    // Invariant 2: pos2 points either to the first uninitialized inner node, or
    // to the least frequent inner node which doesn't have a parent.
    // Initially, no node has a parent.
    for (int a = 0; a < vocabSize - 1; a++) {
      // We start with finding the two nodes with the least count.
      // Finding the least frequent node.
      // pos1 is at least 0 if we haven't exhausted merging vocab words.
      if (pos1 >= 0) {
        // The least frequent is always either at pos1 or pos2.
        if (count[pos1] < count[pos2]) {
          min1 = pos1;
          // Make pos1 point to the least frequent vocabulary word which hasn't
          // been merged yet.
          pos1--;
        } else {
          min1 = pos2;
          // Make pos2 point to the next inner node.
          pos2++;
        }
      } else { // We have merged all vocab words.
        min1 = pos2;
        pos2++;
      }
      // Finding second least frequent node.
      // Finding it is equivalent to finding the least frequent after updating
      // values of pos1 and pos2 done above. So we use the same logic.
      if (pos1 >= 0) {
        if (count[pos1] < count[pos2]) {
          min2 = pos1;
          // Make pos1 point to the least frequent vocabulary word which hasn't
          // been merged yet.
          pos1--;
        } else {
          min2 = pos2;
          // Make pos2 point to the next inner node.
          pos2++;
        }
      } else {
        min2 = pos2;
        pos2++;
      }
      // Now that we have found the two nodes with least frequencies
      // (min1 and min2), merge them.
      // Merged node is assigned the index (vocabSize + a).
      // Count of an interior node is the sum of the counts of its children.
      count[vocabSize + a] = count[min1] + count[min2];
      // Pointer to parent.
      parentNode[min1] = vocabSize + a;
      // Pointer to parent.
      parentNode[min2] = vocabSize + a;
      // Node corresponding to min2 is the right child of its parent.
      binary[min2] = 1;
      // binary[min1] = 0 by default, and corresponds to the left child.
    }

    // ### Now assign binary huffman code to each vocabulary word. ###
    // code is a temporary variable to store the path from leaf (vocab word) to
    // the root in the Huffman tree. The reverse of this path is the leaf's code
    // which is populated in the VocabWord object. code[i] = 1 if ith node in
    // this path is a right child, 0 otherwise.
    int[] code = new int[maxCodeLength];
    // point is a temporary variable to store the pointers to parent starting
    // from leaf to root. point[0] is leaf and point[i + 1] is parent of point[i].
    // Note: Each leaf corresponds to a distinct vocabulary word. So each leaf
    // will have a different code and point array.
    int[] point = new int[maxCodeLength];
    for (int a = 0; a < vocabSize; a++) {
      // i represents the current code length.
      int i = 0;
      int b = a;
      // Index (2 * vocabSize - 2) represents root of the tree.
      // Start at the leaf (vocab word), and stop at root.
      while (b != (2 * vocabSize - 2)) {
        code[i] = binary[b];
        point[i] = b;
        i++;
        b = parentNode[b];
      }
      // Final value of i represents code length.
      vocab.get(a).codeLen = i;
      // Root of the binary tree. There are (vocabSize - 1) non-leaf nodes.
      // If we number them from 0 to (vocabSize - 2), then (vocabSize - 2)
      // represents the root.
      vocab.get(a).point[0] = vocabSize - 2;
      // Reversing to get correct encoding.
      for (b = 0; b < i; b++) {
        vocab.get(a).code[i - b - 1] = code[b];
        // Subtract vocabSize to get index of non-leaf nodes.
        // Non-leaf nodes were stored after vocabulary words (which were
        // vocabSize in number).
        vocab.get(a).point[i - b] = point[b] - vocabSize;
      }
    }
    log.info("Huffman tree built.");
  }

  /**
   * This function initializes the two matrices syn0 and syn1.
   * This initialization must happen before training, but after word count.
   */
  public void initializeMatrices() {
    // Matrix entries initialized to random values between -0.5/vectorSize and
    // 0.5/vectorSize.
    syn0 = FloatMatrix.rand(vocabSize * vectorSize)
      .sub(FloatMatrix.zeros(vocabSize * vectorSize).add(0.5f))
      .div(vectorSize);
    // Matrix entries initialized to 0.0f.
    syn1 = FloatMatrix.zeros(vocabSize * vectorSize);
  }

  /**
   * This function trains the word2vec on a given sentence.
   * The model used is skip-gram, and optimization used is hierarchical softmax.
   * The function assumes that the various parameters, like Huffman tree,
   * weight parameters, etc. have been properly initialized. It will be too
   * costly to check this on every run, since the model could be trained on
   * billions of sentences.
   * The training is incremental because it happens one sentence at a time.
   * This has two advantages:
   * 1. Less memory usage because whole of the corpus need not be cached in memory.
   * 2. The model can be saved, and retrained later.
   *
   * To understand this function, it will be helpful to reference the paper:
   * "word2vec Parameter Learning Explained" by Xin Rong.
   * Paper can be found here: <a href="https://arxiv.org/pdf/1411.2738.pdf">
   *   arxiv link</a>.
   * The code below contains references to this paper.
   *
   * @param sentence Input sentence on which to train the model. The sentence is
   *                 represented as an array of words. The words need not be only
   *                 vocabulary words, but they must be case sensitive. For
   *                 example, "baseball" and "Baseball" are different words.
   */
  public void trainSentence(String[] sentence) {
    // ### Get a new array of words, consisting of only vocabulary words. ###
    // Get a count of words in the sentence which are in vocabulary.
    int wc = 0;
    for (String word: sentence) {
      if (vocabHash.containsKey(word)) {
        wc++;
      }
    }
    // Construct a new sentence.
    String[] newSentence = new String[wc];
    wc = 0;
    for (String word: sentence) {
      if (vocabHash.containsKey(word)) {
        newSentence[wc] = word;
        wc++;
      }
    }
    // Update the learning rate alpha after a block of 10,000 words.
    if (currentWordCount - lastWordCount > 10000) {
      alpha = Math.max(learningRateEnd,
        learningRateStart * (1 - (1.0 * currentWordCount) / totalWordCount));
      lastWordCount = currentWordCount;
    }
    // Increase current word count by sentence length.
    currentWordCount += newSentence.length;
    // Iterate over each word in the new sentence.
    for (int pos = 0; pos < newSentence.length; pos++) {
      // Get the word's ID.
      int word = vocabHash.get(newSentence[pos]);
      // Sample a window size uniformly from 1, ..., window.
      int b = 1 + Random.nextInt(window);
      // ### Train skip-gram. ###
      // Iterate over the context.
      for (int c = pos - b; c <= pos + b ; c++) {
        // Skip the center word.
        if (c == pos) continue;
        // Verify c isn't outside the sentence.
        if (c < 0 || c >= newSentence.length) continue;
        // Get the ID of the context word.
        int contextWord = vocabHash.get(newSentence[c]);
        // Calculate the index of the start of the weights for contextWord
        // in the matrix syn0.
        int l1 = contextWord * vectorSize;
        // neu1e is a vectorSize-dimensional vector, storing partial derivatives
        // of cost function wrt the entries of the output of the hidden layer,
        // multiplied by the negative of the learning rate.
        // See equation (53) in Rong. We want the final neu1e to be equal to
        // equation (53) times the negative of the learning rate. We do this
        // by updating neu1e d times, where d is the depth of the leaf
        // representing the center word. Each time we add a summand to neu1e.
        // This derivative is a vector, each component of which is the partial
        // derivative of cost function wrt the entries of the output of the
        // hidden layer multiplied by the negative of the learning rate.
        // This array stores the sum of
        // alpha * (-s or (1-s)) * inner_node_vector
        // (see s defined below; in equation (53) s is the sigmoid term)
        // over all inner nodes encountered in the path from root to leaf.
        // This is same as the sum in equation (53) in Rong multiplied by
        // the negative of the learning rate.
        // Since output of hidden layer is just some vector from syn0 matrix (in
        // word2vec, activation function at hidden layer is identity and there
        // is no bias, and in skip-gram model that we use here input vector is
        // 1-hot), to update syn0 by gradient descent, we just need to add
        // the vector neu1e to the corresponding vector in syn0.
        // See equations (52)-(54) in Rong. These equations compute partial
        // derivative of cost function wrt output of the hidden layer.
        // Each summand in equation (53) is computed in the for loop below, and
        // added to neu1e.
        FloatMatrix neu1e = FloatMatrix.zeros(vectorSize);
        // ### Hierarchical softmax. ###
        // Start at the root of the Huffman tree. Update the vectors for
        // non-leaf nodes encountered in the path from root to
        // desired output word's leaf. In skip-gram, the desired output word is
        // the center word. And, then finally update the vector for the input
        // word. In skip-gram, the input word is a context word chosen from the
        // window.
        for (int j = 0; j < vocab.get(word).codeLen; j++) {
          // ID corresponding to a non-leaf node.
          // This is j'th node, starting from the root, on the path from the
          // root to the leaf node corresponding to word "word".
          int inner = vocab.get(word).point[j];
          // Calculate the index of the start of the weights for "inner" in
          // the matrix syn1.
          int l2 = inner * vectorSize;
          // Propagate hidden -> output.
          // "f" is the dot product of hidden layer output (which is just
          // the vector for context word coming from the first matrix) and
          // vector of the non-leaf node in consideration.
          // Sigmoid of this dot product equals the probability of taking a left
          // at this inner node.
          // See equation (39) in Rong. f is the argument to sigmoid in equation
          // (39).
          // rdot function is used to compute the dot product of
          // syn0[l1, l1 + 1, l1 + 2, ..., l1 + vectorSize - 1] and
          // syn1[l2, l2 + 1, l2 + 2, ..., l2 + vectorSize - 1].
          float f = JavaBlas.rdot(vectorSize, syn0.data, l1, 1, syn1.data, l2, 1);
          // If f lies outside this range, then ignore it because the probability
          // of that happening is very small. There are 2 possibilities here:
          // 1. The model is very confident and making the correct decision
          // (left or right) at an inner node. In this case, the gradient will
          // be too small to have any meaningful effect on the parameters.
          // 2. The model is very confident and making the wrong decision (left
          // or right). In this case, we have a problem. But since this happens
          // with a very low probability we can ignore it.
          if (-maxVal < f && f < maxVal) {
            // Index of entry in expTable which corresponds to sigmoid of f.
            int ind = (int) ((f + maxVal) * (sigmoidTableSize / (maxVal * 2.0)));
            // Ensuring that ind lies within the range of allowed indices.
            ind = Math.min(sigmoidTableSize - 1, Math.max(0, ind));
            // s = sigmoid(dot product of the two vectors computed above).
            // It's the sigmoid term in equations (51) and (53).
            float s = sigmoidTable[ind];
            // Gradient descent to update non-leaf node's vector:
            // new_vector = old_vector + alpha * negative_of_gradient
            // See equation (51) in Rong.
            // negative_of_gradient = -1 * [(1-s) or -s] * hidden_layer_output
            // hidden_layer_output = vector for center word, because in word2vec
            // activation function at hidden layer is identity and there is no bias.
            // Second term is (1-s) if left taken at interior node, s if right.
            // g = alpha * [(1-s) or -s].
            // (g * vector for inner node) is the summand in equation (53) in
            // Rong, that needs to be summed over all non-leaf nodes to get
            // gradient of vector in syn0 matrix.
            // vocab.get(word).code[j] is 1 if right is taken at j, 0 otherwise.
            float g = (float) (alpha * (1 - vocab.get(word).code[j] - s));
            // Propagate errors hidden <- output.
            // Update neu1e to store the sum of these values.
            // Final value of neu1e will be used to update vector for center
            // word in syn0 matrix by gradient descent.
            // neu1e += g * syn1[l2, l2 + 1, l2 + 2, ..., l2 + vectorSize - 1]
            // raxpy function is used to do this. The next line does exactly this.
            // See equation (53) in Rong.
            // The parenthesized expression in (53) is s - 1 if t_j is 1, and s
            // otherwise, so its negative is either 1 - s  or -s.
            // Final value of neu1e is LHS of equation (53) times the negative
            // of the learning rate.
            JavaBlas.raxpy(vectorSize, g, syn1.data, l2, 1, neu1e.data, 0, 1);
            // Update weights of syn1.
            // syn1[l2, l2 + 1, l2 + 2, ..., l2 + vectorSize - 1] +=
            // g * syn0[l1, l1 + 1, l1 + 2, ..., l1 + vectorSize - 1]
            // raxpy function is used to do this. The next line does exactly this.
            // This step does gradient descent.
            // See equation (51) in Rong. In Rong:
            // v'_j = syn1[l2, l2 + 1, l2 + 2, ..., l2 + vectorSize - 1]
            // h = syn0[l1, l1 + 1, l1 + 2, ..., l1 + vectorSize - 1]
            // -1 * eta * parenthesized expression = g
            JavaBlas.raxpy(vectorSize, g, syn0.data, l1, 1, syn1.data, l2, 1);
          }
        }
        // Update weights of syn0.
        // syn0[l1, l1 + 1, l1 + 2, ..., l1 + vectorSize - 1] += neu1e
        // This comes from equation (53) in Rong.
        // In Rong, h = syn0[l1, l1 + 1, l1 + 2, ..., l1 + vectorSize - 1].
        // This step does gradient descent, since as mentioned above neu1e equals
        // derivatives of the cost function wrt the entries of the output of the
        // hidden layer multiplied by the negative of the learning rate.
        // Also in word2vec, activation function at hidden layer is identity.
        // raxpy function is used to do this.
        JavaBlas.raxpy(vectorSize, 1.0f, neu1e.data, 0, 1, syn0.data, l1, 1);
      }
    }
  }

  /**
   * For every word2vec vector, compute its l2 (Euclidean) norm and store it in
   * an array. This is used while computing similar words.
   * @throws Exception Exception thrown if norm of some word is too small. This
   * possibly indicates training didn't happen properly.
   */
  public void populateNorms() throws Exception {
    log.info("Entered populateNorms.");
    // Initialize the array.
    syn0Norms = new FloatMatrix(vocabSize);
    for (int i = 0; i < vocabSize; i++) {
      // Compute the l2 norm for ith word and store it at index i.
      float l2 = syn0.getRange(i * vectorSize, (i + 1) * vectorSize).norm2();
      Preconditions.checkArgument(l2 > Constants.EPS_NORMS,
        "Word \"" + i + "\"'s vector has very small norm. The norm must be " +
          "greater than " + Constants.EPS_NORMS + ".");
      syn0Norms.put(i, l2);
    }
    log.info("Norms computed.");
  }

  /**
   * This function is used to get num most similar words to word from vocabulary.
   * The similarity of two words is the cosine similarity between their vectors.
   * Cosine similarity between two vectors is their dot product divided by the
   * product of their l2 norms.
   * Note: The most similar word, i.e., the word at index 0 of the returned list,
   * should be the word itself with similarity 1.0.
   * @param word Word for which similar words must be found.
   * @param num Number of most similar words to be found.
   * @return A list of (word, similarity) sorted in descending order by similarity.
   * @throws Exception Exception thrown if given word is not in vocabulary.
   */
  public List<Tuple> mostSimilar(String word, int num) throws Exception {
    // Check if word is in vocabulary.
    Preconditions.checkArgument(vocabHash.containsKey(word), "Word \"" + word +
      "\" not found in vocabulary.");
    // Check if syn0Norms has been initialized. If not, then initialize it.
    if (syn0Norms == null) {
      populateNorms();
    }
    // First, get the vector for the word.
    // To get vector, we need its index.
    int wordIndex = vocabHash.get(word);
    // Get the vector and normalize it by dividing each element by the
    // vector's l2 norm.
    FloatMatrix wordVector = syn0.getRange(
      wordIndex * vectorSize, (wordIndex + 1) * vectorSize)
      .div(syn0Norms.get(wordIndex));
    // This vector will be used to store cosine similarity.
    FloatMatrix cosineVec = FloatMatrix.zeros(vocabSize);
    // Now perform a matrix-vector multiplication to get a vector of size
    // vocabSize. The matrix is syn0, and the vector is the normalized vector of
    // input word.
    // This is stored in cosineVec.
    // ith element of this vector will need to be divided by the norm of ith
    // word to get a vector of cosine similarities.
    // sgemv function is used to perform the matrix vector multiplication
    // specified above.
    NativeBlas.sgemv('T', vectorSize, vocabSize, 1.0f, syn0.data, 0, vectorSize,
      wordVector.data, 0, 1, 0f, cosineVec.data, 0, 1);
    // Divide cosineVec[i] by syn0Norms[i] for all i.
    // Now it stores the cosine similarities.
    cosineVec.divi(syn0Norms);
    // Add elements to a bounded priority queue.
    BoundedPQ pq = new BoundedPQ(num);
    for (int i = 0; i < vocabSize; i++) {
      pq.add(new Tuple(vocab.get(i).word, cosineVec.data[i]));
    }
    // The returned list contains (word, cosine similarity) tuples sorted by
    // cosine similarity in descending order.
    return pq.asList();
  }

  /**
   * This function returns the set of vocabulary words as a set of Strings.
   * @return A set.
   */
  public Set<String> getVocab() {
    Set<String> vocabSet = new HashSet<>();
    for (VocabWord vw: vocab) {
      // Get the String representation.
      vocabSet.add(vw.word);
    }
    return vocabSet;
  }

  /**
   * This function writes a matrix to a BufferedWriter object.
   * A row is (row_ID, space, id, tab, row elements separated by spaces).
   * This way if a single file contains multiple matrices, all of them can be
   * easily read.
   * @param M Matrix to be written.
   * @param id ID of the matrix.
   * @param rows Number of rows of the matrix. This is needed because we store
   *             the matrices as a 1D matrix.
   * @param columns Number of columns of the matrix. This is needed because we
   *                store the matrices as a 1D matrix.
   * @param bw BufferedWriter object to write the matrix to.
   * @throws IOException Throws an IOException if there is an error in writing
   * the matrix.
   */
  public void writeMatrix(FloatMatrix M, int id, int rows, int columns,
    BufferedWriter bw) throws IOException {
    // Each row is (row_ID, space, id, tab, row elements separated by spaces)
    for (int i = 0; i < rows; i++) {
      bw.write(i + " " + id + "\t");
      int offset = i * columns;
      for (int j = 0; j < columns; j++) {
        bw.write(M.data[offset + j] + " ");
      }
      bw.write("\n");
    }
  }

  /**
   * This function writes the two matrices, syn0 and syn1, to a BufferedWriter
   * object.
   * @param bw BufferedWriter object specifying where to write.
   * @throws IOException Throws an IOException if there is an error in writing
   * the matrices.
   */
  public void writeMatrices(BufferedWriter bw) throws IOException {
    // Write syn0 with each vector on a separate line.
    writeMatrix(syn0, 0, vocabSize, vectorSize, bw);
    // Write syn1 with each vector on a separate line.
    writeMatrix(syn1, 1, vocabSize, vectorSize, bw);
  }

  /**
   * This function reads the matrices syn0 and syn1.
   * @param br BufferedReader object from which to read.
   * @throws IOException Throws an IOException if there is an error in reading
   * the matrices.
   */
  public void readMatrices(BufferedReader br) throws IOException {
    // There are (2 * vocabSize) rows.
    // Each row contains ID followed by space followed by 0 or 1 (denoting syn0
    // or syn1) followed by tab followed by vectorSize floats separated by spaces.
    syn0 = new FloatMatrix(vectorSize * vocabSize);
    syn1 = new FloatMatrix(vectorSize * vocabSize);
    String [] line;
    for (int i = 0; i < 2 * vocabSize; i++) {
      line = br.readLine().trim().split(Constants.TABS_REGEX);
      Preconditions.checkArgument(line.length == 2, "Problem in reading matrix.");
      String[] keyPair = line[0].split(Constants.WHITESPACE_REGEX);
      // idx denotes row number in matrix specified by matrixID.
      int idx = Integer.parseInt(keyPair[0]);
      Preconditions.checkArgument(0 <= idx && idx < vocabSize,
        "Wrong word index. It should be between 0 and " + (vocabSize - 1) +
          ". Got " + idx);
      // matrixID specifies whether it's a row in syn0 or syn1.
      int matrixID = Integer.parseInt(keyPair[1]);
      Preconditions.checkArgument(matrixID == 0 || matrixID == 1,
        "Wrong matrix ID. It should be either 0 or 1. Got " + matrixID + ".");
      // Entries of the row.
      String[] vals = line[1].trim().split(Constants.WHITESPACE_REGEX);
      Preconditions.checkArgument(vals.length == vectorSize,
        "Wrong row length. It should be " + vectorSize + ". Got " + vals.length +
          " elements.");
      int offset = idx * vectorSize;
      if (matrixID == 0) {
        for (int j = 0; j < vectorSize; j++) {
          syn0.data[offset + j] = Float.parseFloat(vals[j]);
        }
      } else {
        for (int j = 0; j < vectorSize; j++) {
          syn1.data[offset + j] = Float.parseFloat(vals[j]);
        }
      }
    }
  }

  /**
   * This function writes the necessary parameters needed to define the Huffman
   * tree.
   * @param bw BufferedWriter object specifying where to write.
   * @throws IOException Throws an IOException if there is an error in writing
   * the tree.
   */
  public void writeHuffmanTree(BufferedWriter bw) throws IOException {
    log.info("Writing Huffman tree.");
    bw.write(vectorSize + "\n");
    bw.write(vocabSize + "\n");
    // There are vocabSize rows, one for every word.
    // Each row contains (word, count, codeLen, code, point) separated by tabs.
    // Within code and point, values are separated by spaces.
    for (int i = 0; i < vocab.size(); i++) {
      bw.write(vocab.get(i).word + "\t" + vocab.get(i).count + "\t" +
        vocab.get(i).codeLen + "\t");
      for (int j = 0; j < vocab.get(i).code.length; j++) {
        // Trailing spaces won't cause a problem.
        bw.write(vocab.get(i).code[j] + " ");
      }
      bw.write("\t");
      for (int j = 0; j < vocab.get(i).point.length; j++) {
        // Trailing spaces won't cause a problem.
        bw.write(vocab.get(i).point[j] + " ");
      }
      bw.write("\n");
    }
    log.info("Huffman tree written.");
  }

  /**
   * This function reads the Huffman tree.
   * @param br BufferedReader object from which to read.
   * @throws IOException Throws an IOException if there is an error in reading
   * the tree.
   */
  public void readHuffmanTree(BufferedReader br) throws IOException {
    log.info("Reading Huffman tree.");
    String line;
    vectorSize = Integer.parseInt(br.readLine());
    vocabSize = Integer.parseInt(br.readLine());
    Preconditions.checkArgument(vectorSize > 0, "vectorSize must be > 0. Got " +
      vectorSize + ".");
    Preconditions.checkArgument(vocabSize > 0, "vocabSize must be > 0. Got " +
      vocabSize + ".");
    // Now read vocabSize rows.
    // Each row contains (word, count, codeLen, code, point) separated by tabs.
    // Within code and point, values are separated by spaces.
    vocab = new ArrayList<>();
    vocabHash = new HashMap<>();
    trainWordsCount = 0L;
    int idx = 0;
    while ((line = br.readLine()) != null) {
      String[] values = line.trim().split("\\t+");
      String word = values[0];
      long count = Long.parseLong(values[1]);
      Preconditions.checkArgument(count > 0, "count must be > 0. Got " + count);
      int codeLen = Integer.parseInt(values[2]);
      Preconditions.checkArgument(
        codeLen > 0, "codeLen must be > 0. Got " + codeLen);
      // Read a String representing code array. Split on whitespace, and then
      // convert each element to Integer using parseInt function.
      int[] code = Arrays.stream(values[3].trim().split("\\s+"))
        .mapToInt(Integer::parseInt).toArray();
      Preconditions.checkArgument(Arrays.stream(code).allMatch(x -> x == 0 || x == 1),
        "All elements of array 'code' must be either 0 or 1.");
      // Elements of point can be positive, zero or negative.
      // Read a String representing point array. Split on whitespace, and then
      // convert each element to Integer using parseInt function.
      int[] point = Arrays.stream(values[4].trim().split("\\s+"))
        .mapToInt(Integer::parseInt).toArray();
      vocab.add(idx, new VocabWord(word, count, code, point, codeLen));
      vocabHash.put(word, idx);
      trainWordsCount += count;
      idx++;
    }
    totalWordCount = numEpochs * trainWordsCount + 1;
    Preconditions.checkArgument(idx == vocabSize, "Huffman tree wasn't " +
      "properly read. It had "  + idx + " rows, while " + vocabSize + " rows " +
      "were expected.");
    log.info("Huffman tree read.");
  }

  /**
   * Save the model to a text file.
   * The saved model can be read using read function.
   * This is for saving to a local file.
   * @param path Path where the model will be saved.
   * @throws IOException Throws an IOException if there is an error in saving
   * the model.
   */
  public void save(String path) throws IOException {
    log.info("Saving model at " + path + ".");
    BufferedWriter bw = new BufferedWriter(new FileWriter(path));
    writeHuffmanTree(bw);
    writeMatrices(bw);
    // Write syn0Norms in a line.
    StringBuilder sb = new StringBuilder();
    for (int i = 0; i < vocabSize; i++) {
      // Trailing spaces won't cause a problem.
      sb.append(syn0Norms.data[i]).append(" ");
    }
    bw.write(sb.toString().trim() + "\n");
    // Write word and index.
    for (String w: vocabHash.keySet()) {
      bw.write(w + "\t" + vocabHash.get(w) + "\n");
    }
    bw.close();
    log.info("Model saved.");
  }

  /**
   * Read the model from a file.
   * The model must have been written using save function of this class.
   * This is for reading from a local file.
   * @param path Path where the model is saved.
   * @throws IOException Throws an IOException if there is an error in reading
   * the model.
   */
  public void read(String path) throws IOException {
    log.info("Reading model at " + path + ".");
    BufferedReader br = new BufferedReader(new FileReader(path));
    readHuffmanTree(br);
    readMatrices(br);
    // Read syn0Norms.
    // The next line is a vocabSize-dimensional array.
    String[] line;
    syn0Norms = new FloatMatrix(vocabSize);
    line = br.readLine().split("\\s+");
    for (int i = 0; i < vocabSize; i++) {
      syn0Norms.data[i] = Float.parseFloat(line[i]);
    }
    // Next vocabSize lines contain a word and its index separated by a space.
    Map<Integer, String> idToWord = new HashMap<>();
    for (int i = 0; i < vocabSize; i++) {
      line = br.readLine().split("\\s+");
      vocabHash.put(line[0], Integer.parseInt(line[1]));
      idToWord.put(Integer.parseInt(line[1]), line[0]);
    }
    br.close();
    // Populate "vocab" list.
    vocab = new ArrayList<>(vocabSize);
    for (int i = 0; i < vocabSize; i++) {
      vocab.add(i, new VocabWord(idToWord.get(i)));
    }
    log.info("Finished reading model.");
  }
}