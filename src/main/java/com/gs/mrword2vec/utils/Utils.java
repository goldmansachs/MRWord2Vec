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

import java.io.BufferedReader;
import java.io.FileReader;
import java.io.IOException;
import java.util.*;
import java.util.stream.Collectors;

/**
 * Implements various utility functions.
 */
public class Utils {

  /**
   * This function adds 1 to the value corresponding to the specified key in the
   * specified map. If the key doesn't exist, a new entry is created for the key
   * with value 1.
   * @param map : Map which must be updated.
   * @param w : Key.
   */
  public static void addOneToMap(Map<String, Long> map, String w) {
    if (map.containsKey(w)) {
      map.put(w, map.get(w) + 1);
    } else {
      map.put(w, 1L);
    }
  }

  /**
   * This function is used to pre-compute and store values of sigmoid function
   * in an array on "size" values inside the domain D = [-maxVal, maxVal).
   * D is split into "size" equal parts and the returned array's ith index
   * contains value for ith part.
   * @param maxVal : Specifies the length of domain on which to compute sigmoid.
   * @param size : Number of parts to split the domain into. Also equal to the
   *             length of the returned array.
   * @return : A float array containing values of sigmoid at "size" points.
   */
  public static float[] getSigmoidTable(int maxVal, int size) {
    float[] sigmoidTable = new float[size];
    // Divide the range from -maxVal to +maxVal into "size" intervals of the
    // same length and compute sigmoid(z) at the left endpoint z of each interval.
    for (int i = 0; i < size; i++) {
      // z = left endpoint of ith interval
      // temp = exp(z)
      double temp = Math.exp(((2.0 * i) / size - 1.0) * maxVal);
      // sigmoid(z) = exp(z) / (1.0 + exp(z))
      sigmoidTable[i] = (float) (temp / (1.0 + temp));
    }
    return sigmoidTable;
  }

  /**
   * This functions gets a Map as input and converts it to a LinkedHashMap which
   * is sorted in descending order by the value.
   * LinkedHashMap is a doubly-linked list implementation of Map, which allows
   * predictable iteration order unlike HashMap.
   * @param map Map whose entries need to be sorted.
   * @return A LinkedHashMap sorted in descending order by value.
   */
  public static Map<String, Long> sortByValue(Map<String, Long> map) {
    return map.entrySet() // Get the entries in map as a Set
      .stream() // Get a sequential stream
      .sorted(Map.Entry.<String, Long>comparingByValue().reversed()) // Sort the stream
      .collect(Collectors.toMap(Map.Entry::getKey, Map.Entry::getValue,
        (e1, e2) -> e1, LinkedHashMap::new)); // collect the entries and store them
  }

  /**
   * This function reads a file containing sentences.
   * Each sentence should be in a separate line, with its words separated by
   * space.
   * @param path Path to the file containing the sentences.
   * @return A list of sentences.
   * @throws IOException When unable to read the file.
   */
  public static List<String[]> readFile(String path) throws IOException {
    List<String[]> sentences = new ArrayList<>();
    BufferedReader br = new BufferedReader(new FileReader(path));
    String line;
    while ((line = br.readLine()) != null) {
      sentences.add(line.split("\\s+"));
    }
    return sentences;
  }
}