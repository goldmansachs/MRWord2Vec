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
 * This interface contains various constants.
 */
public interface Constants {

  // Relative paths of various files inside directory containing trained model.
  // This directory is used for storing temporary files. It will get deleted
  // when the job is complete.
  String TEMP_DIR = "/temp";
  String WORDCOUNT_FILE_PATH = "/wordcount";
  String MODEL_PATH = "/model";
  String HUFFMAN_TREE_PATH = "/huffman";
  String NEAREST_NEIGHBOURS_PATH = "/nearestNeighbours";
  String TABS_REGEX = "\\t+";
  String WHITESPACE_REGEX = "\\s+";
  // A small value used to check if a float is very close to 0.
  double EPS_NORMS = 1e-10;
}