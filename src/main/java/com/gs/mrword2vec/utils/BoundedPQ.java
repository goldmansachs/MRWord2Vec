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

import java.util.ArrayList;
import java.util.Collections;
import java.util.List;
import java.util.PriorityQueue;

/**
 * This class implements a bounded priority queue for Tuple objects.
 * "add" function allows as many Tuples as desired to be added, but only maxSize
 * largest Tuples will be maintained. Here Tuple t1 is larger than Tuple t2 iff
 * t1.score > t2.score.
 */
public class BoundedPQ {
  // Maximum number of elements that will be stored in priority queue.
  private final int maxSize;
  // Underlying data structure.
  private PriorityQueue<Tuple> pq;

  public BoundedPQ(int maxSize) {
    this.maxSize = maxSize;
    this.pq = new PriorityQueue<>();
  }

  public void add(Tuple t) {
    // Throw NullPointerException if null.
    if (t == null) {
      throw new NullPointerException("Element to be added is null.");
    }
    // pq is full.
    if (maxSize == pq.size()) {
      // Least element of pq.
      Tuple l = pq.peek();
      // t > l => we update pq, otherwise we don't.
      if (t.compareTo(l) > 0) {
        // Remove l, and put t.
        pq.poll();
        pq.add(t);
      }
    } else {
      // pq isn't full. So just add t.
      pq.add(t);
    }
  }

  /**
   * Converts the entries in the priority queue to a sorted list.
   * @return A sorted list containing the elements of priority queue in
   * descending order.
   */
  public List<Tuple> asList() {
    List<Tuple> list = new ArrayList<>(this.pq);
    Collections.sort(list);
    Collections.reverse(list);
    return list;
  }
}