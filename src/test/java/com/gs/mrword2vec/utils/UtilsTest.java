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

import org.junit.Test;

import java.util.HashMap;
import java.util.Map;

import static junit.framework.TestCase.assertEquals;
import static junit.framework.TestCase.assertTrue;
import static junit.framework.TestCase.assertFalse;

/**
 * Test class for Utils.class.
 */
public class UtilsTest {

  @Test
  public void testAddOneToMap() {
    Map<String, Long> map = new HashMap<>();
    // Adding "a" n1 times.
    int n1 = 12345;
    for (int i = 0; i < n1; i++) {
      Utils.addOneToMap(map, "a");
    }
    // Adding "b" n2 times.
    int n2 = 54321;
    for (int i = 0; i < n2; i++) {
      Utils.addOneToMap(map, "b");
    }
    // Two elements with keys "a' and "b".
    assertTrue(map.size() == 2);
    // The function "addOneToMap" was called n1 times on "a".
    assertTrue(map.get("a") == n1);
    // The function "addOneToMap" was called n2 times on "b".
    assertTrue(map.get("b") == n2);
    // The function "addOneToMap" was never called on "c", so it shouldn't be
    // present.
    assertFalse(map.containsKey("c"));
  }

  @Test
  public void testGetSigmoidTable() {
    float[] table1 = Utils.getSigmoidTable(1, 1);
    // table1 should contain just one entry with value
    // 1/(1 + e) ~ 0.26894142137.
    assertTrue(table1.length == 1);
    assertEquals(0.26894142137, table1[0], 1e-6);
    float[] table2 = Utils.getSigmoidTable(5, 1000);
    // table2 should contain 1000 entries.
    assertTrue(table2.length == 1000);
    // Checking all entries.
    for (int i = 0; i < 1000; i++) {
      // The ith entry should equal sigmoid(x) = 1 / (1 + e^(-x)),
      // where x = -5 + i * 0.01.
      // Here, 0.01 is the granularity size because [-5, 5) is split into 1000
      // parts.
      double expectedValue = 1.0 / (1.0 + Math.exp(-(-5 + i * 0.01)));
      assertEquals(expectedValue, table2[i], 1e-6);
    }
  }

  @Test
  public void testSortByValue() {
    Map<String, Long> map = new HashMap<>();
    map.put("a", 2L);
    map.put("b", 3L);
    map.put("c", 1L);
    map.put("d", 5L);
    map.put("e", 4L);
    Map<String, Long> sortedMap = Utils.sortByValue(map);
    // temp initially is greater than all values in map.
    long temp = 6L;
    // This for loop will iterate over the elements in sortedMap, which are
    // sorted in descending order.
    for (String k: sortedMap.keySet()) {
      long value = sortedMap.get(k);
      assertTrue(value < temp);
      temp = value;
    }
  }
}