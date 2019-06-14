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
package com.gs.mrword2vec.mapreduce.utils;

import com.google.common.base.Preconditions;
import org.apache.hadoop.io.Writable;
import org.jblas.FloatMatrix;

import java.io.DataInput;
import java.io.DataOutput;
import java.io.IOException;

/**
 * This class implements a Writable implementation of vector. This is used for
 * sending output from Word2VecMapper to Word2VecReducer. The key is the word,
 * and the value is the Vector.
 */
public class Vector implements Writable {

  // The content of the vector.
  public float[] values;
  // Length of the vector.
  public int size = -1;

  /**
   * Empty constructor.
   */
  public Vector() {
  }

  /**
   * Constructor to construct a vector from an array.
   */
  public Vector(float[] v) {
    size = v.length;
    values = new float[size];
    // Copies an array from the specified source array, beginning at the
    // specified position, to the specified position of the destination array.
    System.arraycopy(v, 0, values, 0, size);
  }

  /**
   * Constructor using a 1-D FloatMatrix for the entries.
   * @param M 1-D FloatMatrix from which to read the entries.
   * @param start Starting index.
   * @param end Last index + 1.
   */
  public Vector(FloatMatrix M, int start, int end) {
    size = end - start;
    values = new float[size];
    for (int i = 0; i < size; i++) {
      // Put (start + i)th entry of M in the ith position of values.
      values[i] = M.data[start + i];
    }
  }

  /**
   * Add a vector to this vector.
   * @param v Vector to be added.
   */
  public void add(Vector v) {
    // If this object hasn't been initialized.
    if (size == -1) {
      size = v.size;
      values = new float[size];
      // Copies an array from the specified source array, beginning at the
      // specified position, to the specified position of the destination array.
      System.arraycopy(v.values, 0, values, 0, size);
    } else {
      Preconditions.checkArgument(size == v.size, "Vectors must be of the " +
        "same dimension. Got " + v.size + ". Expected " + size + ".");
      for (int i = 0; i < size; i++) {
        values[i] += v.values[i];
      }
    }
  }

  /**
   * Divide the entries by a scalar.
   * @param c Scalar.
   */
  public void divide(int c) {
    for (int i = 0; i < size; i++) {
      values[i] /= c;
    }
  }

  // The next two functions need to be implemented for this class to be able to
  // implement Writable interface. They specify how this class' objects will be
  // serialized and deserialized.

  /**
   * This function deserializes the fields of this object from "in".
   * @param in <code>DataInput</code> object to deserialize Vector object from.
   * @throws IOException If unable to deserialize.
   */
  @Override
  public void readFields(DataInput in) throws IOException {
    size = in.readInt();
    values = new float[size];
    for (int i = 0; i < size; i++) {
      values[i] = in.readFloat();
    }
  }

  /**
   * This function serializes the fields of this object to "out".
   * @param out <code>DataOuput</code> object to serialize Vector object into.
   * @throws IOException If unable to serialize.
   */
  @Override
  public void write(DataOutput out) throws IOException {
    out.writeInt(size);
    for (int i = 0; i < size; i++) {
      out.writeFloat(values[i]);
    }
  }
}