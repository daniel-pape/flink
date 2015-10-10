/*
 * Licensed to the Apache Software Foundation (ASF) under one
 * or more contributor license agreements.  See the NOTICE file
 * distributed with this work for additional information
 * regarding copyright ownership.  The ASF licenses this file
 * to you under the Apache License, Version 2.0 (the
 * "License"); you may not use this file except in compliance
 * with the License.  You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

package org.apache.flink.ml.math

/** Base trait for a matrix representation
  *
  */
trait Matrix {

  /** Number of rows
    *
    * @return
    */
  def numRows: Int

  /** Number of columns
    *
    * @return
    */
  def numCols: Int

  /** Element wise access function
    *
    * @param row row index
    * @param col column index
    * @return matrix entry at (row, col)
    */
  def apply(row: Int, col: Int): Double

  /** Element wise update function
    *
    * @param row row index
    * @param col column index
    * @param value value to set at (row, col)
    */
  def update(row: Int, col: Int, value: Double): Unit

  /** Copies the matrix instance
    *
    * @return Copy of itself
    */
  def copy: Matrix

  /**
    * Returns copy of `this` but with the function
    * `f` applied to each entry.
    */
  def map(f: Double => Double): Matrix

  /**
   * Zero matrix of dimension `numRows` times `numCols`
   * @return
   */
  def zero: Matrix

  // TODO: Dummy impl
  def +(other: Matrix): Matrix = other

  /**
    * Returns `this` multiplied entry-wise with the
    * [[Double]] `scalar`.
    */
  def *(scalar: Double): Matrix



  /** Returns the `colNum`-th column as [[Vector]]
    *
    * @param colNum Position of the column to return
    * @return The `colNum`-th column
    */
  def getColumn(colNum: Int): Vector

  /** Returns the columns of this matrix as [[IndexedSeq]]
    * of [[Vector]]s indexed by column index (starting with zero).
    *
    * @return The columns of this matrix
    */
  def getColumns: IndexedSeq[Vector]

  /**
    * Returns the sample covariance matrix of `this` considering
    * each row of `this` as an observation.
    *
    * That is `this` is considered as data matrix in which
    * each column describes the observations of a single feature
    * across different observations.
    *
    * @return
    */
  final def cov(): Matrix = {
    val columns: IndexedSeq[Vector] = this.getColumns
    val innerProduct: Matrix = columns.map {
      column => column.outer(column)
    }.fold(this.zero)(_ + _)

    innerProduct * (1 / this.numCols)
  }

  def equalsMatrix(matrix: Matrix): Boolean = {
    if(numRows == matrix.numRows && numCols == matrix.numCols) {
      val coordinates = for(row <- 0 until numRows; col <- 0 until numCols) yield (row, col)
      coordinates forall { case(row, col) => this.apply(row, col) == matrix(row, col)}
    } else {
      false
    }
  }

}
