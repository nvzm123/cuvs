/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
package com.nvidia.cuvs.lucene;

import static org.apache.lucene.index.VectorEncoding.FLOAT32;
import static org.apache.lucene.index.VectorSimilarityFunction.EUCLIDEAN;

import java.io.IOException;
import java.util.ArrayList;
import java.util.List;
import java.util.Map;
import org.apache.lucene.codecs.hnsw.FlatFieldVectorsWriter;
import org.apache.lucene.index.DocValuesSkipIndexType;
import org.apache.lucene.index.DocValuesType;
import org.apache.lucene.index.DocsWithFieldSet;
import org.apache.lucene.index.FieldInfo;
import org.apache.lucene.index.IndexOptions;
import org.apache.lucene.index.Sorter;
import org.apache.lucene.tests.util.LuceneTestCase;

public class TestPostIngestGraphFields extends LuceneTestCase {

  public void testNormalOrderUsesAllAcceptedVectors() throws Exception {
    FieldWriter field = newField("vector", 0, 2);
    float[] first = new float[] {1.0f, 2.0f};
    float[] second = new float[] {3.0f, 4.0f};
    field.addValue(0, first);
    field.addValue(1, second);

    var prepared = Lucene99AcceleratedHNSWVectorsWriter.prepareGraphFields(List.of(field), null);

    assertEquals(1, prepared.size());
    assertEquals("vector", prepared.getFirst().fieldInfo().name);
    assertSame(first, prepared.getFirst().vectors().get(0));
    assertSame(second, prepared.getFirst().vectors().get(1));
  }

  public void testSparseDocumentsDoNotBecomePlaceholderVectors() throws Exception {
    FieldWriter field = newField("sparse", 0, 2);
    float[] first = new float[] {1.0f, 2.0f};
    float[] second = new float[] {3.0f, 4.0f};
    field.addValue(2, first);
    field.addValue(9, second);

    var prepared = Lucene99AcceleratedHNSWVectorsWriter.prepareGraphFields(List.of(field), null);

    assertEquals(2, prepared.getFirst().vectors().size());
    assertSame(first, prepared.getFirst().vectors().get(0));
    assertSame(second, prepared.getFirst().vectors().get(1));
  }

  public void testSortedOrderMatchesNewDocumentOrder() throws Exception {
    FieldWriter field = newField("sorted", 0, 2);
    float[] oldDocZero = new float[] {1.0f, 2.0f};
    float[] oldDocTwo = new float[] {3.0f, 4.0f};
    field.addValue(0, oldDocZero);
    field.addValue(2, oldDocTwo);

    var prepared =
        Lucene99AcceleratedHNSWVectorsWriter.prepareGraphFields(List.of(field), reversingDocMap(4));

    assertEquals(2, prepared.getFirst().vectors().size());
    assertSame(oldDocTwo, prepared.getFirst().vectors().get(0));
    assertSame(oldDocZero, prepared.getFirst().vectors().get(1));
  }

  public void testMultipleFieldsRemainIndependent() throws Exception {
    FieldWriter title = newField("title", 0, 2);
    FieldWriter body = newField("body", 1, 3);
    float[] titleVector = new float[] {1.0f, 2.0f};
    float[] bodyVector = new float[] {3.0f, 4.0f, 5.0f};
    title.addValue(0, titleVector);
    body.addValue(0, bodyVector);

    var prepared =
        Lucene99AcceleratedHNSWVectorsWriter.prepareGraphFields(List.of(title, body), null);

    assertEquals(2, prepared.size());
    assertEquals("title", prepared.get(0).fieldInfo().name);
    assertSame(titleVector, prepared.get(0).vectors().getFirst());
    assertEquals("body", prepared.get(1).fieldInfo().name);
    assertSame(bodyVector, prepared.get(1).vectors().getFirst());
  }

  private static FieldWriter newField(String name, int number, int dimensions) {
    FieldInfo fieldInfo =
        new FieldInfo(
            name,
            number,
            false,
            false,
            false,
            IndexOptions.NONE,
            DocValuesType.NONE,
            DocValuesSkipIndexType.NONE,
            -1,
            Map.of(),
            0,
            0,
            0,
            dimensions,
            FLOAT32,
            EUCLIDEAN,
            false,
            false);
    return new FieldWriter(
        AcceleratedHNSWUtils.QuantizationType.NONE,
        fieldInfo,
        new RecordingFlatFieldVectorsWriter());
  }

  private static Sorter.DocMap reversingDocMap(int size) {
    return new Sorter.DocMap() {
      @Override
      public int oldToNew(int docID) {
        return size - 1 - docID;
      }

      @Override
      public int newToOld(int docID) {
        return size - 1 - docID;
      }

      @Override
      public int size() {
        return size;
      }
    };
  }

  private static final class RecordingFlatFieldVectorsWriter
      extends FlatFieldVectorsWriter<float[]> {
    private final List<float[]> vectors = new ArrayList<>();
    private final DocsWithFieldSet docsWithField = new DocsWithFieldSet();
    private boolean finished;

    @Override
    public void addValue(int docID, float[] vectorValue) {
      docsWithField.add(docID);
      vectors.add(vectorValue);
    }

    @Override
    public float[] copyValue(float[] vectorValue) {
      return vectorValue.clone();
    }

    @Override
    public long ramBytesUsed() {
      return 0;
    }

    @Override
    public List<float[]> getVectors() {
      return vectors;
    }

    @Override
    public DocsWithFieldSet getDocsWithFieldSet() {
      return docsWithField;
    }

    @Override
    public void finish() throws IOException {
      finished = true;
    }

    @Override
    public boolean isFinished() {
      return finished;
    }
  }
}
