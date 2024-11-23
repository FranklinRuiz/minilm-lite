package com.minilm.lite;

import com.minilm.lite.utils.CosineSimilarityUtil;
import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.*;

class CosineSimilarityUtilTest {

    @Test
    void testIdenticalVectors() {
        double[] vec1 = {1.0, 2.0, 3.0};
        double[] vec2 = {1.0, 2.0, 3.0};

        double similarity = CosineSimilarityUtil.calculate(vec1, vec2);
        assertEquals(1.0, similarity, 1e-6, "Cosine similarity of identical vectors should be 1.0");
    }

    @Test
    void testOppositeVectors() {
        double[] vec1 = {1.0, 2.0, 3.0};
        double[] vec2 = {-1.0, -2.0, -3.0};

        double similarity = CosineSimilarityUtil.calculate(vec1, vec2);
        assertEquals(-1.0, similarity, 1e-6, "Cosine similarity of opposite vectors should be -1.0");
    }

    @Test
    void testOrthogonalVectors() {
        double[] vec1 = {1.0, 0.0, 0.0};
        double[] vec2 = {0.0, 1.0, 0.0};

        double similarity = CosineSimilarityUtil.calculate(vec1, vec2);
        assertEquals(0.0, similarity, 1e-6, "Cosine similarity of orthogonal vectors should be 0.0");
    }
}
