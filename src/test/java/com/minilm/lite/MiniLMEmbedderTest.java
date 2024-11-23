package com.minilm.lite;

import com.minilm.lite.encoder.MiniLMEmbedder;
import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.*;

class MiniLMEmbedderTest {
    @Test
    void testGetDefaultModel() {
        assertDoesNotThrow(() -> {
            MiniLMEmbedder embedder = MiniLMEmbedder.getDefaultModel();
            assertNotNull(embedder, "Default model should not be null");
        });
    }

    @Test
    void testEmbed() {
        assertDoesNotThrow(() -> {
            MiniLMEmbedder embedder = MiniLMEmbedder.getDefaultModel();
            double[] embedding = embedder.embed("Hello world");
            assertNotNull(embedding, "Embedding should not be null");
            assertTrue(embedding.length > 0, "Embedding should have a valid length");
        });
    }
}
