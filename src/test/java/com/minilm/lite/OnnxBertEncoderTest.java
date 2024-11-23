package com.minilm.lite;

import com.minilm.lite.encoder.OnnxBertEncoder;
import org.junit.jupiter.api.Test;

import java.io.InputStream;

import static org.junit.jupiter.api.Assertions.*;

class OnnxBertEncoderTest {

    private OnnxBertEncoder initializeEncoder() {
        InputStream modelStream = getClass().getClassLoader().getResourceAsStream("all-minilm-l6-v2.onnx");
        InputStream tokenizerStream = getClass().getClassLoader().getResourceAsStream("all-minilm-l6-v2-tokenizer.json");

        assertNotNull(modelStream, "Model file should be found in resources.");
        assertNotNull(tokenizerStream, "Tokenizer file should be found in resources.");

        return assertDoesNotThrow(() ->
                        new OnnxBertEncoder(modelStream, tokenizerStream, OnnxBertEncoder.PoolingMode.MEAN),
                "Failed to initialize OnnxBertEncoder"
        );
    }

    @Test
    void testTokenCount() {
        OnnxBertEncoder encoder = initializeEncoder();

        assertDoesNotThrow(() -> {
            int tokenCount = encoder.countTokens("Hello world");
            assertTrue(tokenCount > 0, "Token count should be greater than 0");
        });
    }

    @Test
    void testEmbed() {
        OnnxBertEncoder encoder = initializeEncoder();

        assertDoesNotThrow(() -> {
            OnnxBertEncoder.EmbeddingAndTokenCount result = encoder.embed("Hello world");

            assertNotNull(result, "Result should not be null");
            assertNotNull(result.embedding, "Embedding should not be null");
            assertTrue(result.embedding.length > 0, "Embedding should have a valid length");
            assertTrue(result.tokenCount > 0, "Token count should be greater than 0");
        });
    }
}
