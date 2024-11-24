package io.github.franklinruiz;

import io.github.franklinruiz.encoder.MiniLMEmbedder;
import io.github.franklinruiz.store.EmbeddingMatch;
import io.github.franklinruiz.store.EmbeddingStore;
import io.github.franklinruiz.store.TextSegment;
import org.junit.jupiter.api.Test;

import java.util.List;

import static org.junit.jupiter.api.Assertions.*;

class EmbeddingStoreTest {
    @Test
    void testInitializeStore() {
        assertDoesNotThrow(() -> {
            EmbeddingStore<TextSegment> store = EmbeddingStore.initialize();
            assertNotNull(store, "EmbeddingStore should not be null");
        });
    }

    @Test
    void testAddAndRetrieveItems() {
        EmbeddingStore<TextSegment> store = EmbeddingStore.initialize();
        TextSegment segment1 = new TextSegment("Hello world");
        TextSegment segment2 = new TextSegment("Goodbye world");

        assertDoesNotThrow(() -> {
            store.addItem(segment1);
            store.addItem(segment2);

            MiniLMEmbedder embedder = MiniLMEmbedder.getDefaultModel();
            double[] queryEmbedding = embedder.embed("Hello again");

            double[] firstEmbedding = embedder.embed(segment1.getText());
            assertEquals(firstEmbedding.length, queryEmbedding.length, "Embedding lengths must match.");

            List<EmbeddingMatch<TextSegment>> results = store.findRelevant(queryEmbedding, 2);

            assertNotNull(results, "Results should not be null");
            assertEquals(2, results.size(), "Should retrieve the correct number of items");
        });
    }
}
