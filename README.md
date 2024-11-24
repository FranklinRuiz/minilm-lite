
# MiniLM-Lite: Lightweight Text Embedding Library

**MiniLM-Lite** is a Java library for generating text embeddings, computing similarity, and managing embedding stores for tasks such as text similarity, semantic search, and clustering. Built on top of ONNX Runtime and the `all-MiniLM-L6-v2` model, this library provides a simple interface for developers looking to integrate embedding generation into their Java projects.

---

## Features

- **Text Embedding Generation**: Convert text into high-dimensional embeddings using the `all-MiniLM-L6-v2` model.
- **Cosine Similarity Calculation**: Efficiently compute the similarity between embeddings.
- **Embedding Storage**: Store and retrieve relevant items using an embedding store with similarity-based scoring.
- **Simple API**: Easy-to-use interfaces for embedding generation and similarity matching.
- **Optimized for Performance**: Powered by ONNX Runtime for efficient inference.

---

## Installation

### Prerequisites

1. **Java 17+**: Ensure Java is installed and configured.
2. **Maven**: This library is distributed via Maven.

### Maven Dependency

Add the following dependency to your `pom.xml` file:

```xml
<dependency>
    <groupId>io.github.franklinruiz</groupId>
    <artifactId>minilm-lite</artifactId>
    <version>1.0.0</version>
</dependency>
```

---

## Getting Started

### 1. Initializing the Library

The library provides a default embedding model and tokenizer that can be loaded directly:

```java
import io.github.franklinruiz.encoder.MiniLMEmbedder;

MiniLMEmbedder embedder = MiniLMEmbedder.getDefaultModel();
```

### 2. Generating Embeddings

Generate embeddings for any input text using the `embed` method:

```java
double[] embedding = embedder.embed("Hello world!");
System.out.println("Embedding Length: " + embedding.length);
```

---

### 3. Calculating Cosine Similarity

Use `CosineSimilarityUtil` to compute the similarity between two embeddings:

```java
import io.github.franklinruiz.utils.CosineSimilarityUtil;

double[] vec1 = embedder.embed("Hello world!");
double[] vec2 = embedder.embed("Hi there!");

double similarity = CosineSimilarityUtil.calculate(vec1, vec2);
System.out.println("Cosine Similarity: "+similarity);
```

---

### 4. Using the Embedding Store

Store embeddings and perform similarity-based queries with the `EmbeddingStore`.

#### Adding Items to the Store

```java
import io.github.franklinruiz.store.EmbeddingStore;
import io.github.franklinruiz.store.TextSegment;

EmbeddingStore<TextSegment> store = EmbeddingStore.initialize();

store.addItem(new TextSegment("Lugares turísticos en Perú"));
store.addItem(new TextSegment("Montañas de los Andes"));
```

#### Querying the Store

Find the most relevant items to a query:

```java
double[] queryEmbedding = embedder.embed("famous places in Peru");
List<EmbeddingMatch<TextSegment>> matches = store.findRelevant(queryEmbedding, 2);

for (EmbeddingMatch<TextSegment> match : matches) {
    System.out.println("Text: " + match.getItem().getText() + ", Similarity: " + match.getScore());
}
```

---

### 5. Full Example: Semantic Search

```java
import io.github.franklinruiz.encoder.MiniLMEmbedder;
import io.github.franklinruiz.store.EmbeddingStore;
import io.github.franklinruiz.store.TextSegment;
import io.github.franklinruiz.store.EmbeddingMatch;

import java.util.List;

public class Main {
    public static void main(String[] args) throws Exception {
        // Initialize the embedder and store
        MiniLMEmbedder embedder = MiniLMEmbedder.getDefaultModel();
        EmbeddingStore<TextSegment> store = EmbeddingStore.initialize();

        // Add items to the store
        store.addItem(new TextSegment("Paris is the capital of France."));
        store.addItem(new TextSegment("The Eiffel Tower is located in Paris."));
        store.addItem(new TextSegment("Berlin is the capital of Germany."));

        // Query the store
        double[] queryEmbedding = embedder.embed("What is the capital of France?");
        List<EmbeddingMatch<TextSegment>> results = store.findRelevant(queryEmbedding, 2);

        // Print results
        for (EmbeddingMatch<TextSegment> match : results) {
            System.out.println("Matched Text: " + match.getItem().getText() + ", Similarity: " + match.getScore());
        }
    }
}
```

---

## Project Structure

- **`MiniLMEmbedder`**: Simplifies embedding generation using the `all-MiniLM-L6-v2` model.
- **`OnnxBertEncoder`**: Handles text tokenization and embedding pooling.
- **`CosineSimilarityUtil`**: Computes cosine similarity between embeddings.
- **`EmbeddingStore`**: Stores embeddings and retrieves relevant items based on similarity scoring.
- **`TextSegment`**: Wrapper for storing textual data in the embedding store.

---

## Tests

This library includes unit tests for all major components. Run the tests using Maven:

```bash
mvn test
```

### Test Coverage

- **Embedding Generation:** Ensures the embeddings are consistent and correct.
- **Cosine Similarity:** Validates the similarity calculations.
- **Embedding Store:** Tests adding and querying items for relevance.

---

## Requirements

- **Java 17+**
- **ONNX Runtime**: Efficient execution of the MiniLM model.
- **Hugging Face Tokenizers**: Preprocessing for text tokenization.

### Required Files

Ensure the following files are included in your `resources` folder:
- `all-minilm-l6-v2.onnx`
- `all-minilm-l6-v2-tokenizer.json`

These files are included with the library if installed via Maven.

---

## Contributing

We welcome contributions! To contribute:

1. Fork the repository.
2. Create a new branch (`git checkout -b feature/your-feature`).
3. Commit your changes (`git commit -m "Add feature"`).
4. Push your changes (`git push origin feature/your-feature`).
5. Submit a Pull Request.

---

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for more details.
