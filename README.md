
# MiniLM-Lite: Lightweight Text Embedding Library
[![Author](https://img.shields.io/badge/Author-Franklin%20Ruiz-blue)](#)
[![Java](https://img.shields.io/badge/Java-17%2B-brightgreen)](#)
[![ONNX Runtime](https://img.shields.io/badge/ONNX%20Runtime-1.20.0-blue)](#)
[![Maven](https://img.shields.io/badge/Maven-Build%20Tool-yellowgreen)](#)
[![all-MiniLM-L6-v2](https://img.shields.io/badge/Model-all--MiniLM--L6--v2-orange)](#)

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
    <version>1.0.1</version>
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
#### Semantic Text Matching: Querying and Finding Relevant Results

```java
import io.github.franklinruiz.encoder.MiniLMEmbedder;
import io.github.franklinruiz.store.EmbeddingStore;
import io.github.franklinruiz.store.TextSegment;
import io.github.franklinruiz.store.EmbeddingMatch;

import java.util.List;

public class Main {
    public static void main(String[] args) {
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
            System.out.println("Matched Text: " + match.item().getText() + ", Similarity: " + match.score());
        }
    }
}
```

```bash
# Result
Matched Text: Paris is the capital of France., Similarity: 0.8560696763850919
Matched Text: The Eiffel Tower is located in Paris., Similarity: 0.41220432645634725
```

#### Topic Classification for Articles Based on Semantic Similarity

```java
import io.github.franklinruiz.encoder.MiniLMEmbedder;
import io.github.franklinruiz.store.EmbeddingStore;
import io.github.franklinruiz.store.TextSegment;
import io.github.franklinruiz.store.EmbeddingMatch;

import java.util.List;

public class Main {
    public static void main(String[] args) {
        // Initialize the embedder and store
        MiniLMEmbedder embedder = MiniLMEmbedder.getDefaultModel();
        EmbeddingStore<TextSegment> topicStore = EmbeddingStore.initialize();

        // Add predefined topics with descriptions to the embedding store
        topicStore.addItem(new TextSegment("Technology: Innovations, advancements in software, artificial intelligence."));
        topicStore.addItem(new TextSegment("Health: Nutrition, physical and mental well-being, medical care."));
        topicStore.addItem(new TextSegment("Education: Learning methods, teaching, and pedagogy."));
        topicStore.addItem(new TextSegment("Sports: Football, basketball, sports competitions, and training."));

        // List of articles to classify by topic
        List<String> articles = List.of(
                "Artificial intelligence is transforming the way businesses operate.",
                "A balanced diet and daily exercise significantly improve mental health.",
                "New pedagogical strategies are being implemented in schools across the world.",
                "The Football World Cup is one of the most anticipated sports events of the year."
        );

        // Iterate over each article to suggest the most relevant topic
        for (String article : articles) {
            // Generate the embedding for the article
            double[] articleEmbedding = embedder.embed(article);

            // Find the most relevant topic using the embedding store (Top-1 result)
            List<EmbeddingMatch<TextSegment>> matches = topicStore.findRelevant(articleEmbedding, 1);

            // Print the article and the suggested topic
            if (!matches.isEmpty()) {
                EmbeddingMatch<TextSegment> bestMatch = matches.get(0);
                System.out.println("Article: \"" + article + "\"");
                System.out.println("Suggested Topic: " + bestMatch.item().getText());
                System.out.println("Similarity: " + bestMatch.score());
                System.out.println();
            }
        }
    }
}
```
```bash
# Result
Article: "Artificial intelligence is transforming the way businesses operate."
Suggested Topic: Technology: Innovations, advancements in software, artificial intelligence.
Similarity: 0.5146333825278299

Article: "A balanced diet and daily exercise significantly improve mental health."
Suggested Topic: Health: Nutrition, physical and mental well-being, medical care.
Similarity: 0.45945618435822316

Article: "New pedagogical strategies are being implemented in schools across the world."
Suggested Topic: Education: Learning methods, teaching, and pedagogy.
Similarity: 0.599657797185379

Article: "The Football World Cup is one of the most anticipated sports events of the year."
Suggested Topic: Sports: Football, basketball, sports competitions, and training.
Similarity: 0.43951509236857456
```

#### Duplicate Text Detection Using Semantic Similarity

```java
import io.github.franklinruiz.encoder.MiniLMEmbedder;
import io.github.franklinruiz.store.EmbeddingStore;
import io.github.franklinruiz.store.TextSegment;
import io.github.franklinruiz.store.EmbeddingMatch;

import java.util.List;

public class Main {
    public static void main(String[] args) {
        // Initialize the embedder and store
        MiniLMEmbedder embedder = MiniLMEmbedder.getDefaultModel();
        EmbeddingStore<TextSegment> textStore = EmbeddingStore.initialize();

        // List of texts to analyze for duplicates
        List<String> texts = List.of(
                "The quick brown fox jumps over the lazy dog.",
                "A fast, brown fox leaps over a sleeping dog.",
                "Artificial intelligence is transforming industries worldwide.",
                "AI is revolutionizing industries across the globe.",
                "The quick brown fox jumps over the lazy dog.",
                "Machine learning is a subset of artificial intelligence."
        );

        // Add texts to the embedding store
        for (String text : texts) {
            textStore.addItem(new TextSegment(text));
        }

        // Threshold for similarity (e.g., 0.90 means 90% similarity)
        double similarityThreshold = 0.90;

        // Find duplicates
        System.out.println("Duplicate or similar texts:");
        for (TextSegment segment : textStore.getAllItems()) {
            List<EmbeddingMatch<TextSegment>> matches = textStore.findRelevant(
                    embedder.embed(segment.getText()), texts.size());

            for (EmbeddingMatch<TextSegment> match : matches) {
                if (!match.item().equals(segment) && match.score() >= similarityThreshold) {
                    System.out.println("\"" + segment.getText() + "\" is similar to \""
                            + match.item().getText() + "\" with similarity: " + match.score());
                }
            }
        }
    }
}
```

```bash
# Duplicate or similar texts
"The quick brown fox jumps over the lazy dog." is similar to "The quick brown fox jumps over the lazy dog." with similarity: 0.9999999999999999
"The quick brown fox jumps over the lazy dog." is similar to "The quick brown fox jumps over the lazy dog." with similarity: 0.9999999999999999
```

#### Sentiment Analysis of User Opinions Using Semantic Similarity

```java
import io.github.franklinruiz.encoder.MiniLMEmbedder;
import io.github.franklinruiz.store.EmbeddingStore;
import io.github.franklinruiz.store.TextSegment;
import io.github.franklinruiz.store.EmbeddingMatch;

import java.util.List;

public class Main {
    public static void main(String[] args) {
        // Initialize the embedder and store
        MiniLMEmbedder embedder = MiniLMEmbedder.getDefaultModel();
        EmbeddingStore<TextSegment> sentimentStore = EmbeddingStore.initialize();

        // Add examples for each sentiment
        sentimentStore.addItem(new TextSegment("This is absolutely fantastic! I love it.")); // Positive
        sentimentStore.addItem(new TextSegment("I am very happy with this experience."));    // Positive
        sentimentStore.addItem(new TextSegment("It was okay, not great but not terrible.")); // Neutral
        sentimentStore.addItem(new TextSegment("Meh, it didn't really impress me."));        // Neutral
        sentimentStore.addItem(new TextSegment("This is terrible, I hate it."));            // Negative
        sentimentStore.addItem(new TextSegment("I am extremely disappointed."));            // Negative

        // List of user opinions to analyze
        List<String> opinions = List.of(
                "I love this product, it's amazing!",
                "It's alright, but I've seen better.",
                "I hate how poorly this works.",
                "This service is fantastic!",
                "Not bad, but not great either."
        );

        // Analyze each opinion
        for (String opinion : opinions) {
            // Generate embedding for the opinion
            double[] opinionEmbedding = embedder.embed(opinion);

            // Find the closest sentiment category
            List<EmbeddingMatch<TextSegment>> matches = sentimentStore.findRelevant(opinionEmbedding, 1);
            if (!matches.isEmpty()) {
                EmbeddingMatch<TextSegment> bestMatch = matches.get(0);
                System.out.println("Opinion: \"" + opinion + "\"");
                System.out.println("Classified as: \"" + bestMatch.item().getText() + "\"");
                System.out.println("Similarity: " + bestMatch.score());
                System.out.println();
            }
        }
    }
}
```
```bash
# Result
Opinion: "I love this product, it's amazing!"
Classified as: "This is absolutely fantastic! I love it."
Similarity: 0.5715483459053936

Opinion: "It's alright, but I've seen better."
Classified as: "It was okay, not great but not terrible."
Similarity: 0.5547731935863001

Opinion: "I hate how poorly this works."
Classified as: "This is terrible, I hate it."
Similarity: 0.5619213813474866

Opinion: "This service is fantastic!"
Classified as: "This is absolutely fantastic! I love it."
Similarity: 0.5332713832611117

Opinion: "Not bad, but not great either."
Classified as: "It was okay, not great but not terrible."
Similarity: 0.7377373708005006
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
