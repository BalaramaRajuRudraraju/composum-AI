package com.composum.ai.backend.base.service.chat;

import java.util.List;

import javax.annotation.Nonnull;
import javax.annotation.Nullable;

import com.composum.ai.backend.base.service.GPTException;

/**
 * Service for generating embeddings using LangChain4J's OpenAI embedding model.
 * This service is independent of Sling and provides a clean interface for embedding generation
 * with Qdrant vector database storage and metadata support.
 */
public interface LangChain4JEmbeddingService {

    /**
     * Generates embeddings for multiple texts using LangChain4J's OpenAI embedding model.
     *
     * @param texts         the list of texts to generate embeddings for
     * @param configuration the GPT configuration containing API keys and model settings
     * @return a list of float arrays representing the embeddings
     * @throws GPTException if an error occurs during embedding generation
     */
    @Nonnull
    List<float[]> generateEmbeddings(@Nonnull List<String> texts, @Nullable GPTConfiguration configuration) throws GPTException;

    /**
     * Generates and stores embeddings for page texts with metadata in Qdrant vector database.
     *
     * @param pageTexts     a list of PageTextMetadata objects containing text content and associated metadata
     * @param configuration the GPT configuration containing API keys and model settings
     * @return a list of EmbeddingResult objects containing embeddings and storage IDs
     * @throws GPTException if an error occurs during embedding generation or storage
     */
    @Nonnull
    List<EmbeddingResult> generateAndStorePageEmbeddings(@Nonnull List<PageTextMetadata> pageTexts, @Nullable GPTConfiguration configuration) throws GPTException;

    /**
     * Generates an embedding for a single text using LangChain4J's OpenAI embedding model.
     *
     * @param text          the text to generate an embedding for
     * @param configuration the GPT configuration containing API keys and model settings
     * @return a float array representing the embedding
     * @throws GPTException if an error occurs during embedding generation
     */
    @Nonnull
    float[] generateSingleEmbedding(@Nonnull String text, @Nullable GPTConfiguration configuration) throws GPTException;

    /**
     * Searches for similar embeddings in the Qdrant vector database.
     *
     * @param queryText     the text to search for similar content
     * @param maxResults    maximum number of results to return
     * @param configuration the GPT configuration containing API keys and model settings
     * @return a list of search results with similarity scores
     * @throws GPTException if an error occurs during search
     */
    @Nonnull
    List<EmbeddingSearchResult> searchSimilarEmbeddings(@Nonnull String queryText, int maxResults, @Nullable GPTConfiguration configuration) throws GPTException;

    /**
     * Metadata associated with page text for embedding storage.
     */
    public static class PageTextMetadata {
        private final String pageUrl;
        private final String text;
        private final String pageTitle;
        private final String pageType;
        private final long lastModified;

        public PageTextMetadata(String pageUrl, String text, String pageTitle, String pageType, long lastModified) {
            this.pageUrl = pageUrl;
            this.text = text;
            this.pageTitle = pageTitle;
            this.pageType = pageType;
            this.lastModified = lastModified;
        }

        public String getPageUrl() { return pageUrl; }
        public String getText() { return text; }
        public String getPageTitle() { return pageTitle; }
        public String getPageType() { return pageType; }
        public long getLastModified() { return lastModified; }
    }

    /**
     * Result of embedding generation and storage operation.
     */
    public static class EmbeddingResult {
        private final String vectorId;
        private final float[] embedding;
        private final String chunkText;
        private final int chunkIndex;
        private final String pageUrl;

        public EmbeddingResult(String vectorId, float[] embedding, String chunkText, int chunkIndex, String pageUrl) {
            this.vectorId = vectorId;
            this.embedding = embedding;
            this.chunkText = chunkText;
            this.chunkIndex = chunkIndex;
            this.pageUrl = pageUrl;
        }

        public String getVectorId() { return vectorId; }
        public float[] getEmbedding() { return embedding; }
        public String getChunkText() { return chunkText; }
        public int getChunkIndex() { return chunkIndex; }
        public String getPageUrl() { return pageUrl; }
    }

    /**
     * Result of similarity search operation.
     */
    public static class EmbeddingSearchResult {
        private final String chunkText;
        private final String pageUrl;
        private final String pageTitle;
        private final double similarityScore;
        private final int chunkIndex;

        public EmbeddingSearchResult(String chunkText, String pageUrl, String pageTitle, double similarityScore, int chunkIndex) {
            this.chunkText = chunkText;
            this.pageUrl = pageUrl;
            this.pageTitle = pageTitle;
            this.similarityScore = similarityScore;
            this.chunkIndex = chunkIndex;
        }

        public String getChunkText() { return chunkText; }
        public String getPageUrl() { return pageUrl; }
        public String getPageTitle() { return pageTitle; }
        public double getSimilarityScore() { return similarityScore; }
        public int getChunkIndex() { return chunkIndex; }
    }
}
