package com.composum.ai.backend.base.service.chat.impl;

import java.util.List;

import javax.annotation.Nonnull;
import javax.annotation.Nullable;

import com.composum.ai.backend.base.service.GPTException;
import com.composum.ai.backend.base.service.chat.GPTConfiguration;

/**
 * Service for generating embeddings using LangChain4J's OpenAI embedding model.
 * This service is independent of Sling and provides a clean interface for embedding generation.
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
     * Generates an embedding for a single text using LangChain4J's OpenAI embedding model.
     *
     * @param text          the text to generate an embedding for
     * @param configuration the GPT configuration containing API keys and model settings
     * @return a float array representing the embedding
     * @throws GPTException if an error occurs during embedding generation
     */
    @Nonnull
    float[] generateSingleEmbedding(@Nonnull String text, @Nullable GPTConfiguration configuration) throws GPTException;
}
