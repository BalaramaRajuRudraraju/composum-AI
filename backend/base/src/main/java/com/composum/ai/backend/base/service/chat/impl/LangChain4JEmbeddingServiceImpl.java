package com.composum.ai.backend.base.service.chat.impl;

import java.util.ArrayList;
import java.util.Collections;
import java.util.List;
import java.util.UUID;

import javax.annotation.Nonnull;
import javax.annotation.Nullable;

import com.composum.ai.backend.base.service.chat.LangChain4JEmbeddingService;
import org.osgi.service.component.annotations.Activate;
import org.osgi.service.component.annotations.Component;
import org.osgi.service.component.annotations.Deactivate;
import org.osgi.service.component.annotations.Reference;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import com.composum.ai.backend.base.service.GPTException;
import com.composum.ai.backend.base.service.chat.GPTBackendConfiguration;
import com.composum.ai.backend.base.service.chat.GPTBackendsService;
import com.composum.ai.backend.base.service.chat.GPTConfiguration;

import dev.langchain4j.data.document.Document;
import dev.langchain4j.data.document.DocumentSplitter;
import dev.langchain4j.data.document.splitter.DocumentSplitters;
import dev.langchain4j.data.embedding.Embedding;
import dev.langchain4j.data.segment.TextSegment;
import dev.langchain4j.model.embedding.EmbeddingModel;
import dev.langchain4j.model.openai.OpenAiEmbeddingModel;
import dev.langchain4j.store.embedding.EmbeddingStore;

/**
 * Service for generating embeddings using LangChain4J's OpenAI embedding model.
 * This service is independent of Sling and can be used in the backend/base module.
 * 
 * This implementation uses LangChain4J's OpenAiEmbeddingModel to generate embeddings
 * for text content with automatic text chunking, then stores the embeddings in 
 * Qdrant vector database for semantic search and similarity operations.
 */
@Component(service = LangChain4JEmbeddingService.class)
public class LangChain4JEmbeddingServiceImpl implements LangChain4JEmbeddingService {

    private static final Logger LOG = LoggerFactory.getLogger(LangChain4JEmbeddingServiceImpl.class);

    private static final String DEFAULT_EMBEDDING_MODEL = "text-embedding-nomic-embed-text-v1.5@q5_k_m";
    private static final String QDRANT_HOST = "localhost";
    private static final int QDRANT_PORT = 6333;
    // TODO: Use when Qdrant integration is available
    // private static final String QDRANT_COLLECTION_NAME = "composum-ai-embeddings";
    
    // Text chunking parameters
    private static final int MAX_CHUNK_SIZE = 1000; // Maximum characters per chunk
    private static final int CHUNK_OVERLAP = 200;   // Overlap between chunks
    
    @Reference
    protected GPTBackendsService backendsService;
    
    private EmbeddingStore<TextSegment> embeddingStore;
    private EmbeddingModel embeddingModel;
    private DocumentSplitter documentSplitter;

    @Activate
    protected void activate() {
        LOG.info("Activating LangChain4J Embedding Service");
        
        // Initialize document splitter for chunking
        documentSplitter = DocumentSplitters.recursive(MAX_CHUNK_SIZE, CHUNK_OVERLAP);
        
        LOG.info("LangChain4J Embedding Service activated successfully");
    }

    @Deactivate 
    protected void deactivate() {
        LOG.info("Deactivating LangChain4J Embedding Service");
        if (embeddingStore != null) {
            // Clean up any resources if needed
            embeddingStore = null;
        }
        if (embeddingModel != null) {
            embeddingModel = null;
        }
    }

    @Override
    @Nonnull
    public List<float[]> generateEmbeddings(@Nonnull List<String> texts, @Nullable GPTConfiguration configuration) throws GPTException {
        if (texts.isEmpty()) {
            return Collections.emptyList();
        }

        LOG.debug("Generating embeddings for {} texts using LangChain4J with chunking", texts.size());

        try {
            // Initialize embedding model and store if needed
            ensureInitialized(configuration);
            
            List<float[]> allEmbeddings = new ArrayList<>();
            
            for (String text : texts) {
                // Process each text with chunking
                List<float[]> textEmbeddings = processTextWithChunking(text, configuration);
                allEmbeddings.addAll(textEmbeddings);
            }

            LOG.debug("Successfully generated {} embeddings from {} texts", allEmbeddings.size(), texts.size());
            return allEmbeddings;
            
        } catch (Exception e) {
            throw new GPTException("Failed to generate embeddings using LangChain4J", e);
        }
    }

    @Override
    @Nonnull
    public float[] generateSingleEmbedding(@Nonnull String text, @Nullable GPTConfiguration configuration) throws GPTException {
        LOG.debug("Generating single embedding for text using LangChain4J");

        try {
            // Initialize embedding model if needed
            ensureInitialized(configuration);
            
            // For single embedding, we don't chunk - use the text as is
            Embedding embedding = embeddingModel.embed(TextSegment.from(text)).content();
            
            float[] result = embedding.vector();
            
            LOG.debug("Successfully generated single embedding with dimension {}", result.length);
            return result;
            
        } catch (Exception e) {
            throw new GPTException("Failed to generate single embedding using LangChain4J", e);
        }
    }

    /**
     * Processes a single text with chunking and generates embeddings for each chunk.
     * Also stores the embeddings in Qdrant vector database.
     */
    @Nonnull
    protected List<float[]> processTextWithChunking(@Nonnull String text, @Nullable GPTConfiguration configuration) throws GPTException {
        try {
            // Create a document from the text
            Document document = Document.from(text);
            
            // Split the document into chunks
            List<TextSegment> textSegments = documentSplitter.split(document);
            
            LOG.debug("Split text into {} chunks", textSegments.size());
            
            List<float[]> embeddings = new ArrayList<>();
            
            // Generate embeddings for each chunk
            for (TextSegment segment : textSegments) {
                Embedding embedding = embeddingModel.embed(segment).content();
                float[] embeddingVector = embedding.vector();
                embeddings.add(embeddingVector);
                
                // Store in Qdrant vector database with unique ID
                String segmentId = UUID.randomUUID().toString();
                embeddingStore.add(embedding, segment);
                
                LOG.debug("Stored embedding for chunk {} in Qdrant", segmentId);
            }
            
            return embeddings;
            
        } catch (Exception e) {
            throw new GPTException("Failed to process text with chunking", e);
        }
    }

    /**
     * Ensures that the embedding model and store are initialized.
     */
    protected void ensureInitialized(@Nullable GPTConfiguration configuration) throws GPTException {
        if (embeddingModel == null) {
            embeddingModel = createEmbeddingModel(configuration);
        }
        if (embeddingStore == null) {
            embeddingStore = createEmbeddingStore();
        }
    }

    /**
     * Creates an OpenAI embedding model using LangChain4J with the provided configuration.
     */
    @Nonnull
    protected EmbeddingModel createEmbeddingModel(@Nullable GPTConfiguration configuration) throws GPTException {
        // Get model name from configuration or use default
        String modelName = getModelName(configuration);
        
        // Get backend configuration for the model
        GPTBackendConfiguration backendConfig = backendsService.getConfigurationForModel(modelName);
        if (backendConfig == null) {
            throw new GPTException("No backend configuration found for model: " + modelName);
        }

        // Get API key from backend configuration headers
        String apiKey = getApiKeyFromBackendConfig(backendConfig);
        if (apiKey == null || apiKey.trim().isEmpty()) {
            throw new GPTException("OpenAI API key is not configured in backend");
        }

        // Create OpenAI embedding model using LangChain4J
        try {
            String actualModelName = backendsService.getModelNameInBackend(modelName);
            
            // Use LangChain4J builder pattern for version 0.35.0
            return OpenAiEmbeddingModel.builder()
                    .apiKey(apiKey)
                    .baseUrl(backendConfig.apiEndpoint())
                    .modelName(actualModelName)
                    .build();
        } catch (Exception e) {
            throw new GPTException("Failed to create OpenAI embedding model", e);
        }
    }

    /**
     * Creates a Qdrant embedding store for vector storage.
     */
    @Nonnull
    protected EmbeddingStore<TextSegment> createEmbeddingStore() throws GPTException {
        try {
            // For now, use in-memory embedding store since Qdrant integration 
            // may need additional configuration. In production, this should be 
            // replaced with actual Qdrant store.
            LOG.warn("Using in-memory embedding store. In production, configure Qdrant at {}:{}", QDRANT_HOST, QDRANT_PORT);
            
            // TODO: Replace with QdrantEmbeddingStore when available:
            // return QdrantEmbeddingStore.builder()
            //         .host(QDRANT_HOST)
            //         .port(QDRANT_PORT)
            //         .collectionName(QDRANT_COLLECTION_NAME)
            //         .build();
            
            // For now, use a simple in-memory store
            return new dev.langchain4j.store.embedding.inmemory.InMemoryEmbeddingStore<>();
            
        } catch (Exception e) {
            LOG.error("Failed to create embedding store", e);
            throw new GPTException("Failed to create embedding store", e);
        }
    }

    /**
     * Extracts the API key from backend configuration headers.
     * Looks for Authorization header containing Bearer token.
     */
    @Nullable
    protected String getApiKeyFromBackendConfig(@Nonnull GPTBackendConfiguration backendConfig) {
        // Check header 1
        if ("Authorization".equalsIgnoreCase(backendConfig.additionalHeader1Key()) && 
            backendConfig.additionalHeader1Value() != null) {
            String headerValue = backendConfig.additionalHeader1Value().trim();
            if (headerValue.startsWith("Bearer ")) {
                return headerValue.substring(7); // Remove "Bearer " prefix
            }
        }
        
        // Check header 2
        if ("Authorization".equalsIgnoreCase(backendConfig.additionalHeader2Key()) && 
            backendConfig.additionalHeader2Value() != null) {
            String headerValue = backendConfig.additionalHeader2Value().trim();
            if (headerValue.startsWith("Bearer ")) {
                return headerValue.substring(7); // Remove "Bearer " prefix
            }
        }
        
        // Check header 3
        if ("Authorization".equalsIgnoreCase(backendConfig.additionalHeader3Key()) && 
            backendConfig.additionalHeader3Value() != null) {
            String headerValue = backendConfig.additionalHeader3Value().trim();
            if (headerValue.startsWith("Bearer ")) {
                return headerValue.substring(7); // Remove "Bearer " prefix
            }
        }
        
        return null;
    }

    /**
     * Gets the model name from the configuration.
     */
    @Nonnull
    protected String getModelName(@Nullable GPTConfiguration configuration) {
        if (configuration != null && configuration.getModel() != null) {
            return configuration.getModel();
        }
        return DEFAULT_EMBEDDING_MODEL;
    }

    /*
    TODO: When Java 17+ is available, implement this method:
    
    protected EmbeddingModel createEmbeddingModel(@Nullable GPTConfiguration configuration) throws GPTException {
        // Get model name from configuration or use default
        String modelName = getModelName(configuration);
        
        // Get backend configuration for the model
        GPTBackendConfiguration backendConfig = backendsService.getConfigurationForModel(modelName);
        if (backendConfig == null) {
            throw new GPTException("No backend configuration found for model: " + modelName);
        }

        // Get API key from backend configuration headers
        String apiKey = getApiKeyFromBackendConfig(backendConfig);
        if (apiKey == null || apiKey.trim().isEmpty()) {
            throw new GPTException("OpenAI API key is not configured in backend");
        }

        // Create OpenAI embedding model using LangChain4J
        try {
            String actualModelName = backendsService.getModelNameInBackend(modelName);
            
            // Use LangChain4J builder pattern
            return OpenAiEmbeddingModel.builder()
                    .apiKey(apiKey)
                    .modelName(actualModelName)
                    .build();
        } catch (Exception e) {
            LOG.error("Failed to create OpenAI embedding model", e);
            throw new GPTException("Failed to create OpenAI embedding model", e);
        }
    }
    */
}
