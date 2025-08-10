package com.composum.ai.backend.base.service.chat.impl;

import java.util.ArrayList;
import java.util.Collections;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
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
import dev.langchain4j.store.embedding.EmbeddingMatch;
import dev.langchain4j.store.embedding.EmbeddingSearchRequest;
import dev.langchain4j.store.embedding.EmbeddingStore;
import dev.langchain4j.store.embedding.qdrant.QdrantEmbeddingStore;

/**
 * Service for generating embeddings using LangChain4J's OpenAI embedding model.
 * This service is independent of Sling and can be used in the backend/base module.
 * This implementation uses LangChain4J's OpenAiEmbeddingModel to generate embeddings
 * for text content with automatic text chunking, then stores the embeddings in 
 * Qdrant vector database for semantic search and similarity operations.
 */
@Component(service = LangChain4JEmbeddingService.class)
public class LangChain4JEmbeddingServiceImpl implements LangChain4JEmbeddingService {

    private static final Logger LOG = LoggerFactory.getLogger(LangChain4JEmbeddingServiceImpl.class);

    private static final String DEFAULT_EMBEDDING_MODEL = "text-embedding-nomic-embed-text-v1.5@q5_k_m";
    private static final String QDRANT_HOST = "192.168.1.131";
    private static final int QDRANT_PORT = 6333;
    private static final int QDRANT_GRPC_PORT = 6334;
    private static final String QDRANT_COLLECTION_NAME = "composum-ai-embeddings";
    
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

    @Override
    @Nonnull
    public List<LangChain4JEmbeddingService.EmbeddingResult> generateAndStorePageEmbeddings(@Nonnull List<LangChain4JEmbeddingService.PageTextMetadata> pageTexts, @Nullable GPTConfiguration configuration) throws GPTException {
        if (pageTexts.isEmpty()) {
            return Collections.emptyList();
        }

        LOG.info("Generating and storing embeddings for {} pages using LangChain4J with Qdrant", pageTexts.size());

        try {
            // Initialize embedding model and store if needed
            ensureInitialized(configuration);
            
            List<LangChain4JEmbeddingService.EmbeddingResult> allResults = new ArrayList<>();
            
            for (LangChain4JEmbeddingService.PageTextMetadata pageText : pageTexts) {
                // Process each page text with chunking and metadata storage
                List<LangChain4JEmbeddingService.EmbeddingResult> pageResults = processPageTextWithMetadata(pageText, configuration);
                allResults.addAll(pageResults);
            }

            LOG.info("Successfully generated and stored {} embeddings from {} pages", allResults.size(), pageTexts.size());
            return allResults;
            
        } catch (Exception e) {
            throw new GPTException("Failed to generate and store page embeddings using LangChain4J", e);
        }
    }

    @Override
    @Nonnull
    public List<LangChain4JEmbeddingService.EmbeddingSearchResult> searchSimilarEmbeddings(@Nonnull String queryText, int maxResults, @Nullable GPTConfiguration configuration) throws GPTException {
        LOG.debug("Searching for similar embeddings for query text using LangChain4J");

        try {
            // Initialize embedding model and store if needed
            ensureInitialized(configuration);
            
            // Generate embedding for the query text
            Embedding queryEmbedding = embeddingModel.embed(queryText).content();
            
            LOG.info("Searching for {} similar embeddings for query (embedding dimension: {})", maxResults, queryEmbedding.vector().length);
            
            // Search for similar embeddings in the vector store using EmbeddingSearchRequest
            EmbeddingSearchRequest searchRequest = EmbeddingSearchRequest.builder()
                    .queryEmbedding(queryEmbedding)
                    .maxResults(maxResults)
                    .minScore(0.75) // Minimum score threshold for similarity
                    .build();
            
            List<EmbeddingMatch<TextSegment>> matches = embeddingStore.search(searchRequest).matches();
            
            // Convert matches to EmbeddingSearchResult objects
            List<LangChain4JEmbeddingService.EmbeddingSearchResult> results = new ArrayList<>();
            
            for (EmbeddingMatch<TextSegment> match : matches) {
                TextSegment segment = match.embedded();
                double score = match.score();
                
                // Parse metadata from the enhanced text format
                String text = segment.text();
                String[] parts = text.split("\n\n\\[METADATA:");
                String actualText = parts[0]; // Original text before metadata
                
                // Extract metadata if available
                String pageUrl = "unknown";
                String pageTitle = "unknown";
                int chunkIndex = 0;
                
                if (parts.length > 1) {
                    String metadataPart = parts[1];
                    pageUrl = extractMetadataValue(metadataPart, "URL=");
                    pageTitle = extractMetadataValue(metadataPart, "Title=");
                    String chunkStr = extractMetadataValue(metadataPart, "Chunk=");
                    try {
                        chunkIndex = Integer.parseInt(chunkStr.replaceAll("\\].*", ""));
                    } catch (NumberFormatException e) {
                        // Use default chunk index
                    }
                }
                
                LangChain4JEmbeddingService.EmbeddingSearchResult result = 
                    new LangChain4JEmbeddingService.EmbeddingSearchResult(actualText, pageUrl, pageTitle, score, chunkIndex);
                results.add(result);
            }
            
            LOG.debug("Found {} similar embeddings with scores", results.size());
            return results;
            
        } catch (Exception e) {
            LOG.error("Error searching for similar embeddings", e);
            throw new GPTException("Failed to search for similar embeddings", e);
        }
    }

    /**
     * Extracts metadata values from the formatted metadata string.
     */
    @Nonnull
    protected String extractMetadataValue(@Nonnull String metadataString, @Nonnull String key) {
        try {
            int startIndex = metadataString.indexOf(key);
            if (startIndex == -1) {
                return "unknown";
            }
            startIndex += key.length();
            
            int endIndex = metadataString.indexOf(",", startIndex);
            if (endIndex == -1) {
                endIndex = metadataString.indexOf("]", startIndex);
            }
            if (endIndex == -1) {
                endIndex = metadataString.length();
            }
            
            return metadataString.substring(startIndex, endIndex).trim();
        } catch (Exception e) {
            LOG.warn("Could not extract metadata value for key {}: {}", key, e.getMessage());
            return "unknown";
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
     * Processes a single page text with metadata, chunking, and stores embeddings in Qdrant with comprehensive metadata.
     */
    @Nonnull
    protected List<LangChain4JEmbeddingService.EmbeddingResult> processPageTextWithMetadata(@Nonnull LangChain4JEmbeddingService.PageTextMetadata pageText, @Nullable GPTConfiguration configuration) throws GPTException {
        try {
            // Create a document from the text
            Document document = Document.from(pageText.getText());
            
            // Split the document into chunks
            List<TextSegment> textSegments = documentSplitter.split(document);
            
            LOG.debug("Split page {} into {} chunks", pageText.getPageUrl(), textSegments.size());
            
            List<LangChain4JEmbeddingService.EmbeddingResult> results = new ArrayList<>();
            
            // Generate embeddings for each chunk with metadata
            for (int i = 0; i < textSegments.size(); i++) {
                TextSegment segment = textSegments.get(i);
                String chunkText = segment.text();
                
                // Generate embedding for this chunk
                Embedding embedding = embeddingModel.embed(segment).content();
                float[] embeddingVector = embedding.vector();
                
                // Create metadata for this chunk and log it
                Map<String, Object> metadata = createChunkMetadata(pageText, i, chunkText);
                LOG.debug("Created metadata for chunk {}: {}", i, metadata.keySet());
                
                // Create TextSegment with metadata as a formatted text (for compatibility)
                String enhancedText = String.format("%s\n\n[METADATA: URL=%s, Title=%s, Type=%s, Chunk=%d]", 
                    chunkText, pageText.getPageUrl(), pageText.getPageTitle(), pageText.getPageType(), i);
                TextSegment segmentWithMetadata = TextSegment.from(enhancedText);
                
                // Store in vector database
                String vectorId = UUID.randomUUID().toString();
                embeddingStore.add(embedding, segmentWithMetadata);
                
                // Create result object
                LangChain4JEmbeddingService.EmbeddingResult result = new LangChain4JEmbeddingService.EmbeddingResult(
                    vectorId, embeddingVector, chunkText, i, pageText.getPageUrl()
                );
                results.add(result);
                
                LOG.debug("Stored embedding for chunk {} of page {} in Qdrant with ID {}", 
                    i, pageText.getPageUrl(), vectorId);
            }
            
            return results;
            
        } catch (Exception e) {
            LOG.error("Error processing page text with metadata for page: {}", pageText.getPageUrl(), e);
            throw new GPTException("Failed to process page text with metadata: " + pageText.getPageUrl(), e);
        }
    }

    /**
     * Creates comprehensive metadata for a text chunk.
     */
    @Nonnull
    protected Map<String, Object> createChunkMetadata(@Nonnull LangChain4JEmbeddingService.PageTextMetadata pageText, int chunkIndex, @Nonnull String chunkText) {
        Map<String, Object> metadata = new HashMap<>();
        
        // Page metadata
        metadata.put("pageUrl", pageText.getPageUrl());
        metadata.put("pageTitle", pageText.getPageTitle());
        metadata.put("pageType", pageText.getPageType());
        metadata.put("lastModified", pageText.getLastModified());
        
        // Chunk metadata
        metadata.put("chunkIndex", chunkIndex);
        metadata.put("chunkLength", chunkText.length());
        metadata.put("chunkHash", Integer.toString(chunkText.hashCode()));
        
        // Processing metadata
        metadata.put("processedAt", System.currentTimeMillis());
        metadata.put("embeddingModel", getModelName(null)); // Use default model name
        
        return metadata;
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
            LOG.info("Creating Qdrant embedding store at {}:{} with collection '{}'", QDRANT_HOST, QDRANT_PORT, QDRANT_COLLECTION_NAME);
            
            // Create Qdrant embedding store
            return QdrantEmbeddingStore.builder()
                    .host(QDRANT_HOST)
                    .port(QDRANT_GRPC_PORT)
                    .collectionName(QDRANT_COLLECTION_NAME)
                    .build();
            
        } catch (Exception e) {
            LOG.error("Failed to create Qdrant embedding store, falling back to in-memory store", e);
            LOG.warn("Using in-memory embedding store. Ensure Qdrant is running at {}:{}", QDRANT_HOST, QDRANT_PORT);
            
            // Fallback to in-memory store if Qdrant is not available
            return new dev.langchain4j.store.embedding.inmemory.InMemoryEmbeddingStore<>();
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
