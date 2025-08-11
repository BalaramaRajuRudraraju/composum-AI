package com.composum.ai.backend.slingbase.servlet;

import java.io.IOException;
import java.util.List;
import java.util.stream.Collectors;

import javax.annotation.Nonnull;
import javax.servlet.Servlet;
import javax.servlet.ServletException;
import javax.servlet.http.HttpServletResponse;

import org.apache.commons.lang3.StringUtils;
import org.apache.sling.api.SlingHttpServletRequest;
import org.apache.sling.api.SlingHttpServletResponse;
import org.apache.sling.api.request.RequestPathInfo;
import org.apache.sling.api.servlets.HttpConstants;
import org.apache.sling.api.servlets.ServletResolverConstants;
import org.apache.sling.api.servlets.SlingSafeMethodsServlet;
import org.osgi.framework.Constants;
import org.osgi.service.component.annotations.Component;
import org.osgi.service.component.annotations.Reference;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import com.composum.ai.backend.base.service.GPTException;
import com.composum.ai.backend.base.service.chat.GPTChatCompletionService;
import com.composum.ai.backend.base.service.chat.GPTChatRequest;
import com.composum.ai.backend.base.service.chat.GPTConfiguration;
import com.composum.ai.backend.base.service.chat.GPTMessageRole;
import com.composum.ai.backend.base.service.chat.LangChain4JEmbeddingService;
import com.composum.ai.backend.slingbase.AIConfigurationService;
import com.google.gson.Gson;
import com.google.gson.GsonBuilder;

/**
 * Semantic Search Servlet that implements RAG (Retrieval Augmented Generation) approach.
 * 
 * This servlet follows the complete RAG pipeline:
 * 1. Converts user question to standalone question using LLM
 * 2. Generates embeddings for the standalone question
 * 3. Performs similarity search in Qdrant vector database
 * 4. Combines retrieved context with user question
 * 5. Generates friendly conversational response using LLM
 * 
 * <h2>Endpoint</h2>
 * <p>GET /bin/cpm/ai/semantic-search.json</p>
 * 
 * <h2>Parameters</h2>
 * <ul>
 *     <li><b>question</b> (query parameter): The user's question to search for.</li>
 *     <li><b>maxResults</b> (query parameter, optional): Maximum number of results to retrieve (default: 5).</li>
 * </ul>
 * 
 * <h2>Response</h2>
 * <p>Returns a JSON object containing the conversational answer and metadata:</p>
 * <pre>
 * {
 *     "answer": "Friendly conversational response...",
 *     "standaloneQuestion": "Reformulated standalone question",
 *     "sources": [
 *         {
 *             "pageUrl": "/content/site/page1",
 *             "pageTitle": "Page Title",
 *             "chunkText": "Relevant text chunk...",
 *             "similarityScore": 0.85
 *         }
 *     ],
 *     "responseTime": 1250
 * }
 * </pre>
 */
@Component(service = Servlet.class,
        property = {
                Constants.SERVICE_DESCRIPTION + "=Composum AI Semantic Search Servlet",
                ServletResolverConstants.SLING_SERVLET_PATHS + "=/bin/cpm/ai/semantic-search",
                ServletResolverConstants.SLING_SERVLET_METHODS + "=" + HttpConstants.METHOD_GET
        })
public class SemanticSearchServlet extends SlingSafeMethodsServlet {

    private static final Logger LOG = LoggerFactory.getLogger(SemanticSearchServlet.class);

    public static final String PARAM_QUESTION = "q";
    public static final String PARAM_MAX_RESULTS = "maxResults";
    public static final int DEFAULT_MAX_RESULTS = 5;

    protected final transient Gson gson = new GsonBuilder().disableHtmlEscaping().create();

    @Reference
    protected transient LangChain4JEmbeddingService embeddingService;

    @Reference
    protected transient GPTChatCompletionService chatService;

    @Reference
    protected transient AIConfigurationService aiConfigurationService;

    @Override
    protected void doGet(@Nonnull SlingHttpServletRequest request, @Nonnull SlingHttpServletResponse response) 
            throws ServletException, IOException {
        
        RequestPathInfo requestInfo = request.getRequestPathInfo();
        if (!"json".equals(requestInfo.getExtension())) {
            throw new ServletException("Only JSON extension is supported");
        }

        String question = request.getParameter(PARAM_QUESTION);
        if (StringUtils.isBlank(question)) {
            throw new ServletException("Missing required parameter: " + PARAM_QUESTION);
        }

        // Get max results parameter
        int maxResults = DEFAULT_MAX_RESULTS;
        String maxResultsParam = request.getParameter(PARAM_MAX_RESULTS);
        if (StringUtils.isNotBlank(maxResultsParam)) {
            try {
                maxResults = Integer.parseInt(maxResultsParam);
            } catch (NumberFormatException e) {
                throw new ServletException("Invalid maxResults parameter: " + maxResultsParam);
            }
        }

        LOG.info("Processing semantic search for question: {} (maxResults: {})", question, maxResults);

        long startTime = System.currentTimeMillis();
        
        try {
            SemanticSearchResult result = performSemanticSearch(question, maxResults, request);
            
            long responseTime = System.currentTimeMillis() - startTime;
            result.responseTime = responseTime;
            
            String json = gson.toJson(result);
            response.setContentType("application/json");
            response.setCharacterEncoding("UTF-8");
            response.getWriter().write(json);
            
            LOG.info("Completed semantic search in {}ms", responseTime);
            
        } catch (Exception e) {
            LOG.error("Error processing semantic search for question: {}", question, e);
            response.setStatus(HttpServletResponse.SC_INTERNAL_SERVER_ERROR);
            response.setContentType("application/json");
            response.setCharacterEncoding("UTF-8");
            response.getWriter().write("{\"error\": \"Failed to process semantic search: " + e.getMessage() + "\"}");
        }
    }

    /**
     * Performs the complete RAG pipeline for semantic search.
     */
    protected SemanticSearchResult performSemanticSearch(@Nonnull String userQuestion, int maxResults, 
                                                        @Nonnull SlingHttpServletRequest request) throws GPTException {
        
        GPTConfiguration config = aiConfigurationService.getGPTConfiguration(
            request.getResourceResolver(), request.getResource().getPath());

        // Step 1: Convert user question to standalone question
        String standaloneQuestion = convertToStandaloneQuestion(userQuestion, config);
        LOG.debug("Converted to standalone question: {}", standaloneQuestion);

        // Step 2: Generate embeddings for standalone question
        // (This is handled internally by the search method)

        // Step 3: Perform similarity search in Qdrant
        List<LangChain4JEmbeddingService.EmbeddingSearchResult> searchResults = 
            embeddingService.searchSimilarEmbeddings(standaloneQuestion, maxResults, config);
        
        LOG.info("Found {} relevant chunks for question", searchResults.size());

        // Step 4: Combine retrieved context
        String combinedContext = combineSearchResults(searchResults);

        // Step 5: Generate conversational response
        String answer = generateConversationalResponse(userQuestion, combinedContext, config, searchResults);

        // Build result object
        SemanticSearchResult result = new SemanticSearchResult();
        result.answer = answer;
        result.standaloneQuestion = standaloneQuestion;
        result.sources = searchResults.stream()
            .map(this::convertToSource)
            .collect(Collectors.toList());

        return result;
    }

    /**
     * Step 1: Converts user question to a standalone question using LLM.
     */
    protected String convertToStandaloneQuestion(@Nonnull String userQuestion, GPTConfiguration config) throws GPTException {
        String systemPrompt = "You are a helpful assistant that reformulates questions to be standalone and complete. " +
            "Take the user's question and rewrite it as a clear, standalone question that contains all necessary context. " +
            "If the question is already standalone, return it as-is. " +
            "Only return the reformulated question, nothing else.";

        String userPrompt = "Reformulate this question to be standalone and complete: " + userQuestion;

        GPTChatRequest chatRequest = new GPTChatRequest(config);
        chatRequest.addMessage(GPTMessageRole.SYSTEM, systemPrompt);
        chatRequest.addMessage(GPTMessageRole.USER, userPrompt);

        try {
            String response = chatService.getSingleChatCompletion(chatRequest);
            return StringUtils.isNotBlank(response) ? response.trim() : userQuestion;
        } catch (Exception e) {
            LOG.warn("Failed to convert to standalone question, using original: {}", e.getMessage());
            return userQuestion;
        }
    }

    /**
     * Step 4: Combines search results into a coherent context.
     */
    protected String combineSearchResults(@Nonnull List<LangChain4JEmbeddingService.EmbeddingSearchResult> results) {
        if (results.isEmpty()) {
            return "No relevant information found.";
        }

        StringBuilder context = new StringBuilder();
        context.append("Relevant information from the knowledge base:\n\n");

        for (int i = 0; i < results.size(); i++) {
            LangChain4JEmbeddingService.EmbeddingSearchResult result = results.get(i);
            context.append(String.format("Source %d (from %s - URL: %s):\n", 
                i + 1, result.getPageTitle(), result.getPageUrl()));
            context.append(result.getChunkText());
            context.append("\n\n");
        }

        return context.toString();
    }

    /**
     * Step 5: Generates a friendly conversational response using LLM.
     */
    protected String generateConversationalResponse(@Nonnull String userQuestion, @Nonnull String context, 
                                                   @Nonnull GPTConfiguration config, 
                                                   @Nonnull List<LangChain4JEmbeddingService.EmbeddingSearchResult> searchResults) throws GPTException {
        
        String systemPrompt = "You are a friendly and helpful AI assistant. Your task is to answer the user's question " +
            "based on the provided context information. Follow these guidelines:\n\n" +
            "1. Provide a clear, conversational, and friendly response\n" +
            "2. IMPORTANT: Select and use information from ONLY ONE source that best answers the question\n" +
            "3. Base your answer exclusively on the content from your selected source\n" +
            "4. If none of the sources contain enough information to answer the question, say " +
            "\"I don't know the answer to that based on the available information.\"\n" +
            "5. Be concise but comprehensive\n" +
            "6. Use a warm, conversational tone\n" +
            "7. Don't mention that you're using a knowledge base or context - just answer naturally\n" +
            "8. At the end of your response, include the phrase 'For more information, visit: [URL]' where [URL] is the " +
            "URL of the source you selected to answer the question (the URLs are provided in the context)\n" +
            "9. Only include the URL of the ONE source you used for your answer\n\n" +
            "Context information:\n" + context;

        String userPrompt = "Please answer this question using information from only ONE of the provided sources, " +
            "and include a link to that specific source page: " + userQuestion;

        // For now, use the same configuration - in production, you might want to configure 
        // this to point to your local LLM at localhost:8080
        GPTChatRequest chatRequest = new GPTChatRequest(config);
        chatRequest.addMessage(GPTMessageRole.SYSTEM, systemPrompt);
        chatRequest.addMessage(GPTMessageRole.USER, userPrompt);

        try {
            String response = chatService.getSingleChatCompletion(chatRequest);
            return StringUtils.isNotBlank(response) ? response.trim() : 
                "I don't know the answer to that based on the available information.";
        } catch (Exception e) {
            LOG.error("Failed to generate conversational response: {}", e.getMessage());
            return "I'm sorry, I'm having trouble processing your question right now. Please try again later.";
        }
    }

    // Note: For production use, configure the ChatService to point to localhost:8080
    // Currently using the default configuration for simplicity

    /**
     * Converts EmbeddingSearchResult to Source object for JSON response.
     */
    protected Source convertToSource(@Nonnull LangChain4JEmbeddingService.EmbeddingSearchResult result) {
        Source source = new Source();
        source.pageUrl = result.getPageUrl();
        source.pageTitle = result.getPageTitle();
        source.chunkText = result.getChunkText();
        source.similarityScore = result.getSimilarityScore();
        return source;
    }

    /**
     * Result object for semantic search response.
     */
    public static class SemanticSearchResult {
        public String answer;
        public String standaloneQuestion;
        public List<Source> sources;
        public long responseTime;
    }

    /**
     * Source information for search results.
     */
    public static class Source {
        public String pageUrl;
        public String pageTitle;
        public String chunkText;
        public double similarityScore;
    }
}
