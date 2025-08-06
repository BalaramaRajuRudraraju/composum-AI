package com.composum.ai.backend.slingbase;

import java.io.IOException;
import java.util.ArrayList;
import java.util.List;
import java.util.Map;
import java.util.TreeMap;

import javax.annotation.Nonnull;
import javax.servlet.Servlet;
import javax.servlet.ServletException;

import org.apache.commons.lang3.StringUtils;
import org.apache.sling.api.SlingHttpServletRequest;
import org.apache.sling.api.SlingHttpServletResponse;
import org.apache.sling.api.request.RequestPathInfo;
import org.apache.sling.api.resource.Resource;
import org.apache.sling.api.servlets.HttpConstants;
import org.apache.sling.api.servlets.ServletResolverConstants;
import org.apache.sling.api.servlets.SlingSafeMethodsServlet;
import org.osgi.framework.Constants;
import org.osgi.service.component.annotations.Component;
import org.osgi.service.component.annotations.Reference;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import com.composum.ai.backend.base.service.GPTException;
import com.composum.ai.backend.base.service.chat.GPTConfiguration;
import com.composum.ai.backend.base.service.chat.GPTEmbeddingService;
import com.composum.ai.backend.base.service.chat.LangChain4JEmbeddingService;
import com.google.gson.Gson;
import com.google.gson.GsonBuilder;

/**
 * Servlet that extracts text from a page and its child pages, then converts them to embeddings.
 * 
 * This servlet takes a path as input, recursively traverses the page and its child pages,
 * extracts markdown text using the ApproximateMarkdownService, stores them in temporary maps,
 * and converts the texts to embeddings using the GPTEmbeddingService.
 * 
 * <h2>Endpoint</h2>
 * <p>GET /bin/cpm/ai/pageembedding</p>
 * 
 * <h2>Parameters</h2>
 * <ul>
 *     <li><b>path</b> (suffix): The path of the page to process.</li>
 *     <li><b>maxDepth</b> (query parameter, optional): Maximum depth for child page traversal (default: 3).</li>
 * </ul>
 * 
 * <h2>Response</h2>
 * <p>Returns a JSON object containing the page paths, their extracted text, and embeddings:</p>
 * <pre>
 * {
 *     "pages": [
 *         {
 *             "path": "/content/site/page1",
 *             "text": "Extracted markdown text...",
 *             "embedding": [0.1, 0.2, ...]
 *         }
 *     ]
 * }
 * </pre>
 */
@Component(service = Servlet.class,
        property = {
                Constants.SERVICE_DESCRIPTION + "=Composum AI Page Embedding Servlet",
                ServletResolverConstants.SLING_SERVLET_PATHS + "=/bin/cpm/ai/pageembedding",
                ServletResolverConstants.SLING_SERVLET_METHODS + "=" + HttpConstants.METHOD_GET
        })
public class PageEmbeddingServlet extends SlingSafeMethodsServlet {

    private static final Logger LOG = LoggerFactory.getLogger(PageEmbeddingServlet.class);

    public static final String PARAM_MAX_DEPTH = "maxDepth";
    public static final int DEFAULT_MAX_DEPTH = 3;

    protected final transient Gson gson = new GsonBuilder().disableHtmlEscaping().create();

    @Reference
    protected transient ApproximateMarkdownService markdownService;

    @Reference
    protected transient LangChain4JEmbeddingService langChain4JEmbeddingService;

    // Fallback to existing GPTEmbeddingService if LangChain4J is not available
    @Reference
    protected transient GPTEmbeddingService embeddingService;

    @Reference
    protected transient AIConfigurationService aiConfigurationService;

    @Override
    protected void doGet(@Nonnull SlingHttpServletRequest request, @Nonnull SlingHttpServletResponse response) 
            throws ServletException, IOException {
        
        RequestPathInfo requestInfo = request.getRequestPathInfo();
        if (!"json".equals(requestInfo.getExtension())) {
            throw new ServletException("Only JSON extension is supported");
        }

        Resource rootResource = requestInfo.getSuffixResource();
        if (rootResource == null) {
            throw new ServletException("Missing path in suffix - specify the page path to process");
        }

        // Get max depth parameter
        int maxDepth = DEFAULT_MAX_DEPTH;
        String maxDepthParam = request.getParameter(PARAM_MAX_DEPTH);
        if (StringUtils.isNotBlank(maxDepthParam)) {
            try {
                maxDepth = Integer.parseInt(maxDepthParam);
            } catch (NumberFormatException e) {
                throw new ServletException("Invalid maxDepth parameter: " + maxDepthParam);
            }
        }

        LOG.info("Processing page embeddings for path {} with maxDepth {}", rootResource.getPath(), maxDepth);

        try {
            PageEmbeddingResult result = processPageEmbeddings(rootResource, request, response, maxDepth);
            
            String json = gson.toJson(result);
            response.setContentType("application/json");
            response.setCharacterEncoding("UTF-8");
            response.getWriter().write(json);
            
        } catch (Exception e) {
            throw new ServletException("Error processing page embeddings: " + e.getMessage(), e);
        }
    }

    /**
     * Processes page embeddings for the given root resource and its children.
     */
    protected PageEmbeddingResult processPageEmbeddings(@Nonnull Resource rootResource, 
                                                       @Nonnull SlingHttpServletRequest request,
                                                       @Nonnull SlingHttpServletResponse response,
                                                       int maxDepth) throws GPTException {
        
        // Collect all pages and their content (similar to RAGServiceImpl.ragAnswer lines 193-197)
        List<Resource> pages = collectPages(rootResource, maxDepth);
        Map<String, String> textToPath = new TreeMap<>();
        Map<String, Resource> textToResource = new TreeMap<>();
        
        // Extract text from each page using ApproximateMarkdownService
        for (Resource page : pages) {
            String markdown = markdownService.approximateMarkdown(page, request, response);
            if (StringUtils.isNotBlank(markdown)) {
                textToPath.put(markdown, page.getPath());
                textToResource.put(markdown, page);
            }
        }

        // Get GPT configuration for embeddings
        GPTConfiguration config = aiConfigurationService.getGPTConfiguration(
            rootResource.getResourceResolver(), rootResource.getPath());

        // Convert texts to embeddings using LangChain4J with chunking and Qdrant storage
        List<String> texts = new ArrayList<>(textToPath.keySet());
        List<float[]> embeddings;
        
        try {
            // Try to use LangChain4J service first (with chunking and Qdrant storage)
            embeddings = langChain4JEmbeddingService.generateEmbeddings(texts, config);
            LOG.info("Generated {} embeddings using LangChain4J with chunking", embeddings.size());
        } catch (Exception e) {
            // Fallback to existing GPTEmbeddingService if LangChain4J fails
            LOG.warn("LangChain4J service failed, falling back to GPTEmbeddingService: {}", e.getMessage());
            embeddings = embeddingService.getEmbeddings(texts, config, null);
            LOG.info("Generated {} embeddings using fallback GPTEmbeddingService", embeddings.size());
        }

        // Build the result
        PageEmbeddingResult result = new PageEmbeddingResult();
        result.pages = new ArrayList<>();
        
        for (int i = 0; i < texts.size(); i++) {
            String text = texts.get(i);
            String path = textToPath.get(text);
            float[] embedding = embeddings.get(i);
            
            PageInfo pageInfo = new PageInfo();
            pageInfo.path = path;
            pageInfo.text = text;
            pageInfo.embedding = embedding;
            
            result.pages.add(pageInfo);
        }

        LOG.info("Processed {} pages for embeddings", result.pages.size());
        return result;
    }

    /**
     * Collects all pages starting from the root resource up to the specified depth.
     * This method recursively traverses the resource tree to find all pages.
     */
    protected List<Resource> collectPages(@Nonnull Resource root, int maxDepth) {
        if (maxDepth < 0) {
            return new ArrayList<>();
        }
        
        List<Resource> pages = new ArrayList<>();
        
        // If this resource is a page content node, add it
        if (isPageContent(root)) {
            pages.add(root);
        } else if (isPage(root)) {
            // If it's a page, get its jcr:content
            Resource content = root.getChild("jcr:content");
            if (content != null) {
                pages.add(content);
            }
        }
        
        // Recursively process children
        if (maxDepth > 0) {
            for (Resource child : root.getChildren()) {
                pages.addAll(collectPages(child, maxDepth - 1));
            }
        }
        
        return pages;
    }

    /**
     * Checks if the resource is a page (cq:Page).
     */
    protected boolean isPage(@Nonnull Resource resource) {
        return AIResourceUtil.isOfNodeType(resource, "cq:Page");
    }

    /**
     * Checks if the resource is page content (cq:PageContent).
     */
    protected boolean isPageContent(@Nonnull Resource resource) {
        return AIResourceUtil.isOfNodeType(resource, "cq:PageContent") ||
               resource.getName().equals("jcr:content");
    }

    /**
     * Result object containing page embeddings.
     */
    public static class PageEmbeddingResult {
        public List<PageInfo> pages;
    }

    /**
     * Information about a single page and its embedding.
     */
    public static class PageInfo {
        public static String path;
        public String text;
        public float[] embedding;
    }
}
