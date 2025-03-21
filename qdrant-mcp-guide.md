# Comprehensive Guide to Extending Qdrant MCP Server with Custom Tools

This guide provides a detailed roadmap for enhancing Qdrant MCP server with custom tools, making it a powerful vector database platform with advanced capabilities.

## Latest Updates - March 21, 2025

In the latest implementation session, we've added several new tool categories to enhance the Qdrant MCP server's capabilities:

### 1. Multilingual Support Tools
- **detect_language**: Identifies the language of text content with confidence scores and alternatives
- **translate_text**: Translates content between languages using various providers (Google, DeepL, Microsoft, etc.)
- **multilingual_search**: Performs semantic search across content in multiple languages with optional result translation

### 2. Advanced Embedding and Search
- **fusion_search**: Combines results from multiple embedding models for more robust search performance using methods like Reciprocal Rank Fusion

### 3. Web Crawling Integration (in progress)
- **crawl_url**: Processes single web pages and extracts content with metadata
- Additional tools planned: batch_crawl, recursive_crawl, sitemap_extract, and content_processor

These new tools further extend the power of the Qdrant MCP server, enabling more sophisticated multilingual applications, improved search accuracy through model fusion, and web content integration.

## Table of Contents

- [Introduction](#introduction)
- [Core Search and Query Tools](#core-search-and-query-tools)
- [Collection Management Tools](#collection-management-tools)
- [Data Processing Tools](#data-processing-tools)
- [Advanced Query Tools](#advanced-query-tools)
- [Analytics and Management Tools](#analytics-and-management-tools)
- [Document Processing Tools](#document-processing-tools)
- [Multilingual Tools](#multilingual-tools)
- [Advanced Search Tools](#advanced-search-tools)
- [Web Crawling Tools](#web-crawling-tools)
- [Integration Tools](#integration-tools)
- [Implementation in MCP Config](#implementation-in-mcp-config)
- [Utility Functions](#utility-functions)
- [Setup and Deployment](#setup-and-deployment)

## Introduction

Qdrant is a powerful vector similarity search engine that can be significantly enhanced through custom tools in the MCP server. This guide outlines the implementation of various tools that extend Qdrant's capabilities for natural language processing, advanced queries, and data management.

## Core Search and Query Tools

### 1. Natural Language Query Tool

```typescript
// tools/nlq.ts
export async function naturalLanguageQuery(params: {
  query: string,
  collection: string,
  limit?: number,
  filter?: Record<string, any>,
  with_payload?: boolean | string[]
}) {
  const embedding = await generateEmbedding(params.query);
  const client = new QdrantClient({ host: 'localhost', port: 6333 });
  
  return await client.search(params.collection, {
    vector: embedding,
    limit: params.limit || 10,
    filter: params.filter,
    with_payload: params.with_payload || true
  });
}
```

### 2. Hybrid Search Tool

```typescript
export async function hybridSearch(params: {
  query: string,
  collection: string,
  limit?: number,
  filter?: Record<string, any>,
  textFieldName: string
}) {
  // Vector search
  const vectorResults = await naturalLanguageQuery({
    query: params.query,
    collection: params.collection,
    limit: params.limit ? params.limit * 3 : 30,
    filter: params.filter
  });
  
  // Keyword search via payload filter
  const keywordFilter = buildKeywordFilter(params.query, params.textFieldName);
  const keywordResults = await client.scroll(params.collection, {
    filter: keywordFilter,
    limit: params.limit ? params.limit * 3 : 30
  });
  
  // Combine and re-rank results
  return reRankResults(vectorResults, keywordResults, params.limit || 10);
}
```

### 3. Multi-Vector Search Tool

```typescript
export async function multiVectorSearch(params: {
  queries: string[],
  collection: string,
  weights?: number[],
  limit?: number
}) {
  const embeddings = await Promise.all(
    params.queries.map(query => generateEmbedding(query))
  );
  
  // Normalize weights if provided
  const weights = params.weights || embeddings.map(() => 1 / embeddings.length);
  
  return await client.search(params.collection, {
    vector: combineVectors(embeddings, weights),
    limit: params.limit || 10
  });
}
```

## Collection Management Tools

### 4. Collection Creation Tool

```typescript
export async function createCollection(params: {
  name: string,
  vector_size: number,
  distance?: 'Cosine' | 'Euclid' | 'Dot',
  hnsw_config?: Record<string, any>,
  optimizers_config?: Record<string, any>
}) {
  return await client.createCollection(params.name, {
    vectors: {
      size: params.vector_size,
      distance: params.distance || 'Cosine'
    },
    hnsw_config: params.hnsw_config,
    optimizers_config: params.optimizers_config
  });
}
```

### 5. Collection Migration Tool

```typescript
export async function migrateCollection(params: {
  source_collection: string,
  target_collection: string,
  batch_size?: number,
  transform_fn?: string
}) {
  const batchSize = params.batch_size || 100;
  let offset = null;
  const transformFn = params.transform_fn ? 
    eval(`(point) => { ${params.transform_fn} }`) : 