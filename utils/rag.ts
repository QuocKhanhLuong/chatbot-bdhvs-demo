import knowledgeBase from './knowledge_base.json';

// Type definition for knowledge base items
type KnowledgeBaseItem = {
    id: string;
    source?: string;
    content: string;
};

/**
 * Retrieves relevant documents based on user query
 * Uses simple keyword matching with pre-processed JSON data
 * @param query - User's question
 * @returns Matching documents as a single string
 */
export async function retrieveContext(query: string): Promise<string> {
    if (knowledgeBase.length === 0) {
        console.warn('Knowledge base is empty - please run: node scripts/ingest-pdf.js');
        return '';
    }

    const queryLower = query.toLowerCase();
    const matchingDocs: KnowledgeBaseItem[] = [];

    for (const doc of knowledgeBase as KnowledgeBaseItem[]) {
        const docLower = doc.content.toLowerCase();

        // Extract keywords from query (split by spaces and filter short words)
        const keywords = queryLower
            .split(/\s+/)
            .filter(word => word.length > 2);

        // Check if any keyword is in the document
        const hasMatch = keywords.some(keyword => docLower.includes(keyword));

        if (hasMatch) {
            matchingDocs.push(doc);
        }
    }

    // Limit to top 5 most relevant chunks
    const topMatches = matchingDocs.slice(0, 5);

    return topMatches.map(doc => doc.content).join("\n\n");
}
