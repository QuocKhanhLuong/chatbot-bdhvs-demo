const fs = require('fs');
const path = require('path');

// Dynamic import for pdf-parse (ESM module)
let pdfParse;

/**
 * Chunk text into smaller pieces for better RAG accuracy
 * @param {string} text - Full text to chunk
 * @param {number} maxChunkSize - Maximum characters per chunk
 * @returns {string[]} Array of text chunks
 */
function chunkText(text, maxChunkSize = 800) {
    const chunks = [];
    const paragraphs = text.split(/\n\n+/);

    let currentChunk = '';

    for (const paragraph of paragraphs) {
        // If adding this paragraph would exceed max size, save current chunk
        if (currentChunk.length + paragraph.length > maxChunkSize && currentChunk.length > 0) {
            chunks.push(currentChunk.trim());
            currentChunk = '';
        }

        // If single paragraph is too long, split it
        if (paragraph.length > maxChunkSize) {
            const sentences = paragraph.split(/[.!?]+/);
            for (const sentence of sentences) {
                if (currentChunk.length + sentence.length > maxChunkSize && currentChunk.length > 0) {
                    chunks.push(currentChunk.trim());
                    currentChunk = '';
                }
                currentChunk += sentence + '. ';
            }
        } else {
            currentChunk += paragraph + '\n\n';
        }
    }

    // Add remaining chunk
    if (currentChunk.trim().length > 0) {
        chunks.push(currentChunk.trim());
    }

    return chunks.filter(chunk => chunk.length > 50); // Filter out very small chunks
}

/**
 * Main ingestion function
 */
async function ingestPDFs() {
    console.log('🚀 Starting PDF ingestion...\n');

    // Load pdf-parse dynamically
    try {
        pdfParse = (await import('pdf-parse')).default;
    } catch (err) {
        console.error('❌ Failed to load pdf-parse:', err.message);
        return;
    }

    const pdfDir = path.join(__dirname, '../data/pdf');
    const outputPath = path.join(__dirname, '../utils/knowledge_base.json');

    // Check if PDF directory exists
    if (!fs.existsSync(pdfDir)) {
        console.error(`❌ PDF directory not found: ${pdfDir}`);
        console.log('Creating data/pdf directory...');
        fs.mkdirSync(pdfDir, { recursive: true });
        console.log('✅ Please place your PDF files in data/pdf/ and run this script again.');
        return;
    }

    // Get all PDF files
    const files = fs.readdirSync(pdfDir).filter(f => f.endsWith('.pdf'));

    if (files.length === 0) {
        console.error('❌ No PDF files found in data/pdf/');
        return;
    }

    console.log(`📚 Found ${files.length} PDF files:`);
    files.forEach(f => console.log(`  - ${f}`));
    console.log('');

    const knowledgeBase = [];
    let chunkId = 1;

    // Process each PDF
    for (const file of files) {
        try {
            console.log(`📄 Processing: ${file}...`);
            const filePath = path.join(pdfDir, file);
            const dataBuffer = fs.readFileSync(filePath);
            const data = await pdfParse(dataBuffer);

            console.log(`  ✓ Extracted ${data.text.length} characters`);

            // Chunk the text
            const chunks = chunkText(data.text);
            console.log(`  ✓ Created ${chunks.length} chunks`);

            // Add to knowledge base
            for (const chunk of chunks) {
                knowledgeBase.push({
                    id: `${chunkId}`,
                    source: file,
                    content: chunk
                });
                chunkId++;
            }

            console.log(`  ✅ Successfully processed ${file}\n`);
        } catch (error) {
            console.error(`  ❌ Error processing ${file}:`, error.message);
        }
    }

    // Save to JSON
    console.log(`💾 Saving ${knowledgeBase.length} chunks to ${outputPath}...`);
    fs.writeFileSync(outputPath, JSON.stringify(knowledgeBase, null, 2), 'utf-8');

    console.log('✅ Ingestion complete!');
    console.log(`\n📊 Summary:`);
    console.log(`  - PDFs processed: ${files.length}`);
    console.log(`  - Total chunks: ${knowledgeBase.length}`);
    console.log(`  - Output file: ${outputPath}`);
}

// Run the ingestion
ingestPDFs().catch(error => {
    console.error('❌ Fatal error:', error);
    process.exit(1);
});
