import fs from 'fs';
import path from 'path';
const pdfParseModule = require('pdf-parse');
const pdfParse = pdfParseModule.default || pdfParseModule;

/**
 * Load and extract text from all PDF files in the data/pdf directory
 * @returns Array of text chunks from all PDFs
 */
export async function loadPDFDocuments(): Promise<string[]> {
    try {
        const pdfDir = path.join(process.cwd(), 'data/pdf');

        // Check if directory exists
        if (!fs.existsSync(pdfDir)) {
            console.error('PDF directory not found:', pdfDir);
            return [];
        }

        const files = fs.readdirSync(pdfDir).filter((f: string) => f.endsWith('.pdf'));
        console.log(`Found ${files.length} PDF files:`, files);

        const allChunks: string[] = [];

        for (const file of files) {
            try {
                console.log(`Loading PDF: ${file}...`);
                const filePath = path.join(pdfDir, file);
                const dataBuffer = fs.readFileSync(filePath);
                const data = await pdfParse(dataBuffer);

                console.log(`Extracted ${data.text.length} characters from ${file}`);

                // Split text into paragraphs (split by double newlines)
                const paragraphs = data.text
                    .split(/\n\n+/)
                    .map((p: string) => p.trim())
                    .filter((p: string) => p.length > 50); // Only keep meaningful paragraphs

                console.log(`Created ${paragraphs.length} chunks from ${file}`);
                allChunks.push(...paragraphs);
            } catch (error) {
                console.error(`Error processing PDF ${file}:`, error);
            }
        }

        console.log(`Total chunks loaded: ${allChunks.length}`);
        return allChunks;
    } catch (error) {
        console.error('Error loading PDF documents:', error);
        return [];
    }
}

/**
 * Synchronous version: Load PDFs once and cache
 * Use this to avoid async issues in Next.js API routes
 */
let cachedDocuments: string[] | null = null;
let loadingPromise: Promise<string[]> | null = null;

export function getPDFDocuments(): Promise<string[]> {
    if (cachedDocuments !== null) {
        return Promise.resolve(cachedDocuments);
    }

    if (loadingPromise === null) {
        loadingPromise = loadPDFDocuments().then(docs => {
            cachedDocuments = docs;
            loadingPromise = null;
            return docs;
        });
    }

    return loadingPromise;
}
