
import { logTitle } from "./utils";
import VectorStore from "./VectorStore";
import 'dotenv/config';
import { InferenceClient } from "@huggingface/inference";

export default class EmbeddingRetriever {
    private embeddingModel: string;
    private vectorStore: VectorStore;
    private hfClient: InferenceClient;

    constructor(embeddingModel: string) {
        this.embeddingModel = embeddingModel;
        this.vectorStore = new VectorStore();
        this.hfClient = new InferenceClient(process.env.HUGGINGFACE_API_KEY!);
    }

    async embedDocument(document: string) {
        logTitle('EMBEDDING DOCUMENT');
        const embedding = await this.embed(document);
        this.vectorStore.addEmbedding(embedding, document);
        return embedding;
    }

    async embedQuery(query: string) {
        logTitle('EMBEDDING QUERY');
        const embedding = await this.embed(query);
        return embedding;
    }

    private async embed(document: string): Promise<number[]> {
        // Use the InferenceClient to get the embedding for a single document
        const output = await this.hfClient.featureExtraction({
            model: this.embeddingModel,
            inputs: document,
            provider: "hf-inference",
        });
        // output is an array of numbers (embedding)
        console.log(output);
        return output as number[];
    }

    async retrieve(query: string, topK: number = 3): Promise<string[]> {
        const queryEmbedding = await this.embedQuery(query);
        return this.vectorStore.search(queryEmbedding, topK);
    }
}