import MCPClient from "./MCPClient";
import Agent from "./Agent";
import path from "path";
import EmbeddingRetriever from "./EmbeddingRetriever";
import fs from "fs";
import { logTitle } from "./utils";
import 'dotenv/config';

const outPath = path.join(process.cwd(), 'output');
const TASK = `
Verifique as issues do repositório do GitHub https://github.com/igorrhamon/the_old_reader/ e responda a seguinte pergunta:
Como resolver a issue #1?
Crie um arquivo de texto com o passo a passo para resolver a issue.
Obtenhao contexto 
O arquivo deve ser salvo no diretório output com o nome "resolucao_issue_1.txt".
`

const fetchMCP = new MCPClient("mcp-server-fetch", "uvx", ['mcp-server-fetch']);
const fileMCP = new MCPClient("mcp-server-file", "npx", ['-y', '@modelcontextprotocol/server-filesystem', outPath]);
const githubMCP = new MCPClient(
    "github2",
    "npx",
    [
        "-y",
        "@modelcontextprotocol/server-github",
        `--GITHUB_PERSONAL_ACCESS_TOKEN=${process.env.GITHUB_PERSONAL_ACCESS_TOKEN}`
    ]
);

async function main() {
    // RAG
    const context = await retrieveContext();

    // Agent
    const agent = new Agent('gemini-2.0-flash', [fetchMCP, fileMCP, githubMCP], '', context);
    await agent.init();
    await agent.invoke(TASK);
    await agent.close();
}

main()

async function retrieveContext() {
    // RAG
    const embeddingRetriever = new EmbeddingRetriever("sentence-transformers/all-MiniLM-L6-v2");
    const knowledgeDir = path.join(process.cwd(), 'knowledge');
    const files = fs.readdirSync(knowledgeDir);
    for await (const file of files) {
        const content = fs.readFileSync(path.join(knowledgeDir, file), 'utf-8');
        await embeddingRetriever.embedDocument(content);
    }
    const context = (await embeddingRetriever.retrieve(TASK, 3)).join('\n');
    logTitle('CONTEXT');
    console.log(context);
    return context
}