
/**
 * @license
 * SPDX-License-Identifier: Apache-2.0
 */
import { GoogleGenerativeAI } from '@google/generative-ai';
import { Tool } from "@modelcontextprotocol/sdk/types.js";
import 'dotenv/config';
import { logTitle } from "./utils";


export interface ToolCall {
    id: string;
    function: {
        name: string;
        arguments: string;
    };
}


export default class ChatAssistant {
    private ai: GoogleGenerativeAI;
    private model: string;
    private messages: { role: string; content: string }[] = [];
    private tools: Tool[];

    constructor(model: string, systemPrompt: string = '', tools: Tool[] = [], context: string = '') {
        this.ai = new GoogleGenerativeAI(process.env.API_KEY!);
        this.model = model;
        this.tools = tools;
        if (systemPrompt) this.messages.push({ role: "system", content: systemPrompt });
        if (context) this.messages.push({ role: "user", content: context });
    }

    async chat(prompt?: string): Promise<{ content: string, toolCalls: ToolCall[] }> {
        logTitle('CHAT');
        if (prompt) {
            this.messages.push({ role: "user", content: prompt });
        }
        // Gemini expects a single string or array of content blocks
        const contents = this.messages.map(m => ({ role: 'user', parts: [{ text: m.content }] }));
        const model = this.model || 'gemini-2.0-flash';
        const geminiModel = this.ai.getGenerativeModel({ model });
        const response = await geminiModel.generateContent({ contents });
        let content = "";
        // Gemini does not support tool calls in the same way as OpenAI, so we return an empty array
        let toolCalls: ToolCall[] = [];
        logTitle('RESPONSE');
        // Try to extract the text from the response
        if (response && response.response && response.response.candidates && response.response.candidates[0] && response.response.candidates[0].content && response.response.candidates[0].content.parts) {
            content = response.response.candidates[0].content.parts.map((p: any) => p.text).join('');
            process.stdout.write(content);
        } else if (response && response.response && response.response.text) {
            content = response.response.text();
            process.stdout.write(content);
        }
        this.messages.push({ role: "assistant", content });
        return {
            content,
            toolCalls,
        };
    }


    public appendToolResult(toolCallId: string, toolOutput: string) {
        // Gemini does not support tool calls, so this is a no-op or could be adapted for future use
    }

    // Gemini does not support tool definitions in the same way as OpenAI
    // This is a placeholder for compatibility
    private getToolsDefinition(): any[] {
        return [];
    }
}
