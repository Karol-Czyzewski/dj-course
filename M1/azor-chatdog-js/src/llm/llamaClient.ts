/**
 * Local LLaMA model client using node-llama-cpp
 */

import {
  LlamaChatSession,
  getLlama,
  type ChatHistoryItem,
  type LLamaChatPromptOptions,
  type LlamaModel,
} from "node-llama-cpp";
import type {
  ILLMClient,
  ILLMChatSession,
  Message,
  LLMResponse,
} from "../types/index.js";
import { validateLlamaConfig } from "./llamaValidation.js";

type LlamaGenerationConfig = {
  temperature?: number;
  topP?: number;
  topK?: number;
  minP?: number;
  maxTokens?: number;
  seed?: number;
  responsePrefix?: string;
};

type LlamaChatSessionInternal = LlamaChatSession & {
  _chatHistory?: ChatHistoryItem[];
  _chatHistoryStateRef?: Record<string, unknown>;
  _lastEvaluation?: unknown;
  _canUseContextWindowForCompletion?: boolean;
};

const DEFAULT_GENERATION_CONFIG: LlamaGenerationConfig = {
  temperature: undefined,
  topP: undefined,
  topK: undefined,
  minP: undefined,
  maxTokens: undefined,
  seed: undefined,
  responsePrefix: undefined,
};

/**
 * LLaMA chat session wrapper backed by node-llama-cpp
 */
class LlamaChatSessionWrapper implements ILLMChatSession {
  private history: Message[] = [];
  private readonly sessionPromise: Promise<LlamaChatSession>;
  private readonly client: LlamaClient;

  constructor(
    client: LlamaClient,
    systemInstruction: string,
    initialHistory?: Message[]
  ) {
    this.client = client;
    this.history = initialHistory ? [...initialHistory] : [];
    this.sessionPromise = this.client.createNativeSession(systemInstruction, [
      ...this.history,
    ]);
  }

  private async getNativeSession(): Promise<LlamaChatSession> {
    return this.sessionPromise;
  }

  async sendMessage(text: string): Promise<LLMResponse> {
    const session = await this.getNativeSession();
    const responseText = await session.prompt(
      text,
      this.client.getPromptOptions()
    );

    this.history.push({
      role: "user",
      parts: [{ text }],
    });

    this.history.push({
      role: "model",
      parts: [{ text: responseText }],
    });

    return { text: responseText };
  }

  getHistory(): Message[] {
    return this.history;
  }
}

/**
 * LLaMA LLM Client implementation
 */
export class LlamaClient implements ILLMClient {
  private static modelCache = new Map<string, Promise<LlamaModel>>();

  private modelName: string;
  private modelPath: string;
  private gpuLayers: number;
  private contextSize: number;
  private readonly generationConfig: LlamaGenerationConfig;

  constructor(
    modelName: string,
    modelPath: string,
    gpuLayers: number,
    contextSize: number,
    generationConfig: LlamaGenerationConfig
  ) {
    this.modelName = modelName;
    this.modelPath = modelPath;
    this.gpuLayers = gpuLayers;
    this.contextSize = contextSize;
    this.generationConfig = generationConfig;
  }

  /**
   * Create client from environment variables
   */
  static fromEnvironment(): LlamaClient {
    const config = validateLlamaConfig();
    return new LlamaClient(
      config.modelName,
      config.llamaModelPath,
      config.llamaGpuLayers,
      config.llamaContextSize,
      {
        ...DEFAULT_GENERATION_CONFIG,
        temperature: parseNullableFloat(process.env.LLAMA_TEMPERATURE),
        topP: parseNullableFloat(process.env.LLAMA_TOP_P),
        topK: parseNullableInt(process.env.LLAMA_TOP_K),
        minP: parseNullableFloat(process.env.LLAMA_MIN_P),
        maxTokens: parseNullableInt(process.env.LLAMA_MAX_TOKENS),
        seed: parseNullableInt(process.env.LLAMA_SEED),
        responsePrefix: process.env.LLAMA_RESPONSE_PREFIX?.trim() || undefined,
      }
    );
  }

  /**
   * Create a chat session
   */
  createChatSession(
    systemInstruction: string,
    history?: Message[]
  ): ILLMChatSession {
    return new LlamaChatSessionWrapper(this, systemInstruction, history);
  }

  /**
   * Count tokens in history (heuristic)
   */
  countHistoryTokens(history: Message[]): number {
    let totalTokens = 0;
    for (const msg of history) {
      for (const part of msg.parts) {
        totalTokens += Math.ceil(part.text.length / 4);
      }
    }
    return totalTokens;
  }

  getModelName(): string {
    return this.modelName;
  }

  isAvailable(): boolean {
    return !!this.modelPath && this.modelPath.length > 0;
  }

  preparingForUseMessage(): string {
    return `Loading LLaMA model from ${this.modelPath}...`;
  }

  readyForUseMessage(): string {
    return `LLaMA ${this.modelName} ready (GPU layers: ${this.gpuLayers}, Context: ${this.contextSize})`;
  }

  getPromptOptions(): LLamaChatPromptOptions | undefined {
    const options: LLamaChatPromptOptions = {};

    if (isFiniteNumber(this.generationConfig.temperature)) {
      options.temperature = this.generationConfig.temperature;
    }
    if (isFiniteNumber(this.generationConfig.topP)) {
      options.topP = this.generationConfig.topP;
    }
    if (isFiniteNumber(this.generationConfig.topK)) {
      options.topK = this.generationConfig.topK;
    }
    if (isFiniteNumber(this.generationConfig.minP)) {
      options.minP = this.generationConfig.minP;
    }
    if (isFiniteNumber(this.generationConfig.maxTokens)) {
      options.maxTokens = this.generationConfig.maxTokens;
    }
    if (isFiniteNumber(this.generationConfig.seed)) {
      options.seed = this.generationConfig.seed;
    }
    if (this.generationConfig.responsePrefix) {
      options.responsePrefix = this.generationConfig.responsePrefix;
    }

    return Object.keys(options).length > 0 ? options : undefined;
  }

  async createNativeSession(
    systemInstruction: string,
    history?: Message[]
  ): Promise<LlamaChatSession> {
    const model = await this.loadModel();
    const context = await model.createContext({
      contextSize: this.contextSize,
    });
    const sequence = context.getSequence();

    const session = new LlamaChatSession({
      contextSequence: sequence,
      systemPrompt: systemInstruction,
      autoDisposeSequence: true,
    });

    if (history && history.length > 0) {
      this.hydrateNativeHistory(session, history);
    }

    return session;
  }

  private hydrateNativeHistory(
    session: LlamaChatSession,
    history: Message[]
  ): void {
    const chatHistoryItems = this.convertToChatHistory(history);

    if (chatHistoryItems.length === 0) {
      return;
    }

    const sessionRef = session as unknown as LlamaChatSessionInternal;
    const existingHistory = Array.isArray(sessionRef._chatHistory)
      ? sessionRef._chatHistory
      : [];

    sessionRef._chatHistory = [...existingHistory, ...chatHistoryItems];
    sessionRef._chatHistoryStateRef = {};
    sessionRef._lastEvaluation = undefined;
    sessionRef._canUseContextWindowForCompletion = false;
  }

  private convertToChatHistory(history: Message[]): ChatHistoryItem[] {
    const items: ChatHistoryItem[] = [];

    for (const message of history) {
      const textParts = message.parts.map((part) => part.text);

      if (message.role === "user") {
        items.push({
          type: "user",
          text: textParts.join(""),
        });
      } else {
        items.push({
          type: "model",
          response: textParts.length > 0 ? textParts : [""],
        });
      }
    }

    return items;
  }

  private async loadModel(): Promise<LlamaModel> {
    const cacheKey = `${this.modelPath}|${this.gpuLayers}`;

    if (!LlamaClient.modelCache.has(cacheKey)) {
      const modelPromise = (async () => {
        try {
          const llama = await getLlama();
          return llama.loadModel({
            modelPath: this.modelPath,
            gpuLayers: this.gpuLayers,
          });
        } catch (error) {
          LlamaClient.modelCache.delete(cacheKey);
          throw error;
        }
      })();

      LlamaClient.modelCache.set(cacheKey, modelPromise);
    }

    return LlamaClient.modelCache.get(cacheKey)!;
  }
}

function parseNullableFloat(value?: string): number | undefined {
  if (value === undefined || value === "") {
    return undefined;
  }
  const parsed = Number.parseFloat(value);
  return Number.isFinite(parsed) ? parsed : undefined;
}

function parseNullableInt(value?: string): number | undefined {
  if (value === undefined || value === "") {
    return undefined;
  }
  const parsed = Number.parseInt(value, 10);
  return Number.isFinite(parsed) ? parsed : undefined;
}

function isFiniteNumber(value: number | undefined): value is number {
  return value !== undefined && Number.isFinite(value);
}
