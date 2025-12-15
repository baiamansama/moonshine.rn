import * as ort from 'onnxruntime-react-native';
import * as FileSystem from 'expo-file-system';
import { Asset } from 'expo-asset';

const log = (...args: any[]) => console.log(`[MoonshineONNX]`, ...args);

// Special token IDs (from tokenizer_config.json)
const BOS_TOKEN_ID = 1; // <|startoftranscript|>
const EOS_TOKEN_ID = 2; // <|endoftext|>
const PAD_TOKEN_ID = 0;

const MAX_LENGTH = 448;
const SAMPLE_RATE = 16000;

export interface MoonshineConfig {
  encoderAsset: number; // require('./assets/onnx/encoder.onnx')
  decoderAsset: number;
  tokenizerAsset: number;
}

interface TokenizerData {
  model: {
    vocab: Record<string, number>;
  };
  added_tokens: Array<{ id: number; content: string }>;
}

export class MoonshineONNX {
  private encoderSession: ort.InferenceSession | null = null;
  private decoderSession: ort.InferenceSession | null = null;
  private tokenizer: TokenizerData | null = null;
  private idToToken: Map<number, string> = new Map();
  private isLoaded = false;

  async load(config: MoonshineConfig, onProgress?: (p: number) => void): Promise<void> {
    try {
      log('Loading models...');
      onProgress?.(0);

      // Load encoder
      const encoderUri = await this.getAssetUri(config.encoderAsset);
      log('Loading encoder from:', encoderUri);
      this.encoderSession = await ort.InferenceSession.create(encoderUri);
      log('Encoder loaded, inputs:', this.encoderSession.inputNames, 'outputs:', this.encoderSession.outputNames);
      onProgress?.(0.33);

      // Load decoder
      const decoderUri = await this.getAssetUri(config.decoderAsset);
      log('Loading decoder from:', decoderUri);
      this.decoderSession = await ort.InferenceSession.create(decoderUri);
      log('Decoder loaded, inputs:', this.decoderSession.inputNames, 'outputs:', this.decoderSession.outputNames);
      onProgress?.(0.66);

      // Load tokenizer
      const tokenizerUri = await this.getAssetUri(config.tokenizerAsset);
      log('Loading tokenizer from:', tokenizerUri);
      const tokenizerJson = await FileSystem.readAsStringAsync(tokenizerUri);
      this.tokenizer = JSON.parse(tokenizerJson);
      this.buildIdToToken();
      log('Tokenizer loaded, vocab size:', this.idToToken.size);
      onProgress?.(1);

      this.isLoaded = true;
      log('All models loaded successfully');
    } catch (error) {
      log('Load error:', error);
      throw error;
    }
  }

  private async getAssetUri(asset: number): Promise<string> {
    const [loaded] = await Asset.loadAsync(asset);
    if (!loaded.localUri) {
      throw new Error('Failed to load asset');
    }
    return loaded.localUri;
  }

  private buildIdToToken(): void {
    if (!this.tokenizer) return;

    // Build from vocab
    const vocab = this.tokenizer.model?.vocab || {};
    for (const [token, id] of Object.entries(vocab)) {
      this.idToToken.set(id, token);
    }

    // Add special tokens
    for (const added of this.tokenizer.added_tokens || []) {
      this.idToToken.set(added.id, added.content);
    }
  }

  get ready(): boolean {
    return this.isLoaded;
  }

  async transcribe(audioData: Float32Array): Promise<string> {
    if (!this.isLoaded || !this.encoderSession || !this.decoderSession) {
      throw new Error('Models not loaded');
    }

    log('Transcribe start, samples:', audioData.length);

    // Normalize audio
    const normalized = this.normalizeAudio(audioData);

    // Run encoder
    log('Running encoder...');
    const encoderOutput = await this.runEncoder(normalized);
    log('Encoder output shape:', encoderOutput.dims);

    // Run decoder (greedy search)
    log('Running decoder...');
    const tokens = await this.greedyDecode(encoderOutput);
    log('Generated tokens:', tokens.length);

    // Decode tokens to text
    const text = this.decodeTokens(tokens);
    log('Transcription:', text);

    return text;
  }

  private normalizeAudio(audio: Float32Array): Float32Array {
    // Normalize to zero mean and unit variance
    let sum = 0;
    for (let i = 0; i < audio.length; i++) {
      sum += audio[i];
    }
    const mean = sum / audio.length;

    let variance = 0;
    for (let i = 0; i < audio.length; i++) {
      variance += (audio[i] - mean) ** 2;
    }
    const std = Math.sqrt(variance / audio.length) || 1;

    const normalized = new Float32Array(audio.length);
    for (let i = 0; i < audio.length; i++) {
      normalized[i] = (audio[i] - mean) / std;
    }
    return normalized;
  }

  private async runEncoder(audio: Float32Array): Promise<ort.Tensor> {
    // Input shape: [batch, samples] = [1, audio_length]
    const inputTensor = new ort.Tensor('float32', audio, [1, audio.length]);

    const feeds: Record<string, ort.Tensor> = {};
    const inputName = this.encoderSession!.inputNames[0];
    feeds[inputName] = inputTensor;

    const results = await this.encoderSession!.run(feeds);
    const outputName = this.encoderSession!.outputNames[0];
    return results[outputName];
  }

  private async greedyDecode(encoderOutput: ort.Tensor): Promise<number[]> {
    const tokens: number[] = [BOS_TOKEN_ID];

    for (let step = 0; step < MAX_LENGTH; step++) {
      // Prepare decoder inputs
      const inputIds = new BigInt64Array(tokens.map(t => BigInt(t)));
      const inputIdsTensor = new ort.Tensor('int64', inputIds, [1, tokens.length]);

      const feeds: Record<string, ort.Tensor> = {};

      // Try to match decoder input names
      const inputNames = this.decoderSession!.inputNames;
      log('Decoder step', step, 'input names:', inputNames);

      for (const name of inputNames) {
        if (name.includes('input_ids') || name.includes('decoder_input_ids')) {
          feeds[name] = inputIdsTensor;
        } else if (name.includes('encoder') || name.includes('hidden')) {
          feeds[name] = encoderOutput;
        }
      }

      // If we couldn't match, try positional
      if (Object.keys(feeds).length < inputNames.length) {
        feeds[inputNames[0]] = inputIdsTensor;
        if (inputNames.length > 1) {
          feeds[inputNames[1]] = encoderOutput;
        }
      }

      const results = await this.decoderSession!.run(feeds);
      const logitsName = this.decoderSession!.outputNames[0];
      const logits = results[logitsName];

      // Get last token logits
      const logitsData = logits.data as Float32Array;
      const vocabSize = logits.dims[logits.dims.length - 1];
      const seqLen = logits.dims[1];
      const lastTokenOffset = (seqLen - 1) * vocabSize;

      // Argmax
      let bestToken = 0;
      let bestScore = -Infinity;
      for (let i = 0; i < vocabSize; i++) {
        const score = logitsData[lastTokenOffset + i];
        if (score > bestScore) {
          bestScore = score;
          bestToken = i;
        }
      }

      if (step < 5) {
        log('Step', step, 'token:', bestToken, 'score:', bestScore.toFixed(2));
      }

      if (bestToken === EOS_TOKEN_ID) {
        break;
      }

      tokens.push(bestToken);
    }

    // Remove BOS token
    return tokens.slice(1);
  }

  private decodeTokens(tokens: number[]): string {
    const pieces: string[] = [];
    for (const id of tokens) {
      const token = this.idToToken.get(id);
      if (token && !token.startsWith('<|') && !token.endsWith('|>')) {
        // Handle byte-level BPE (Ġ = space)
        pieces.push(token.replace(/Ġ/g, ' '));
      }
    }
    return pieces.join('').trim();
  }

  dispose(): void {
    this.encoderSession?.release();
    this.decoderSession?.release();
    this.encoderSession = null;
    this.decoderSession = null;
    this.isLoaded = false;
  }
}
