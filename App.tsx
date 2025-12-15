import { StatusBar } from 'expo-status-bar';
import { useEffect, useRef, useState, useCallback } from 'react';
import {
  StyleSheet,
  Text,
  View,
  TouchableOpacity,
  ScrollView,
  ActivityIndicator,
  Platform,
  Alert,
} from 'react-native';
import { SafeAreaView, SafeAreaProvider } from 'react-native-safe-area-context';
import { AudioManager, AudioRecorder } from 'react-native-audio-api';
import { MoonshineONNX } from './src/MoonshineONNX';

// =============================================================================
// MOONSHINE ONNX MODEL CONFIGURATION
// =============================================================================
// Using ONNX Runtime for cross-platform inference
// Models: onnx-community/moonshine-tiny-ONNX (English, quantized int8)
// For Arabic: Export UsefulSensors/moonshine-tiny-ar to ONNX
// =============================================================================

const ONNX_CONFIG = {
  encoderAsset: require('./assets/onnx/encoder_model_int8.onnx'),
  decoderAsset: require('./assets/onnx/decoder_model_merged_int8.onnx'),
  tokenizerAsset: require('./assets/onnx/tokenizer.json'),
};

const SAMPLE_RATE = 16000;
const BUFFER_SIZE = 1600; // 100ms chunks
const MAX_RECORDING_SECONDS = 10;
const MAX_AUDIO_CHUNKS = Math.ceil((MAX_RECORDING_SECONDS * SAMPLE_RATE) / BUFFER_SIZE);

const log = (...args: any[]) => console.log(`[App]`, ...args);

function MoonshineApp() {
  const [isRecording, setIsRecording] = useState(false);
  const [isProcessing, setIsProcessing] = useState(false);
  const [hasPermission, setHasPermission] = useState(false);
  const [transcription, setTranscription] = useState('');
  const [error, setError] = useState<string | null>(null);
  const [modelReady, setModelReady] = useState(false);
  const [loadProgress, setLoadProgress] = useState(0);

  const recorderRef = useRef<AudioRecorder | null>(null);
  const audioChunksRef = useRef<Float32Array[]>([]);
  const moonshineRef = useRef<MoonshineONNX | null>(null);

  // Load ONNX models
  useEffect(() => {
    const loadModels = async () => {
      try {
        log('Loading ONNX models...');
        const moonshine = new MoonshineONNX();
        await moonshine.load(ONNX_CONFIG, (p) => {
          setLoadProgress(p);
          log('Load progress:', Math.round(p * 100) + '%');
        });
        moonshineRef.current = moonshine;
        setModelReady(true);
        log('Models loaded successfully');
      } catch (err: any) {
        log('Model load error:', err);
        setError(`Failed to load models: ${err?.message || err}`);
      }
    };
    loadModels();

    return () => {
      moonshineRef.current?.dispose();
    };
  }, []);

  // Request mic permission
  useEffect(() => {
    const requestPermission = async () => {
      try {
        if (Platform.OS === 'ios') {
          AudioManager.setAudioSessionOptions({
            iosCategory: 'playAndRecord',
            iosMode: 'spokenAudio',
            iosOptions: ['allowBluetooth', 'defaultToSpeaker'],
          });
        }
        const status = await AudioManager.requestRecordingPermissions();
        setHasPermission(status === 'Granted');
        log('Permission:', status);
      } catch (err: any) {
        setError(`Permission error: ${err?.message || err}`);
      }
    };
    requestPermission();
  }, []);

  // Initialize recorder
  useEffect(() => {
    if (!hasPermission) return;

    const recorder = new AudioRecorder({
      sampleRate: SAMPLE_RATE,
      bufferLengthInSamples: BUFFER_SIZE,
    });

    recorder.onAudioReady(({ buffer }) => {
      if (audioChunksRef.current.length < MAX_AUDIO_CHUNKS) {
        audioChunksRef.current.push(new Float32Array(buffer.getChannelData(0)));
      }
    });

    recorderRef.current = recorder;
    log('Recorder initialized');

    return () => {
      recorder.stop();
    };
  }, [hasPermission]);

  const handleStartRecording = useCallback(() => {
    if (!recorderRef.current || !modelReady) {
      Alert.alert('Not Ready', 'Please wait for models to load');
      return;
    }
    audioChunksRef.current = [];
    setTranscription('');
    setError(null);
    recorderRef.current.start();
    setIsRecording(true);
    log('Recording started');
  }, [modelReady]);

  const handleStopRecording = useCallback(async () => {
    if (!recorderRef.current) return;

    recorderRef.current.stop();
    setIsRecording(false);
    log('Recording stopped, chunks:', audioChunksRef.current.length);

    if (!moonshineRef.current || audioChunksRef.current.length === 0) {
      setError('No audio recorded');
      return;
    }

    setIsProcessing(true);
    setError(null);

    try {
      // Concatenate audio chunks
      const totalSamples = audioChunksRef.current.reduce((sum, arr) => sum + arr.length, 0);
      const audio = new Float32Array(totalSamples);
      let offset = 0;
      for (const chunk of audioChunksRef.current) {
        audio.set(chunk, offset);
        offset += chunk.length;
      }

      log('Processing audio, samples:', totalSamples, 'duration:', (totalSamples / SAMPLE_RATE).toFixed(2) + 's');

      const result = await moonshineRef.current.transcribe(audio);
      setTranscription(result);
    } catch (err: any) {
      log('Transcription error:', err);
      setError(`Transcription failed: ${err?.message || err}`);
    } finally {
      setIsProcessing(false);
    }
  }, []);

  // Loading state
  if (!modelReady) {
    return (
      <SafeAreaView style={styles.container}>
        <View style={styles.loadingContainer}>
          {!error && <ActivityIndicator size="large" color="#007AFF" />}
          <Text style={styles.titleText}>Moonshine ASR</Text>
          <Text style={styles.subtitleText}>ONNX Runtime</Text>
          {!error && (
            <Text style={styles.progressText}>
              Loading: {Math.round(loadProgress * 100)}%
            </Text>
          )}
          {error && (
            <View style={styles.errorContainer}>
              <Text style={styles.errorText}>{error}</Text>
            </View>
          )}
        </View>
        <StatusBar style="auto" />
      </SafeAreaView>
    );
  }

  // Permission state
  if (!hasPermission) {
    return (
      <SafeAreaView style={styles.container}>
        <View style={styles.loadingContainer}>
          <Text style={styles.errorText}>Microphone permission required</Text>
        </View>
        <StatusBar style="auto" />
      </SafeAreaView>
    );
  }

  return (
    <SafeAreaView style={styles.container}>
      <View style={styles.header}>
        <Text style={styles.titleArabic}>النسخ المباشر</Text>
        <Text style={styles.title}>Live Transcription (ONNX)</Text>
        <Text style={styles.subtitle}>
          {isRecording ? 'Recording...' : isProcessing ? 'Processing...' : 'Tap to start'}
        </Text>
      </View>

      <ScrollView
        style={styles.transcriptionContainer}
        contentContainerStyle={styles.transcriptionContent}
      >
        {transcription ? (
          <Text style={styles.transcriptionText}>{transcription}</Text>
        ) : (
          <Text style={styles.placeholderText}>Your speech will appear here...</Text>
        )}
      </ScrollView>

      {error && (
        <View style={styles.errorContainer}>
          <Text style={styles.errorText}>{error}</Text>
        </View>
      )}

      <View style={styles.controls}>
        <TouchableOpacity
          style={[styles.recordButton, isRecording && styles.recordButtonActive]}
          onPress={isRecording ? handleStopRecording : handleStartRecording}
          activeOpacity={0.7}
          disabled={isProcessing}
        >
          {isProcessing ? (
            <ActivityIndicator color="#FFF" />
          ) : (
            <View style={[styles.recordButtonInner, isRecording && styles.recordButtonInnerActive]} />
          )}
        </TouchableOpacity>
        <Text style={styles.recordLabel}>
          {isRecording ? 'Tap to stop' : isProcessing ? 'Processing...' : 'Tap to record'}
        </Text>
      </View>

      <StatusBar style="auto" />
    </SafeAreaView>
  );
}

export default function App() {
  return (
    <SafeAreaProvider>
      <MoonshineApp />
    </SafeAreaProvider>
  );
}

const styles = StyleSheet.create({
  container: { flex: 1, backgroundColor: '#F5F5F7' },
  loadingContainer: { flex: 1, justifyContent: 'center', alignItems: 'center', padding: 20 },
  titleText: { fontSize: 24, fontWeight: '700', color: '#1C1C1E', marginTop: 16 },
  subtitleText: { fontSize: 14, color: '#8E8E93', marginTop: 4 },
  progressText: { marginTop: 16, fontSize: 16, color: '#007AFF', fontWeight: '600' },
  errorContainer: { marginTop: 20, marginHorizontal: 20, padding: 16, backgroundColor: '#FFF5F5', borderRadius: 12 },
  errorText: { fontSize: 14, color: '#FF3B30', textAlign: 'center' },
  header: { paddingHorizontal: 20, paddingTop: 20, paddingBottom: 12 },
  titleArabic: { fontSize: 28, fontWeight: '700', color: '#1C1C1E', textAlign: 'right', writingDirection: 'rtl' },
  title: { fontSize: 16, fontWeight: '500', color: '#8E8E93', marginTop: 2 },
  subtitle: { fontSize: 14, color: '#8E8E93', marginTop: 8 },
  transcriptionContainer: { flex: 1, marginHorizontal: 20, marginVertical: 12, backgroundColor: '#FFF', borderRadius: 16, shadowColor: '#000', shadowOffset: { width: 0, height: 2 }, shadowOpacity: 0.05, shadowRadius: 8, elevation: 2 },
  transcriptionContent: { padding: 20, minHeight: '100%' },
  transcriptionText: { fontSize: 20, lineHeight: 32, color: '#1C1C1E' },
  placeholderText: { fontSize: 16, color: '#C7C7CC', fontStyle: 'italic' },
  controls: { alignItems: 'center', paddingVertical: 30 },
  recordButton: { width: 80, height: 80, borderRadius: 40, backgroundColor: '#FFF', justifyContent: 'center', alignItems: 'center', shadowColor: '#000', shadowOffset: { width: 0, height: 4 }, shadowOpacity: 0.1, shadowRadius: 12, elevation: 4 },
  recordButtonActive: { backgroundColor: '#FFE5E5' },
  recordButtonInner: { width: 56, height: 56, borderRadius: 28, backgroundColor: '#FF3B30' },
  recordButtonInnerActive: { width: 24, height: 24, borderRadius: 4 },
  recordLabel: { marginTop: 12, fontSize: 14, color: '#8E8E93' },
});
