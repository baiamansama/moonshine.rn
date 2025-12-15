import { StatusBar } from 'expo-status-bar';
import { useEffect, useRef, useState, useCallback } from 'react';
import {
  StyleSheet,
  Text,
  View,
  TouchableOpacity,
  ScrollView,
  ActivityIndicator,
} from 'react-native';
import { SafeAreaView, SafeAreaProvider } from 'react-native-safe-area-context';
import { useSpeechToText } from 'react-native-executorch';
import { AudioManager, AudioRecorder } from 'react-native-audio-api';

// =============================================================================
// MOONSHINE ARABIC MODEL CONFIGURATION
// =============================================================================
// Files needed in assets/models/:
//   - moonshine_tiny_ar_encoder_xnnpack.pte
//   - moonshine_tiny_ar_decoder_xnnpack.pte
//   - tokenizer.json
//
// Once exported, uncomment the require() lines below.
// =============================================================================

// Option 1: Local assets (uncomment when files are available)
// const MOONSHINE_ARABIC_MODEL = {
//   isMultilingual: false,
//   encoderSource: require('./assets/models/moonshine_tiny_ar_encoder_xnnpack.pte'),
//   decoderSource: require('./assets/models/moonshine_tiny_ar_decoder_xnnpack.pte'),
//   tokenizerSource: require('./assets/models/tokenizer.json'),
// };

// Option 2: Remote URLs (for when hosted on HuggingFace)
const MOONSHINE_ARABIC_MODEL = {
  isMultilingual: false,
  encoderSource: 'https://huggingface.co/software-mansion/react-native-executorch-moonshine-tiny-ar/resolve/v0.6.0/xnnpack/moonshine_tiny_ar_encoder_xnnpack.pte',
  decoderSource: 'https://huggingface.co/software-mansion/react-native-executorch-moonshine-tiny-ar/resolve/v0.6.0/xnnpack/moonshine_tiny_ar_decoder_xnnpack.pte',
  tokenizerSource: 'https://huggingface.co/software-mansion/react-native-executorch-moonshine-tiny-ar/resolve/v0.6.0/tokenizer.json',
};

// Option 3: Test with Whisper English (working model)
// import { WHISPER_TINY_EN } from 'react-native-executorch';

const SAMPLE_RATE = 16000;
const BUFFER_SIZE = 1600; // 100ms chunks

function MoonshineApp() {
  const [isRecording, setIsRecording] = useState(false);
  const [hasPermission, setHasPermission] = useState(false);
  const recorderRef = useRef<AudioRecorder | null>(null);

  console.log('[App] Mounting with model config');

  const {
    isReady,
    isGenerating,
    downloadProgress,
    error,
    committedTranscription,
    nonCommittedTranscription,
    stream,
    streamInsert,
    streamStop,
  } = useSpeechToText({
    model: MOONSHINE_ARABIC_MODEL,
    // For Whisper test: model: WHISPER_TINY_EN,
  });

  useEffect(() => {
    console.log('[Model]', { isReady, downloadProgress, error: error || 'none' });
  }, [isReady, downloadProgress, error]);

  useEffect(() => {
    if (committedTranscription || nonCommittedTranscription) {
      console.log('[Transcription]', { committedTranscription, nonCommittedTranscription });
    }
  }, [committedTranscription, nonCommittedTranscription]);

  useEffect(() => {
    const setupAudio = async () => {
      console.log('[Audio] Setup...');
      try {
        AudioManager.setAudioSessionOptions({
          iosCategory: 'playAndRecord',
          iosMode: 'spokenAudio',
          iosOptions: ['allowBluetooth', 'defaultToSpeaker'],
        });
        const status = await AudioManager.requestRecordingPermissions();
        console.log('[Audio] Permission:', status);
        setHasPermission(status === 'Granted');
      } catch (err) {
        console.error('[Audio] Error:', err);
      }
    };
    setupAudio();
  }, []);

  useEffect(() => {
    if (!hasPermission) return;

    console.log('[Recorder] Init 16kHz/1600');
    const recorder = new AudioRecorder({
      sampleRate: SAMPLE_RATE,
      bufferLengthInSamples: BUFFER_SIZE,
    });

    let chunks = 0;
    recorder.onAudioReady(({ buffer }) => {
      chunks++;
      if (chunks % 10 === 0) console.log('[Recorder] Chunk', chunks);
      streamInsert(buffer.getChannelData(0));
    });

    recorderRef.current = recorder;
    return () => { recorder.stop(); };
  }, [hasPermission, streamInsert]);

  const handleStart = useCallback(() => {
    if (!recorderRef.current || !isReady) return;
    console.log('[Recording] Start');
    stream();
    recorderRef.current.start();
    setIsRecording(true);
  }, [isReady, stream]);

  const handleStop = useCallback(() => {
    if (!recorderRef.current) return;
    console.log('[Recording] Stop');
    recorderRef.current.stop();
    streamStop();
    setIsRecording(false);
  }, [streamStop]);

  const fullTranscription = `${committedTranscription}${nonCommittedTranscription}`.trim();

  // Model not ready - show status
  if (!isReady) {
    const isModelMissing = error?.includes('fetch') || error?.includes('401');

    return (
      <SafeAreaView style={styles.container}>
        <View style={styles.loadingContainer}>
          {!error && <ActivityIndicator size="large" color="#007AFF" />}

          <Text style={styles.titleArabic}>Moonshine Arabic</Text>

          {downloadProgress > 0 && downloadProgress < 1 && (
            <Text style={styles.progressText}>
              Downloading: {Math.round(downloadProgress * 100)}%
            </Text>
          )}

          {isModelMissing ? (
            <View style={styles.infoContainer}>
              <Text style={styles.infoTitle}>Model Not Available</Text>
              <Text style={styles.infoText}>
                Waiting for moonshine-tiny-ar export.
              </Text>
              <Text style={styles.infoText}>
                See MAINTAINER_REQUEST.md
              </Text>
              <View style={styles.fileList}>
                <Text style={styles.fileItem}>• encoder.pte</Text>
                <Text style={styles.fileItem}>• decoder.pte</Text>
                <Text style={styles.fileItem}>• tokenizer.json</Text>
              </View>
            </View>
          ) : error ? (
            <View style={styles.errorContainer}>
              <Text style={styles.errorText}>{error}</Text>
            </View>
          ) : (
            <Text style={styles.loadingText}>Loading model...</Text>
          )}
        </View>
        <StatusBar style="auto" />
      </SafeAreaView>
    );
  }

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
        <Text style={styles.title}>Live Arabic Transcription</Text>
        <Text style={styles.subtitle}>
          {isGenerating ? 'جاري النسخ...' : 'اضغط للبدء'}
        </Text>
      </View>

      <ScrollView
        style={styles.transcriptionContainer}
        contentContainerStyle={styles.transcriptionContent}
      >
        {fullTranscription ? (
          <Text style={styles.transcriptionText}>
            {committedTranscription}
            <Text style={styles.pendingText}>{nonCommittedTranscription}</Text>
          </Text>
        ) : (
          <Text style={styles.placeholderText}>سيظهر كلامك هنا...</Text>
        )}
      </ScrollView>

      <View style={styles.controls}>
        <TouchableOpacity
          style={[styles.recordButton, isRecording && styles.recordButtonActive]}
          onPress={isRecording ? handleStop : handleStart}
          activeOpacity={0.7}
        >
          <View style={[styles.recordButtonInner, isRecording && styles.recordButtonInnerActive]} />
        </TouchableOpacity>
        <Text style={styles.recordLabel}>
          {isRecording ? 'اضغط للإيقاف' : 'اضغط للتسجيل'}
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
  loadingText: { marginTop: 16, fontSize: 16, color: '#8E8E93' },
  progressText: { marginTop: 12, fontSize: 16, color: '#007AFF', fontWeight: '600' },
  infoContainer: { marginTop: 20, padding: 20, backgroundColor: '#F0F4FF', borderRadius: 16, maxWidth: 300, alignItems: 'center' },
  infoTitle: { fontSize: 18, fontWeight: '600', color: '#1C1C1E', marginBottom: 8 },
  infoText: { fontSize: 14, color: '#666', textAlign: 'center', marginBottom: 4 },
  fileList: { marginTop: 12, alignItems: 'flex-start' },
  fileItem: { fontSize: 13, color: '#007AFF', fontFamily: 'Courier', marginVertical: 2 },
  errorContainer: { marginTop: 20, padding: 16, backgroundColor: '#FFF5F5', borderRadius: 12, maxWidth: 300 },
  errorText: { fontSize: 14, color: '#FF3B30', textAlign: 'center' },
  header: { paddingHorizontal: 20, paddingTop: 20, paddingBottom: 12 },
  titleArabic: { fontSize: 28, fontWeight: '700', color: '#1C1C1E', textAlign: 'right', writingDirection: 'rtl' },
  title: { fontSize: 16, fontWeight: '500', color: '#8E8E93', marginTop: 2 },
  subtitle: { fontSize: 14, color: '#8E8E93', marginTop: 8, textAlign: 'right', writingDirection: 'rtl' },
  transcriptionContainer: { flex: 1, marginHorizontal: 20, marginVertical: 12, backgroundColor: '#FFF', borderRadius: 16, shadowColor: '#000', shadowOffset: { width: 0, height: 2 }, shadowOpacity: 0.05, shadowRadius: 8, elevation: 2 },
  transcriptionContent: { padding: 20, minHeight: '100%' },
  transcriptionText: { fontSize: 22, lineHeight: 36, color: '#1C1C1E', textAlign: 'right', writingDirection: 'rtl' },
  pendingText: { color: '#8E8E93' },
  placeholderText: { fontSize: 18, color: '#C7C7CC', fontStyle: 'italic', textAlign: 'right', writingDirection: 'rtl' },
  controls: { alignItems: 'center', paddingVertical: 30 },
  recordButton: { width: 80, height: 80, borderRadius: 40, backgroundColor: '#FFF', justifyContent: 'center', alignItems: 'center', shadowColor: '#000', shadowOffset: { width: 0, height: 4 }, shadowOpacity: 0.1, shadowRadius: 12, elevation: 4 },
  recordButtonActive: { backgroundColor: '#FFE5E5' },
  recordButtonInner: { width: 56, height: 56, borderRadius: 28, backgroundColor: '#FF3B30' },
  recordButtonInnerActive: { width: 24, height: 24, borderRadius: 4 },
  recordLabel: { marginTop: 12, fontSize: 14, color: '#8E8E93' },
});
