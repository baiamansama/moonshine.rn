const { getDefaultConfig } = require('expo/metro-config');
const { wrapWithAudioAPIMetroConfig } = require('react-native-audio-api/metro-config');

const defaultConfig = getDefaultConfig(__dirname);

// Add support for .pte, .bin, and .onnx files (for AI models)
defaultConfig.resolver.assetExts.push('pte', 'bin', 'onnx');

module.exports = wrapWithAudioAPIMetroConfig(defaultConfig);
