const { getDefaultConfig } = require('expo/metro-config');
const { wrapWithAudioAPIMetroConfig } = require('react-native-audio-api/metro-config');

const defaultConfig = getDefaultConfig(__dirname);

// Add support for .pte and .bin files (for ExecuTorch models)
defaultConfig.resolver.assetExts.push('pte', 'bin');

module.exports = wrapWithAudioAPIMetroConfig(defaultConfig);
