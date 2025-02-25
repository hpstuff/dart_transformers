import 'dart:convert';

import 'package:dart_transformers/hub/config.dart';
import 'package:dart_transformers/hub/hub.dart';
import 'package:dart_transformers/hub/hub_api.dart';
import 'package:flutter_test/flutter_test.dart';
import 'dart:io';

void main() {
  final downloadDestination = Directory.systemTemp.createTempSync('huggingface-tests');
  HubApi hubApi = HubApi(downloadBase: downloadDestination.uri);

  setUp(() async {
    // Setup code if needed
  });

  tearDown(() async {
    try {
      if (await downloadDestination.exists()) {
        await downloadDestination.delete(recursive: true);
      }
    } catch (e) {
      fail("Can't remove test download destination $downloadDestination, error: $e");
    }
  });

  test('testConfigDownload', () async {
    try {
      final configLoader = LanguageModelConfigurationFromHub.fromName(modelName: 't5-base', hubApi: hubApi);
      final config = await configLoader.modelConfig;

      // Test leaf value (Int)
      expect(config["eos_token_id"]?.intValue, 1);

      // Test leaf value (String)
      expect(config["model_type"]?.stringValue, 't5');

      // Test leaf value (Array)
      expect(config["architectures"]?.arrayValue, ['T5ForConditionalGeneration']);

      // Test nested wrapper
      expect(config["task_specific_params"]?["summarization"]?["max_length"]?.intValue, 200);
    } catch (e, s) {
      fail('Cannot download test configuration from the Hub: $e, $s');
    }
  });

  test('testConfigCamelCase', () async {
    try {
      final configLoader = LanguageModelConfigurationFromHub.fromName(modelName: 't5-base', hubApi: hubApi);
      final config = await configLoader.modelConfig;

      // Test leaf value (Int)
      expect(config["eosTokenId"]?.intValue, 1);

      // Test leaf value (String)
      expect(config["modelType"]?.stringValue, 't5');

      expect(config["taskSpecificParams"]?["summarization"]?["maxLength"]?.intValue, 200);
    } catch (e, s) {
      fail('Cannot download test configuration from the Hub: $e $s');
    }
  });

  test('testConfigUnicode', () {
    // These are two different characters
    final json = '{"vocab": {"à": 1, "à": 2}}';
    final dict = jsonDecode(json) as Map<String, dynamic>;
    final config = Config(dict);

    expect(config["vocab"]?.dictionary.length, 2);
  });
}
