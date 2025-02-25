import 'dart:io';

import 'package:dart_transformers/hub/hub.dart';
import 'package:dart_transformers/hub/hub_api.dart';
import 'package:dart_transformers/tokenizers/tokenizer.dart';
import 'package:flutter_test/flutter_test.dart';

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

  test('testFromPretrained', () async {
    final tokenizer = await AutoTokenizer.fromPretrained('coreml-projects/Llama-2-7b-chat-coreml', hubApi: hubApi);
    final inputIds = tokenizer.encode("Today she took a train to the West");
    expect(inputIds, [1, 20628, 1183, 3614, 263, 7945, 304, 278, 3122]);
  });
  test('testWhisper', () async {
    final tokenizer = await AutoTokenizer.fromPretrained('openai/whisper-large-v2', hubApi: hubApi);
    final inputIds = tokenizer.encode("Today she took a train to the West");
    expect(inputIds, [50258, 50363, 27676, 750, 1890, 257, 3847, 281, 264, 4055, 50257]);
  });

  test('testFromModelFolder', () async {
    final filesToDownload = ["config.json", "tokenizer_config.json", "tokenizer.json"];
    final repo = Repo(id: "coreml-projects/Llama-2-7b-chat-coreml");
    final localModelFolder = await hubApi.snapshot(repo, globs: filesToDownload);

    final tokenizer = await AutoTokenizer.fromModelFolder(localModelFolder, hubApi: hubApi);
    final inputIds = tokenizer.encode("Today she took a train to the West");
    expect(inputIds, [1, 20628, 1183, 3614, 263, 7945, 304, 278, 3122]);
  });

  test('testWhisperFromModelFolder', () async {
    final filesToDownload = ["config.json", "tokenizer_config.json", "tokenizer.json"];
    final repo = Repo(id: "openai/whisper-large-v2");
    final localModelFolder = await hubApi.snapshot(repo, globs: filesToDownload);

    final tokenizer = await AutoTokenizer.fromModelFolder(localModelFolder, hubApi: hubApi);
    final inputIds = tokenizer.encode("Today she took a train to the West");
    expect(inputIds, [50258, 50363, 27676, 750, 1890, 257, 3847, 281, 264, 4055, 50257]);
  });
}
