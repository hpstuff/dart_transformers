import 'dart:convert';
import 'dart:io' as io;

import 'package:dart_transformers/hub/hub.dart';
import 'package:dart_transformers/hub/hub_api.dart';
import 'package:dart_transformers/tokenizers/bpe_tokenizer.dart';
import 'package:dart_transformers/tokenizers/tokenizer.dart';
import 'package:dart_transformers/tokenizers/tokenizer_error.dart';
import 'package:flutter_test/flutter_test.dart';

void main() {
  TestWidgetsFlutterBinding.ensureInitialized();
  io.HttpOverrides.global = null;
  final downloadDestination = io.Directory.systemTemp.createTempSync('huggingface-tests');
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

  group("testDistilGPT2", () {
    final hubModelName = "distilgpt2";
    final encodedSamplesFilename = "gpt2_encoded_tokens";
    final unknownTokenId = 50256;
    Tokenizer? tokenizer;
    Map<String, dynamic>? dataset;
    List<Map<String, dynamic>> edgeCases = [];

    setUpAll(() async {
      final configuration = LanguageModelConfigurationFromHub.fromName(modelName: hubModelName, hubApi: hubApi);
      final tokenizerConfig = await configuration.tokenizerConfig;
      final tokenizerData = await configuration.tokenizerData;

      if (tokenizerConfig == null) {
        throw TokenizerError.tokenizerConfigNotFound();
      }

      tokenizer = AutoTokenizer.from(tokenizerConfig, tokenizerData);

      final file = io.File('test_assets/$encodedSamplesFilename.json');
      dataset = jsonDecode(await file.readAsString());

      final testFile = io.File('test_assets/tokenizer_tests.json');
      final Map<String, dynamic> testCase = jsonDecode(await testFile.readAsString());
      edgeCases = List.from(testCase[hubModelName] ?? []).cast();
    });

    test("testDistilGPT2-tokenize", () {
      final tokenized = tokenizer?.tokenize(dataset?["text"]);
      expect(tokenized, dataset?["bpe_tokens"]);
    });

    test("testDistilGPT2-encode", () {
      final encoded = tokenizer?.encode(dataset?["text"]);
      expect(encoded, dataset?["token_ids"]);
    });

    test("testDistilGPT2-decode", () {
      final decoded = tokenizer?.decode(List.from(dataset?["token_ids"]).cast());
      expect(decoded, dataset?["decoded_text"]);
    });

    test("testDistilGPT2-edgeCases", () {
      for (final edgeCase in edgeCases) {
        final encoded = tokenizer?.encode(edgeCase["input"]);
        expect(encoded, edgeCase["encoded"]["input_ids"]);
      }
    });

    test("testDistilGPT2-unknownToken", () {
      final tokenizerModel = (tokenizer as PreTrainedTokenizer).model;

      expect(tokenizerModel.unknownTokenId, unknownTokenId);
      expect(tokenizerModel.unknownTokenId, tokenizerModel.convertTokenToId("_this_token_does_not_exist_"));
      if (tokenizerModel.unknownTokenId != null) {
        expect(tokenizerModel.unknownToken, tokenizerModel.convertIdToToken(tokenizerModel.unknownTokenId!));
      } else {
        expect(tokenizerModel.unknownTokenId, isNull);
      }
    });
  });

  group("testFalcon", () {
    final hubModelName = "tiiuae/falcon-7b";
    final encodedSamplesFilename = "falcon_encoded";
    final unknownTokenId = null;
    Tokenizer? tokenizer;
    Map<String, dynamic>? dataset;
    List<Map<String, dynamic>> edgeCases = [];

    setUpAll(() async {
      final configuration = LanguageModelConfigurationFromHub.fromName(modelName: hubModelName, hubApi: hubApi);
      final tokenizerConfig = await configuration.tokenizerConfig;
      final tokenizerData = await configuration.tokenizerData;

      if (tokenizerConfig == null) {
        throw TokenizerError.tokenizerConfigNotFound();
      }

      tokenizer = AutoTokenizer.from(tokenizerConfig, tokenizerData);

      final file = io.File('test_assets/$encodedSamplesFilename.json');
      dataset = jsonDecode(await file.readAsString());

      final testFile = io.File('test_assets/tokenizer_tests.json');
      final Map<String, dynamic> testCase = jsonDecode(await testFile.readAsString());
      edgeCases = List.from(testCase[hubModelName] ?? []).cast();
    });

    test("testFalcon-tokenize", () {
      final tokenized = tokenizer?.tokenize(dataset?["text"]);
      expect(tokenized, dataset?["bpe_tokens"]);
    });

    test("testFalcon-encode", () {
      final encoded = tokenizer?.encode(dataset?["text"]);
      expect(encoded, dataset?["token_ids"]);
    });

    test("testFalcon-decode", () {
      final decoded = tokenizer?.decode(List.from(dataset?["token_ids"]).cast());
      expect(decoded, dataset?["decoded_text"]);
    });

    test("testFalcon-edgeCases", () {
      for (final edgeCase in edgeCases) {
        final encoded = tokenizer?.encode(edgeCase["input"]);
        expect(encoded, edgeCase["encoded"]["input_ids"]);
      }
    });

    test("testFalcon-unknownToken", () {
      final tokenizerModel = (tokenizer as PreTrainedTokenizer).model;

      expect(tokenizerModel.unknownTokenId, unknownTokenId);
      expect(tokenizerModel.unknownTokenId, tokenizerModel.convertTokenToId("_this_token_does_not_exist_"));
      if (tokenizerModel.unknownTokenId != null) {
        expect(tokenizerModel.unknownToken, tokenizerModel.convertIdToToken(tokenizerModel.unknownTokenId!));
      } else {
        expect(tokenizerModel.unknownTokenId, isNull);
      }
    });
  });

  group("testLlama", () {
    final hubModelName = "coreml-projects/Llama-2-7b-chat-coreml";
    final encodedSamplesFilename = "llama_encoded";
    final unknownTokenId = 0;
    Tokenizer? tokenizer;
    Map<String, dynamic>? dataset;
    List<Map<String, dynamic>> edgeCases = [];

    setUpAll(() async {
      final configuration = LanguageModelConfigurationFromHub.fromName(modelName: hubModelName, hubApi: hubApi);
      final tokenizerConfig = await configuration.tokenizerConfig;
      final tokenizerData = await configuration.tokenizerData;

      if (tokenizerConfig == null) {
        throw TokenizerError.tokenizerConfigNotFound();
      }

      tokenizer = AutoTokenizer.from(tokenizerConfig, tokenizerData);

      final file = io.File('test_assets/$encodedSamplesFilename.json');
      dataset = jsonDecode(await file.readAsString());

      final testFile = io.File('test_assets/tokenizer_tests.json');
      final Map<String, dynamic> testCase = jsonDecode(await testFile.readAsString());
      edgeCases = List.from(testCase[hubModelName] ?? []).cast();
    });

    test("testLlama2-hexaEncode", () {
      final tokenized = tokenizer?.tokenize("\n");
      expect(tokenized, ["▁", "<0x0A>"]);
    });

    test("testLlama2-tokenize", () {
      final tokenized = tokenizer?.tokenize(dataset?["text"]);
      expect(tokenized, dataset?["bpe_tokens"]);
    });

    test("testLlama2-encode", () {
      final encoded = tokenizer?.encode(dataset?["text"]);
      expect(encoded, dataset?["token_ids"]);
    });

    test("testLlama2-decode", () {
      final decoded = tokenizer?.decode(List.from(dataset?["token_ids"]).cast());
      expect(decoded, dataset?["decoded_text"]);
    });

    test("testLlama2-edgeCases", () {
      for (final edgeCase in edgeCases) {
        final encoded = tokenizer?.encode(edgeCase["input"]);
        expect(encoded, edgeCase["encoded"]["input_ids"]);
      }
    });

    test("testLlama2-unknownToken", () {
      final tokenizerModel = (tokenizer as PreTrainedTokenizer).model;

      expect(tokenizerModel.unknownTokenId, unknownTokenId);
      expect(tokenizerModel.unknownTokenId, tokenizerModel.convertTokenToId("_this_token_does_not_exist_"));
      if (tokenizerModel.unknownTokenId != null) {
        expect(tokenizerModel.unknownToken, tokenizerModel.convertIdToToken(tokenizerModel.unknownTokenId!));
      } else {
        expect(tokenizerModel.unknownTokenId, isNull);
      }
    });
  });

  group("testLlama3", () {
    final hubModelName = "pcuenq/Llama-3.2-1B-Instruct-tokenizer";
    final encodedSamplesFilename = "llama_3.2_encoded";
    final unknownTokenId = null;
    Tokenizer? tokenizer;
    Map<String, dynamic>? dataset;
    List<Map<String, dynamic>> edgeCases = [];

    setUpAll(() async {
      final configuration = LanguageModelConfigurationFromHub.fromName(modelName: hubModelName, hubApi: hubApi);
      final tokenizerConfig = await configuration.tokenizerConfig;
      final tokenizerData = await configuration.tokenizerData;

      if (tokenizerConfig == null) {
        throw TokenizerError.tokenizerConfigNotFound();
      }

      tokenizer = AutoTokenizer.from(tokenizerConfig, tokenizerData);

      final file = io.File('test_assets/$encodedSamplesFilename.json');
      dataset = jsonDecode(await file.readAsString());

      final testFile = io.File('test_assets/tokenizer_tests.json');
      final Map<String, dynamic> testCase = jsonDecode(await testFile.readAsString());
      edgeCases = List.from(testCase[hubModelName] ?? []).cast();
    });

    test("testLlama3-tokenize", () {
      final tokenized = tokenizer?.tokenize(dataset?["text"]);
      expect(tokenized, dataset?["bpe_tokens"]);
    });

    test("testLlama3-encode", () {
      final encoded = tokenizer?.encode(dataset?["text"]);
      expect(encoded, dataset?["token_ids"]);
    });

    test("testLlama3-decode", () {
      final decoded = tokenizer?.decode(List.from(dataset?["token_ids"]).cast());
      expect(decoded, dataset?["decoded_text"]);
    });

    test("testLlama3-edgeCases", () {
      for (final edgeCase in edgeCases) {
        final encoded = tokenizer?.encode(edgeCase["input"]);
        expect(encoded, edgeCase["encoded"]["input_ids"]);
      }
    });

    test("testLlama3-unknownToken", () {
      final tokenizerModel = (tokenizer as PreTrainedTokenizer).model;

      expect(tokenizerModel.unknownTokenId, unknownTokenId);
      expect(tokenizerModel.unknownTokenId, tokenizerModel.convertTokenToId("_this_token_does_not_exist_"));
      if (tokenizerModel.unknownTokenId != null) {
        expect(tokenizerModel.unknownToken, tokenizerModel.convertIdToToken(tokenizerModel.unknownTokenId!));
      } else {
        expect(tokenizerModel.unknownTokenId, isNull);
      }
    });
  });

  group("testWhisperLarge", () {
    final hubModelName = "openai/whisper-large-v2";
    final encodedSamplesFilename = "whisper_large_v2_encoded";
    final unknownTokenId = 50257;
    Tokenizer? tokenizer;
    Map<String, dynamic>? dataset;
    List<Map<String, dynamic>> edgeCases = [];

    setUpAll(() async {
      final configuration = LanguageModelConfigurationFromHub.fromName(modelName: hubModelName, hubApi: hubApi);
      final tokenizerConfig = await configuration.tokenizerConfig;
      final tokenizerData = await configuration.tokenizerData;

      if (tokenizerConfig == null) {
        throw TokenizerError.tokenizerConfigNotFound();
      }

      tokenizer = AutoTokenizer.from(tokenizerConfig, tokenizerData);

      final file = io.File('test_assets/$encodedSamplesFilename.json');
      dataset = jsonDecode(await file.readAsString());

      final testFile = io.File('test_assets/tokenizer_tests.json');
      final Map<String, dynamic> testCase = jsonDecode(await testFile.readAsString());
      edgeCases = List.from(testCase[hubModelName] ?? []).cast();
    });

    test("testWhisperLarge-tokenize", () {
      final tokenized = tokenizer?.tokenize(dataset?["text"]);
      expect(tokenized, dataset?["bpe_tokens"]);
    });

    test("testWhisperLarge-encode", () {
      final encoded = tokenizer?.encode(dataset?["text"]);
      expect(encoded, dataset?["token_ids"]);
    });

    test("testWhisperLarge-decode", () {
      final decoded = tokenizer?.decode(List.from(dataset?["token_ids"]).cast());
      expect(decoded, dataset?["decoded_text"]);
    });

    test("testWhisperLarge-edgeCases", () {
      for (final edgeCase in edgeCases) {
        final encoded = tokenizer?.encode(edgeCase["input"]);
        expect(encoded, edgeCase["encoded"]["input_ids"]);
      }
    });

    test("testWhisperLarge-unknownToken", () {
      final tokenizerModel = (tokenizer as PreTrainedTokenizer).model;

      expect(tokenizerModel.unknownTokenId, unknownTokenId);
      expect(tokenizerModel.unknownTokenId, tokenizerModel.convertTokenToId("_this_token_does_not_exist_"));
      if (tokenizerModel.unknownTokenId != null) {
        expect(tokenizerModel.unknownToken, tokenizerModel.convertIdToToken(tokenizerModel.unknownTokenId!));
      } else {
        expect(tokenizerModel.unknownTokenId, isNull);
      }
    });
  });

  group("testWhisperTiny", () {
    final hubModelName = "openai/whisper-tiny.en";
    final encodedSamplesFilename = "whisper_tiny_en_encoded";
    final unknownTokenId = 50256;
    Tokenizer? tokenizer;
    Map<String, dynamic>? dataset;
    List<Map<String, dynamic>> edgeCases = [];

    setUpAll(() async {
      final configuration = LanguageModelConfigurationFromHub.fromName(modelName: hubModelName, hubApi: hubApi);
      final tokenizerConfig = await configuration.tokenizerConfig;
      final tokenizerData = await configuration.tokenizerData;

      if (tokenizerConfig == null) {
        throw TokenizerError.tokenizerConfigNotFound();
      }

      tokenizer = AutoTokenizer.from(tokenizerConfig, tokenizerData);

      final file = io.File('test_assets/$encodedSamplesFilename.json');
      dataset = jsonDecode(await file.readAsString());

      final testFile = io.File('test_assets/tokenizer_tests.json');
      final Map<String, dynamic> testCase = jsonDecode(await testFile.readAsString());
      edgeCases = List.from(testCase[hubModelName] ?? []).cast();
    });

    test("testWhisperTiny-tokenize", () {
      final tokenized = tokenizer?.tokenize(dataset?["text"]);
      expect(tokenized, dataset?["bpe_tokens"]);
    });

    test("testWhisperTiny-encode", () {
      final encoded = tokenizer?.encode(dataset?["text"]);
      expect(encoded, dataset?["token_ids"]);
    });

    test("testWhisperTiny-decode", () {
      final decoded = tokenizer?.decode(List.from(dataset?["token_ids"]).cast());
      expect(decoded, dataset?["decoded_text"]);
    });

    test("testWhisperTiny-edgeCases", () {
      for (final edgeCase in edgeCases) {
        final encoded = tokenizer?.encode(edgeCase["input"]);
        expect(encoded, edgeCase["encoded"]["input_ids"]);
      }
    });

    test("testWhisperTiny-unknownToken", () {
      final tokenizerModel = (tokenizer as PreTrainedTokenizer).model;

      expect(tokenizerModel.unknownTokenId, unknownTokenId);
      expect(tokenizerModel.unknownTokenId, tokenizerModel.convertTokenToId("_this_token_does_not_exist_"));
      if (tokenizerModel.unknownTokenId != null) {
        expect(tokenizerModel.unknownToken, tokenizerModel.convertIdToToken(tokenizerModel.unknownTokenId!));
      } else {
        expect(tokenizerModel.unknownTokenId, isNull);
      }
    });
  });

  group("testT5Base", () {
    final hubModelName = "t5-base";
    final encodedSamplesFilename = "t5_base_encoded";
    final unknownTokenId = 2;
    Tokenizer? tokenizer;
    Map<String, dynamic>? dataset;
    List<Map<String, dynamic>> edgeCases = [];

    setUpAll(() async {
      final configuration = LanguageModelConfigurationFromHub.fromName(modelName: hubModelName, hubApi: hubApi);
      final tokenizerConfig = await configuration.tokenizerConfig;
      final tokenizerData = await configuration.tokenizerData;

      if (tokenizerConfig == null) {
        throw TokenizerError.tokenizerConfigNotFound();
      }

      tokenizer = AutoTokenizer.from(tokenizerConfig, tokenizerData);

      final file = io.File('test_assets/$encodedSamplesFilename.json');
      dataset = jsonDecode(await file.readAsString());

      final testFile = io.File('test_assets/tokenizer_tests.json');
      final Map<String, dynamic> testCase = jsonDecode(await testFile.readAsString());
      edgeCases = List.from(testCase[hubModelName] ?? []).cast();
    });

    test("testT5Base-tokenize", () {
      final tokenized = tokenizer?.tokenize(dataset?["text"]);
      expect(tokenized, dataset?["bpe_tokens"]);
    });

    test("testT5Base-encode", () {
      final encoded = tokenizer?.encode(dataset?["text"]);
      expect(encoded, dataset?["token_ids"]);
    });

    test("testT5Base-decode", () {
      final decoded = tokenizer?.decode(List.from(dataset?["token_ids"]).cast());
      expect(decoded, dataset?["decoded_text"]);
    });

    test("testT5Base-edgeCases", () {
      for (final edgeCase in edgeCases) {
        final encoded = tokenizer?.encode(edgeCase["input"]);
        expect(encoded, edgeCase["encoded"]["input_ids"]);
      }
    });

    test("testT5Base-unknownToken", () {
      final tokenizerModel = (tokenizer as PreTrainedTokenizer).model;

      expect(tokenizerModel.unknownTokenId, unknownTokenId);
      expect(tokenizerModel.unknownTokenId, tokenizerModel.convertTokenToId("_this_token_does_not_exist_"));
      if (tokenizerModel.unknownTokenId != null) {
        expect(tokenizerModel.unknownToken, tokenizerModel.convertIdToToken(tokenizerModel.unknownTokenId!));
      } else {
        expect(tokenizerModel.unknownTokenId, isNull);
      }
    });
  });

  group("testDistilbert", () {
    final hubModelName = "distilbert/distilbert-base-multilingual-cased";
    final encodedSamplesFilename = "distilbert_cased_encoded";
    final unknownTokenId = 100;
    Tokenizer? tokenizer;
    Map<String, dynamic>? dataset;
    List<Map<String, dynamic>> edgeCases = [];

    setUpAll(() async {
      final configuration = LanguageModelConfigurationFromHub.fromName(modelName: hubModelName, hubApi: hubApi);
      final tokenizerConfig = await configuration.tokenizerConfig;
      final tokenizerData = await configuration.tokenizerData;

      if (tokenizerConfig == null) {
        throw TokenizerError.tokenizerConfigNotFound();
      }

      tokenizer = AutoTokenizer.from(tokenizerConfig, tokenizerData);

      final file = io.File('test_assets/$encodedSamplesFilename.json');
      dataset = jsonDecode(await file.readAsString());

      final testFile = io.File('test_assets/tokenizer_tests.json');
      final Map<String, dynamic> testCase = jsonDecode(await testFile.readAsString());
      edgeCases = List.from(testCase[hubModelName] ?? []).cast();
    });

    test("testDistilbert-tokenize", () {
      final tokenized = tokenizer?.tokenize(dataset?["text"]);
      expect(tokenized, dataset?["bpe_tokens"]);
    });

    test("testDistilbert-encode", () {
      final encoded = tokenizer?.encode(dataset?["text"]);
      expect(encoded, dataset?["token_ids"]);
    });

    test("testDistilbert-decode", () {
      final decoded = tokenizer?.decode(List.from(dataset?["token_ids"]).cast());
      expect(decoded, dataset?["decoded_text"]);
    });

    test("testDistilbert-edgeCases", () {
      for (final edgeCase in edgeCases) {
        final encoded = tokenizer?.encode(edgeCase["input"]);
        expect(encoded, edgeCase["encoded"]["input_ids"]);
      }
    });

    test("testDistilbert-unknownToken", () {
      final tokenizerModel = (tokenizer as PreTrainedTokenizer).model;

      expect(tokenizerModel.unknownTokenId, unknownTokenId);
      expect(tokenizerModel.unknownTokenId, tokenizerModel.convertTokenToId("_this_token_does_not_exist_"));
      if (tokenizerModel.unknownTokenId != null) {
        expect(tokenizerModel.unknownToken, tokenizerModel.convertIdToToken(tokenizerModel.unknownTokenId!));
      } else {
        expect(tokenizerModel.unknownTokenId, isNull);
      }
    });
  });

  group("testBert", () {
    final hubModelName = "google-bert/bert-base-uncased";
    final encodedSamplesFilename = "bert_uncased_encoded";
    final unknownTokenId = 100;
    Tokenizer? tokenizer;
    Map<String, dynamic>? dataset;
    List<Map<String, dynamic>> edgeCases = [];

    setUpAll(() async {
      final configuration = LanguageModelConfigurationFromHub.fromName(modelName: hubModelName, hubApi: hubApi);
      final tokenizerConfig = await configuration.tokenizerConfig;
      final tokenizerData = await configuration.tokenizerData;

      if (tokenizerConfig == null) {
        throw TokenizerError.tokenizerConfigNotFound();
      }

      tokenizer = AutoTokenizer.from(tokenizerConfig, tokenizerData);

      final file = io.File('test_assets/$encodedSamplesFilename.json');
      dataset = jsonDecode(await file.readAsString());

      final testFile = io.File('test_assets/tokenizer_tests.json');
      final Map<String, dynamic> testCase = jsonDecode(await testFile.readAsString());
      edgeCases = List.from(testCase[hubModelName] ?? []).cast();
    });

    test("testBert-tokenize", () {
      final tokenized = tokenizer?.tokenize(dataset?["text"]);
      expect(tokenized, dataset?["bpe_tokens"]);
    });

    test("testBert-encode", () {
      final encoded = tokenizer?.encode(dataset?["text"]);
      expect(encoded, dataset?["token_ids"]);
    });

    test("testBert-decode", () {
      final decoded = tokenizer?.decode(List.from(dataset?["token_ids"]).cast());
      expect(decoded, dataset?["decoded_text"]);
    });

    test("testBert-edgeCases", () {
      for (final edgeCase in edgeCases) {
        final encoded = tokenizer?.encode(edgeCase["input"]);
        expect(encoded, edgeCase["encoded"]["input_ids"]);
      }
    });

    test("testBert-unknownToken", () {
      final tokenizerModel = (tokenizer as PreTrainedTokenizer).model;

      expect(tokenizerModel.unknownTokenId, unknownTokenId);
      expect(tokenizerModel.unknownTokenId, tokenizerModel.convertTokenToId("_this_token_does_not_exist_"));
      if (tokenizerModel.unknownTokenId != null) {
        expect(tokenizerModel.unknownToken, tokenizerModel.convertIdToToken(tokenizerModel.unknownTokenId!));
      } else {
        expect(tokenizerModel.unknownTokenId, isNull);
      }
    });
  });

  group("testGemma", () {
    final hubModelName = "pcuenq/gemma-tokenizer";
    final encodedSamplesFilename = "gemma_encoded";
    final unknownTokenId = 3;
    Tokenizer? tokenizer;
    Map<String, dynamic>? dataset;
    List<Map<String, dynamic>> edgeCases = [];

    setUpAll(() async {
      final configuration = LanguageModelConfigurationFromHub.fromName(modelName: hubModelName, hubApi: hubApi);
      final tokenizerConfig = await configuration.tokenizerConfig;
      final tokenizerData = await configuration.tokenizerData;

      if (tokenizerConfig == null) {
        throw TokenizerError.tokenizerConfigNotFound();
      }

      tokenizer = AutoTokenizer.from(tokenizerConfig, tokenizerData);

      final file = io.File('test_assets/$encodedSamplesFilename.json');
      dataset = jsonDecode(await file.readAsString());

      final testFile = io.File('test_assets/tokenizer_tests.json');
      final Map<String, dynamic> testCase = jsonDecode(await testFile.readAsString());
      edgeCases = List.from(testCase[hubModelName] ?? []).cast();
    });

    test("testGemma-tokenize", () {
      final tokenized = tokenizer?.tokenize(dataset?["text"]);
      expect(tokenized, dataset?["bpe_tokens"]);
    });

    test("testGemma-encode", () {
      final encoded = tokenizer?.encode(dataset?["text"]);
      expect(encoded, dataset?["token_ids"]);
    });

    test("testGemma-decode", () {
      final decoded = tokenizer?.decode(List.from(dataset?["token_ids"]).cast());
      expect(decoded, dataset?["decoded_text"]);
    });

    test("testGemma-edgeCases", () {
      for (final edgeCase in edgeCases) {
        final encoded = tokenizer?.encode(edgeCase["input"]);
        expect(encoded, edgeCase["encoded"]["input_ids"]);
      }
    });

    test("testGemma-unicodeEdgeCase", () {
      final cases = ["à" /* 0x61 0x300 */, "à" /* 0xe0 */];
      final result = [217138, 1305];

      for (final c in List.generate(cases.length, (i) => (cases[i], result[i]))) {
        final encoded = tokenizer?.encode(" ${c.$1}");
        expect(encoded, [2, c.$2]);
      }
    });

    test("testGemma-unknownToken", () {
      final tokenizerModel = (tokenizer as PreTrainedTokenizer).model;

      expect(tokenizerModel.unknownTokenId, unknownTokenId);
      expect(tokenizerModel.unknownTokenId, tokenizerModel.convertTokenToId("_this_token_does_not_exist_"));
      if (tokenizerModel.unknownTokenId != null) {
        expect(tokenizerModel.unknownToken, tokenizerModel.convertIdToToken(tokenizerModel.unknownTokenId!));
      } else {
        expect(tokenizerModel.unknownTokenId, isNull);
      }
    });
  });

  group("pretrained", () {
    test("testGemmaUnicode", () async {
      final tokenizer = await AutoTokenizer.fromPretrained("pcuenq/gemma-tokenizer");

      expect(((tokenizer as PreTrainedTokenizer).model as BPETokenizer).vocabCount, 256000);
    });

    test("testPhiSimple", () async {
      final tokenizer = await AutoTokenizer.fromPretrained("microsoft/phi-4");

      expect(
        tokenizer.encode("hello"),
        [15339],
      );
      expect(
        tokenizer.encode("hello world"),
        [15339, 1917],
      );
      expect(
        tokenizer.encode("<|im_start|>user<|im_sep|>Who are you?<|im_end|><|im_start|>assistant<|im_sep|>"),
        [100264, 882, 100266, 15546, 527, 499, 30, 100265, 100264, 78191, 100266],
      );
    });

    test("testDeepSeek", () async {
      final tokenizer = await AutoTokenizer.fromPretrained("deepseek-ai/DeepSeek-R1-Distill-Qwen-7B");

      expect(
        tokenizer.encode("Who are you?"),
        [151646, 15191, 525, 498, 30],
      );
    });

    test("testLlama", () async {
      final tokenizer = await AutoTokenizer.fromPretrained("coreml-projects/Llama-2-7b-chat-coreml");

      expect(
        tokenizer.encode("Who are you?"),
        [1, 11644, 526, 366, 29973],
      );
    });

    test("testLocalTokenizerFromPretrained", () async {
      final downloadDestination = io.Directory.systemTemp.createTempSync('hf-local-pretrained-tests-downloads');
      final hubApi = HubApi(downloadBase: downloadDestination.uri);
      final downloadedTo = await hubApi.snapshot(Repo(id: "pcuenq/gemma-tokenizer"));

      final tokenizer = await AutoTokenizer.fromModelFolder(downloadedTo) as PreTrainedTokenizer?;
      expect(tokenizer, isNotNull);

      await downloadDestination.delete(recursive: true);
    });

    test("testBertCased", () async {
      final tokenizer = await AutoTokenizer.fromPretrained("distilbert/distilbert-base-multilingual-cased");

      expect(
        tokenizer.encode("mąka"),
        [101, 181, 102075, 10113, 102],
      );
      expect(
        tokenizer.tokenize("Car"),
        ["Car"],
      );
    });

    test("testBertCasedResaved", () async {
      final tokenizer = await AutoTokenizer.fromPretrained("pcuenq/distilbert-base-multilingual-cased-tokenizer");

      expect(
        tokenizer.encode("mąka"),
        [101, 181, 102075, 10113, 102],
      );
    });

    test("testBertUncased", () async {
      final tokenizer = await AutoTokenizer.fromPretrained("google-bert/bert-base-uncased");

      expect(
        tokenizer.tokenize("mąka"),
        ["ma", "##ka"],
      );
      expect(
        tokenizer.encode("mąka"),
        [101, 5003, 2912, 102],
      );
      expect(
        tokenizer.tokenize("département"),
        ["depart", "##ement"],
      );
      expect(
        tokenizer.encode("département"),
        [101, 18280, 13665, 102],
      );
      expect(
        tokenizer.tokenize("Car"),
        ["car"],
      );

      expect(
        tokenizer.tokenize("€4"),
        ["€", "##4"],
      );
      expect(
        tokenizer.tokenize("test \$1 R2 #3 €4 £5 ¥6 ₣7 ₹8 ₱9 test"),
        ["test", "\$", "1", "r", "##2", "#", "3", "€", "##4", "£5", "¥", "##6", "[UNK]", "₹", "##8", "₱", "##9", "test"],
      );
    });

    test("testEncodeDecode", () async {
      final tokenizer = await AutoTokenizer.fromPretrained("google-bert/bert-base-uncased");

      final text = "l'eure";
      final tokenized = tokenizer.tokenize(text);
      expect(tokenized, ["l", "'", "eu", "##re"]);
      final encoded = tokenizer.encode(text);
      expect(encoded, [101, 1048, 1005, 7327, 2890, 102]);
      final decoded = tokenizer.decode(encoded, skipSpecialTokens: true);
      // Note: this matches the behaviour of the Python "slow" tokenizer, but the fast one produces "l ' eure"
      expect(decoded, "l'eure");
    });
  });
}
