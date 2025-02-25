import 'dart:convert';
import 'dart:io';

import 'package:dart_transformers/tokenizers/bert_tokenizer.dart';
import 'package:flutter_test/flutter_test.dart';

import '../squad_database.dart';

void main() {
  TestWidgetsFlutterBinding.ensureInitialized();
  group('BertTokenizer Tests', () {
    late BertTokenizer bertTokenizer;

    setUp(() async {
      final file = File('test_assets/bert-vocab.txt');
      final vocabTxt = await file.readAsString();
      final tokens = vocabTxt.split('\n');
      final vocab = <String, int>{};
      for (var i = 0; i < tokens.length; i++) {
        vocab[tokens[i]] = i;
      }
      bertTokenizer = BertTokenizer(vocab: vocab);
    });

    test('Basic Tokenizer', () {
      final basicTokenizer = BasicTokenizer();

      final text = "Brave gaillard, d'où [UNK] êtes vous?";
      final tokens = ["brave", "gaillard", ",", "d", "'", "ou", "[UNK]", "etes", "vous", "?"];

      expect(basicTokenizer.tokenize(text), containsAllInOrder(tokens));

      expect(["foo", "bar"], containsAllInOrder(["foo", "bar"]));
    });

    test('Full Basic Tokenizer', () async {
      final file = File('test_assets/basic_tokenized_questions.json');
      final jsonString = await file.readAsString();
      final List<dynamic> sampleTokens = jsonDecode(jsonString);

      final basicTokenizer = BasicTokenizer();

      final examples = await Squad.getExamples();

      expect(sampleTokens.length, examples.length);

      for (var i = 0; i < examples.length; i++) {
        final example = examples[i];
        final output = basicTokenizer.tokenize(example.question);
        expect(output, containsAllInOrder(sampleTokens[i]));
      }
    });

    test('Full Bert Tokenizer', () async {
      final file = File('test_assets/tokenized_questions.json');
      final jsonString = await file.readAsString();
      final List<dynamic> sampleTokens = jsonDecode(jsonString);

      final tokenizer = bertTokenizer;

      final examples = await Squad.getExamples();

      expect(sampleTokens.length, examples.length);

      for (var i = 0; i < examples.length; i++) {
        final example = examples[i];
        final output = tokenizer.tokenizeToIds(example.question);
        expect(output, containsAllInOrder(sampleTokens[i]));
      }
    });

    test('Mixed Chinese English Tokenization', () {
      final tokenizer = bertTokenizer;
      final text = "你好，世界！Hello, world!";
      final expectedTokens = ["[UNK]", "[UNK]", "，", "世", "[UNK]", "！", "hello", ",", "world", "!"];
      final tokens = tokenizer.tokenize(text);

      expect(tokens, expectedTokens);
    });

    test('Pure Chinese Tokenization', () {
      final tokenizer = bertTokenizer;
      final text = "明日，大家上山看日出。";
      final expectedTokens = ["明", "日", "，", "大", "家", "上", "山", "[UNK]", "日", "出", "。"];
      final tokens = tokenizer.tokenize(text);

      expect(tokens, expectedTokens);
    });

    test('Chinese With Numerals Tokenization', () {
      final tokenizer = bertTokenizer;
      final text = "2020年奥运会在东京举行。";
      final expectedTokens = ["2020", "年", "[UNK]", "[UNK]", "会", "[UNK]", "[UNK]", "京", "[UNK]", "行", "。"];
      final tokens = tokenizer.tokenize(text);

      expect(tokens, expectedTokens);
    });

    test('Chinese With Special Tokens', () {
      final tokenizer = bertTokenizer;
      final text = "[CLS] 机器学习是未来。 [SEP]";
      final expectedTokens = ["[CLS]", "[UNK]", "[UNK]", "学", "[UNK]", "[UNK]", "[UNK]", "[UNK]", "。", "[SEP]"];
      final tokens = tokenizer.tokenize(text);

      expect(tokens, expectedTokens);
    });

    test('Performance Example', () {
      final tokenizer = bertTokenizer;

      final stopwatch = Stopwatch()..start();
      tokenizer.tokenizeToIds("Brave gaillard, d'où [UNK] êtes vous?");
      stopwatch.stop();
    });

    test('Wordpiece Detokenizer', () async {
      final file = File('test_assets/question_tokens.json');
      final jsonString = await file.readAsString();
      final List<dynamic> jsonData = jsonDecode(jsonString);

      final tokenizer = bertTokenizer;

      for (var question in jsonData) {
        expect(
          question["basic"].join(' '),
          tokenizer.convertWordpieceToBasicTokenList(
            (question["wordpiece"] as List<dynamic>).cast<String>(),
          ),
        );
      }
    });

    test('Encoder Decoder', () {
      final text = """
      Wake up (Wake up)
      Grab a brush and put a little makeup
      Hide your scars to fade away the shakeup (Hide the scars to fade away the shakeup)
      Why'd you leave the keys upon the table?
      Here you go, create another fable, you wanted to
      Grab a brush and put a little makeup, you wanted to
      Hide the scars to fade away the shakeup, you wanted to
      Why'd you leave the keys upon the table? You wanted to
      """;

      final decoded = """
      wake up ( wake up )
      grab a brush and put a little makeup
      hide your scars to fade away the shakeup ( hide the scars to fade away the shakeup )
      why ' d you leave the keys upon the table ?
      here you go , create another fable , you wanted to
      grab a brush and put a little makeup , you wanted to
      hide the scars to fade away the shakeup , you wanted to
      why ' d you leave the keys upon the table ? you wanted to
      """;

      final tokenizer = bertTokenizer;
      for (var i = 0; i < text.split('\n').length; i++) {
        final line = text.split('\n')[i].trim();
        final expected = decoded.split('\n')[i].trim();
        final encoded = tokenizer.encode(line);
        final decodedLine = tokenizer.decode(encoded);
        expect(decodedLine, expected);
      }
    });
  });
}
