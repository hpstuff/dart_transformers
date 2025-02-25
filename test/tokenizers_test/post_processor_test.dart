import 'package:dart_transformers/hub/config.dart';
import 'package:dart_transformers/tokenizers/post_processor.dart';
import 'package:dart_transformers/tokenizers/pre_tokenizer.dart';
import 'package:flutter_test/flutter_test.dart';

void main() {
  group('PostProcessorTests', () {
    test('testRobertaProcessing', () {
      final testCases = [
        // Should keep spaces; uneven spaces; ignore `addPrefixSpace`.
        (
          Config({
            'cls': [0, '[HEAD]'],
            'sep': [0, '[END]'],
            'trimOffset': false,
            'addPrefixSpace': true,
          }),
          [' The', ' sun', 'sets ', '  in  ', '   the  ', 'west'],
          null,
          ['[HEAD]', ' The', ' sun', 'sets ', '  in  ', '   the  ', 'west', '[END]']
        ),
        // Should leave only one space around each token.
        (
          Config({
            'cls': [0, '[START]'],
            'sep': [0, '[BREAK]'],
            'trimOffset': true,
            'addPrefixSpace': true,
          }),
          [' The ', ' sun', 'sets ', '  in ', '  the    ', 'west'],
          null,
          ['[START]', ' The ', ' sun', 'sets ', ' in ', ' the ', 'west', '[BREAK]']
        ),
        // Should ignore empty tokens pair.
        (
          Config({
            'cls': [0, '[START]'],
            'sep': [0, '[BREAK]'],
            'trimOffset': true,
            'addPrefixSpace': true,
          }),
          [' The ', ' sun', 'sets ', '  in ', '  the    ', 'west'],
          [],
          ['[START]', ' The ', ' sun', 'sets ', ' in ', ' the ', 'west', '[BREAK]']
        ),
        // Should trim all whitespace.
        (
          Config({
            'cls': [0, '[CLS]'],
            'sep': [0, '[SEP]'],
            'trimOffset': true,
            'addPrefixSpace': false,
          }),
          [' The ', ' sun', 'sets ', '  in ', '  the    ', 'west'],
          null,
          ['[CLS]', 'The', 'sun', 'sets', 'in', 'the', 'west', '[SEP]']
        ),
        // Should add tokens.
        (
          Config({
            'cls': [0, '[CLS]'],
            'sep': [0, '[SEP]'],
            'trimOffset': true,
            'addPrefixSpace': true,
          }),
          [' The ', ' sun', 'sets ', '  in ', '  the    ', 'west'],
          ['.', 'The', ' cat ', '   is ', ' sitting  ', ' on', 'the ', 'mat'],
          [
            '[CLS]',
            ' The ',
            ' sun',
            'sets ',
            ' in ',
            ' the ',
            'west',
            '[SEP]',
            '[SEP]',
            '.',
            'The',
            ' cat ',
            ' is ',
            ' sitting ',
            ' on',
            'the ',
            'mat',
            '[SEP]'
          ]
        ),
        (
          Config({
            'cls': [0, '[CLS]'],
            'sep': [0, '[SEP]'],
            'trimOffset': true,
            'addPrefixSpace': true,
          }),
          [' 你 ', ' 好 ', ','],
          [' 凯  ', '  蒂  ', '!'],
          ['[CLS]', ' 你 ', ' 好 ', ',', '[SEP]', '[SEP]', ' 凯 ', ' 蒂 ', '!', '[SEP]']
        ),
      ];

      for (var (config, tokens, tokensPair, result) in testCases) {
        final processor = RobertaProcessing.fromConfig(config);
        final output = processor.postProcess(tokens, tokensPair: tokensPair?.cast());
        expect(output, equals(result));
      }
    });

    test('testSplitWithDelimiterBehavior', () {
      final text = "the-final--countdown";

      expect(
        text.splitWithDelimiterBehavior("-", behavior: DelimiterBehavior.remove),
        ["the", "final", "countdown"],
      );

      expect(
        text.splitWithDelimiterBehavior("-", behavior: DelimiterBehavior.isolate),
        ["the", "-", "final", "-", "-", "countdown"],
      );

      expect(
        text.splitWithDelimiterBehavior("-", behavior: DelimiterBehavior.mergeWithPrevious),
        ["the-", "final-", "-", "countdown"],
      );

      expect(
        text.splitWithDelimiterBehavior("-", behavior: DelimiterBehavior.mergeWithNext),
        ["the", "-final", "-", "-countdown"],
      );
    });

    test('testSplitWithMergeWithNext', () {
      final text = "▁à";

      expect(
        text.splitWithDelimiterBehavior("▁", behavior: DelimiterBehavior.mergeWithNext),
        ["▁à"],
      );
    });
  });
}
