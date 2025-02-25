import 'package:dart_transformers/tokenizers/pre_tokenizer.dart';
import 'package:dart_transformers/tokenizers/tokenizer.dart';
import 'package:flutter_test/flutter_test.dart';

void main() {
  test('testPhiAddedTokens', () async {
    final tokenizer = await AutoTokenizer.fromPretrained('microsoft/Phi-3-mini-128k-instruct');
    final inputIds = tokenizer.encode('This is the <|end|>. My only friend, the <|end|>');
    expect(inputIds, [910, 338, 278, 29871, 32007, 29889, 1619, 871, 5121, 29892, 278, 29871, 32007]);

    final decoded = tokenizer.decode(inputIds);
    expect(decoded, 'This is the <|end|>. My only friend, the <|end|>');
  });

  test('testGemmaAddedTokens', () async {
    final tokenizer = await AutoTokenizer.fromPretrained('pcuenq/gemma-tokenizer');
    final inputIds = tokenizer.encode("This\n\nis\na\ntest.");
    expect(inputIds, [2, 1596, 109, 502, 108, 235250, 108, 2195, 235265]);

    final decoded = tokenizer.decode(inputIds);
    expect(decoded, '<bos>This\n\nis\na\ntest.');
  });

  test('testSplitWithCaptureGroups', () {
    final addedTokensRegexp = RegExp(r'(<\|end\|>)\s*|(<\|raw\|>)\s*');

    expect(
      'eating <|raw|> meat <|end|> That\'s all'.splitWithDelimiterBehavior(
        addedTokensRegexp,
        behavior: DelimiterBehavior.isolate,
      ),
      ['eating ', '<|raw|>', 'meat ', '<|end|>', 'That\'s all'],
    );

    expect(
      '<|raw|>'.splitWithDelimiterBehavior(
        addedTokensRegexp,
        behavior: DelimiterBehavior.isolate,
      ),
      ['<|raw|>'],
    );

    expect(
      'This string doesn\'t have those separators'.splitWithDelimiterBehavior(
        addedTokensRegexp,
        behavior: DelimiterBehavior.isolate,
      ),
      ['This string doesn\'t have those separators'],
    );

    expect(
      'start <|end|>'.splitWithDelimiterBehavior(
        addedTokensRegexp,
        behavior: DelimiterBehavior.isolate,
      ),
      ['start ', '<|end|>'],
    );

    expect(
      'start <|end|> '.splitWithDelimiterBehavior(
        addedTokensRegexp,
        behavior: DelimiterBehavior.isolate,
      ),
      ['start ', '<|end|>'],
    );

    expect(
      'start <|end|>       '.splitWithDelimiterBehavior(
        addedTokensRegexp,
        behavior: DelimiterBehavior.isolate,
      ),
      ['start ', '<|end|>'],
    );

    expect(
      'start <|end|>       for real'.splitWithDelimiterBehavior(
        addedTokensRegexp,
        behavior: DelimiterBehavior.isolate,
      ),
      ['start ', '<|end|>', 'for real'],
    );

    expect(
      '<|raw|><|end|>'.splitWithDelimiterBehavior(
        addedTokensRegexp,
        behavior: DelimiterBehavior.isolate,
      ),
      ['<|raw|>', '<|end|>'],
    );
  });
}
