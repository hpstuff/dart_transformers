import 'package:dart_transformers/tokenizers/pre_tokenizer.dart';
import 'package:flutter_test/flutter_test.dart';

void main() {
  test('testSplitBehaviorMergedWithPrevious', () {
    expect(
      "the-final--countdown".splitWithDelimiterBehavior("-", behavior: DelimiterBehavior.mergeWithPrevious),
      ["the-", "final-", "-", "countdown"],
    );

    expect(
      "the-final--countdown-".splitWithDelimiterBehavior("-", behavior: DelimiterBehavior.mergeWithPrevious),
      ["the-", "final-", "-", "countdown-"],
    );

    expect(
      "the-final--countdown--".splitWithDelimiterBehavior("-", behavior: DelimiterBehavior.mergeWithPrevious),
      ["the-", "final-", "-", "countdown-", "-"],
    );

    expect(
      "-the-final--countdown--".splitWithDelimiterBehavior("-", behavior: DelimiterBehavior.mergeWithPrevious),
      ["-", "the-", "final-", "-", "countdown-", "-"],
    );

    expect(
      "--the-final--countdown--".splitWithDelimiterBehavior("-", behavior: DelimiterBehavior.mergeWithPrevious),
      ["-", "-", "the-", "final-", "-", "countdown-", "-"],
    );
  });

  test('testSplitBehaviorMergedWithNext', () {
    expect(
      "the-final--countdown".splitWithDelimiterBehavior("-", behavior: DelimiterBehavior.mergeWithNext),
      ["the", "-final", "-", "-countdown"],
    );

    expect(
      "-the-final--countdown".splitWithDelimiterBehavior("-", behavior: DelimiterBehavior.mergeWithNext),
      ["-the", "-final", "-", "-countdown"],
    );

    expect(
      "--the-final--countdown".splitWithDelimiterBehavior("-", behavior: DelimiterBehavior.mergeWithNext),
      ["-", "-the", "-final", "-", "-countdown"],
    );

    expect(
      "--the-final--countdown-".splitWithDelimiterBehavior("-", behavior: DelimiterBehavior.mergeWithNext),
      ["-", "-the", "-final", "-", "-countdown", "-"],
    );
  });

  test('testSplitBehaviorOther', () {
    expect(
      "the-final--countdown".splitWithDelimiterBehavior("-", behavior: DelimiterBehavior.isolate),
      ["the", "-", "final", "-", "-", "countdown"],
    );

    expect(
      "the-final--countdown".splitWithDelimiterBehavior("-", behavior: DelimiterBehavior.remove),
      ["the", "final", "countdown"],
    );
  });
}
