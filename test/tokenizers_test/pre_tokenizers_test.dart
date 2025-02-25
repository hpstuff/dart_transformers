import 'package:dart_transformers/hub/config.dart';
import 'package:dart_transformers/tokenizers/pre_tokenizer.dart';
import 'package:flutter_test/flutter_test.dart';

void main() {
  TestWidgetsFlutterBinding.ensureInitialized();
  group('PreTokenizerTests', () {
    test('testWhitespacePreTokenizer', () {
      final preTokenizer = WhitespacePreTokenizer(Config({}));

      expect(
        preTokenizer.preTokenize("Hey friend!"),
        ["Hey", "friend!"],
      );
      expect(
        preTokenizer.preTokenize("Hey friend!     How are you?!?"),
        ["Hey", "friend!", "How", "are", "you?!?"],
      );
      expect(
        preTokenizer.preTokenize("   Hey,    friend,    what's up?  "),
        ["Hey,", "friend,", "what's", "up?"],
      );
    });

    test('testPunctuationPreTokenizer', () {
      final preTokenizer = PunctuationPreTokenizer(Config({}));

      expect(
        preTokenizer.preTokenize("Hey friend!"),
        ["Hey friend", "!"],
      );
      expect(
        preTokenizer.preTokenize("Hey friend!     How are you?!?"),
        ["Hey friend", "!", "     How are you", "?!?"],
      );
      expect(
        preTokenizer.preTokenize("   Hey,    friend,    what's up?  "),
        ["   Hey", ",", "    friend", ",", "    what", "'", "s up", "?", "  "],
      );
    });

    test('testByteLevelPreTokenizer', () {
      final preTokenizer1 = ByteLevelPreTokenizer(Config({}));

      expect(
        preTokenizer1.preTokenize("Hey friend!"),
        ["Hey", "Ġfriend", "!"],
      );
      expect(
        preTokenizer1.preTokenize("Hey friend!     How are you?!?"),
        ["Hey", "Ġfriend", "!", "ĠĠĠĠ", "ĠHow", "Ġare", "Ġyou", "?!?"],
      );
      expect(
        preTokenizer1.preTokenize("   Hey,    friend,    what's up?  "),
        ["ĠĠ", "ĠHey", ",", "ĠĠĠ", "Ġfriend", ",", "ĠĠĠ", "Ġwhat", "'s", "Ġup", "?", "ĠĠ"],
      );

      final preTokenizer2 = ByteLevelPreTokenizer(
        Config({"addPrefixSpace": true}),
      );

      expect(
        preTokenizer2.preTokenize("Hey friend!"),
        ["ĠHey", "Ġfriend", "Ġ!"],
      );
      expect(
        preTokenizer2.preTokenize("Hey friend!     How are you?!?"),
        ["ĠHey", "Ġfriend", "Ġ!", "ĠĠĠĠ", "ĠHow", "Ġare", "Ġyou", "Ġ?!?"],
      );
      expect(
        preTokenizer2.preTokenize("   Hey,    friend,    what's up?  "),
        ["ĠĠ", "ĠHey", "Ġ,", "ĠĠĠ", "Ġfriend", "Ġ,", "ĠĠĠ", "Ġwhat", "Ġ's", "Ġup", "Ġ?", "ĠĠ"],
      );

      final preTokenizer3 = ByteLevelPreTokenizer(
        Config({"useRegex": false}),
      );

      expect(
        preTokenizer3.preTokenize("Hey friend!"),
        ["HeyĠfriend!"],
      );
      expect(
        preTokenizer3.preTokenize("Hey friend!     How are you?!?"),
        ["HeyĠfriend!ĠĠĠĠĠHowĠareĠyou?!?"],
      );
      expect(
        preTokenizer3.preTokenize("   Hey,    friend,    what's up?  "),
        ["ĠĠĠHey,ĠĠĠĠfriend,ĠĠĠĠwhat'sĠup?ĠĠ"],
      );
    });

    test('testDigitsPreTokenizer', () {
      final preTokenizer1 = DigitsPreTokenizer(Config({}));

      expect(
        preTokenizer1.preTokenize("1 12 123! 1234abc"),
        ["1", " ", "12", " ", "123", "! ", "1234", "abc"],
      );

      final preTokenizer2 = DigitsPreTokenizer(
        Config({"individualDigits": true}),
      );

      expect(
        preTokenizer2.preTokenize("1 12 123! 1234abc"),
        ["1", " ", "1", "2", " ", "1", "2", "3", "! ", "1", "2", "3", "4", "abc"],
      );
    });

    test('testSplitPreTokenizer', () {
      final preTokenizer1 = SplitPreTokenizer(
        Config({
          "pattern": {"String": " "}
        }),
      );
      expect(
        preTokenizer1.preTokenize("Hey friend!"),
        ["Hey", " ", "friend!"],
      );
      expect(
        preTokenizer1.preTokenize("Hey friend!     How are you?!?"),
        ["Hey", " ", "friend!", " ", " ", " ", " ", " ", "How", " ", "are", " ", "you?!?"],
      );
      expect(
        preTokenizer1.preTokenize("   Hey,    friend,    what's up?  "),
        [" ", " ", " ", "Hey,", " ", " ", " ", " ", "friend,", " ", " ", " ", " ", "what's", " ", "up?", " ", " "],
      );

      final preTokenizer2 = SplitPreTokenizer(Config({
        "pattern": {"Regex": "\\s"}
      }));
      expect(
        preTokenizer2.preTokenize("Hey friend!"),
        ["Hey", " ", "friend!"],
      );
      expect(
        preTokenizer2.preTokenize("Hey friend!     How are you?!?"),
        ["Hey", " ", "friend!", " ", " ", " ", " ", " ", "How", " ", "are", " ", "you?!?"],
      );
      expect(
        preTokenizer2.preTokenize("   Hey,    friend,    what's up?  "),
        [" ", " ", " ", "Hey,", " ", " ", " ", " ", "friend,", " ", " ", " ", " ", "what's", " ", "up?", " ", " "],
      );

      final preTokenizer3 = SplitPreTokenizer(
        Config({
          "pattern": {
            "Regex": r"(i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}{1,3}| ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+",
          },
          "invert": true,
        }),
      );
      expect(
        preTokenizer3.preTokenize("Hello"),
        ["Hello"],
      );

      expect(
        preTokenizer3.preTokenize("Hey friend!"),
        ["Hey", " friend", "!"],
      );
      expect(
        preTokenizer3.preTokenize("Hey friend!     How are you?!?"),
        ["Hey", " friend", "!", "    ", " How", " are", " you", "?!?"],
      );
    });

    test('testMetaspacePreTokenizer', () {
      final preTokenizer = MetaspacePreTokenizer(
        Config({
          "addPrefixSpace": true,
          "replacement": "▁",
          "prependScheme": "always",
        }),
      );

      final text = "Hey my friend <s>how▁are you";
      final tokens = text
          .splitBy('<s>', includeSeparators: true)
          .expand(
            (section) => preTokenizer.preTokenize(section),
          )
          .toList();

      expect(
        tokens,
        ["▁Hey", "▁my", "▁friend", "▁", "▁<s>", "▁how", "▁are", "▁you"],
      );
    });

    test('testBertPreTokenizer', () {
      final preTokenizer1 = BertPreTokenizer(Config({}));
      expect(
        preTokenizer1.preTokenize("Hey friend!"),
        ["Hey", "friend", "!"],
      );
      expect(
        preTokenizer1.preTokenize("Hey friend!     How are you?!?"),
        ["Hey", "friend", "!", "How", "are", "you", "?", "!", "?"],
      );
      expect(
        preTokenizer1.preTokenize("   Hey,    friend ,    what's up?  "),
        ["Hey", ",", "friend", ",", "what", "'", "s", "up", "?"],
      );
      expect(
        preTokenizer1.preTokenize("   Hey,    friend ,  0 99  what's up?  "),
        ["Hey", ",", "friend", ",", "0", "99", "what", "'", "s", "up", "?"],
      );
    });
  });
}
