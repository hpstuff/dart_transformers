import 'package:flutter_test/flutter_test.dart';
import 'package:dart_transformers/hub/config.dart';
import 'package:dart_transformers/tokenizers/normalizer.dart';

void main() {
  group('NormalizerTests', () {
    test('testLowercaseNormalizer', () {
      final testCases = [
        ('Café', 'café'),
        ('François', 'françois'),
        ('Ωmega', 'ωmega'),
        ('über', 'über'),
        ('háček', 'háček'),
        ('Häagen-Dazs', 'häagen-dazs'),
        ('你好!', '你好!'),
        ('𝔄𝔅ℭ⓵⓶⓷︷,︸,i⁹,i₉,㌀,¼', '𝔄𝔅ℭ⓵⓶⓷︷,︸,i⁹,i₉,㌀,¼'),
        ('\u{00C5}', '\u{00E5}'),
      ];

      for (var (arg, res) in testCases) {
        final normalizer = LowercaseNormalizer(Config({}));
        expect(normalizer.normalize(arg), res);
      }

      final config = Config({'type': NormalizerType.lowercase.name});
      expect(NormalizerFactory.fromConfig(config) is LowercaseNormalizer, true);
    });

    test('testNFDNormalizer', () {
      final testCases = [
        ('caf\u{65}\u{301}', 'cafe\u{301}'),
        ('François', 'François'),
        ('Ωmega', 'Ωmega'),
        ('über', 'über'),
        ('háček', 'háček'),
        ('Häagen-Dazs', 'Häagen-Dazs'),
        ('你好!', '你好!'),
        ('𝔄𝔅ℭ⓵⓶⓷︷,︸,i⁹,i₉,㌀,¼', '𝔄𝔅ℭ⓵⓶⓷︷,︸,i⁹,i₉,㌀,¼'),
        ('\u{00C5}', '\u{0041}\u{030A}'),
      ];

      for (var (arg, res) in testCases) {
        final normalizer = NFDNormalizer(Config({}));
        expect(normalizer.normalize(arg), res);
      }

      final config = Config({'type': NormalizerType.nfd.name});
      expect(NormalizerFactory.fromConfig(config) is NFDNormalizer, true);
    });

    test('testNFCNormalizer', () {
      final testCases = [
        ("café", "café"),
        ("François", "François"),
        ("Ωmega", "Ωmega"),
        ("über", "über"),
        ("háček", "háček"),
        ("Häagen-Dazs", "Häagen-Dazs"),
        ("你好!", "你好!"),
        ("𝔄𝔅ℭ⓵⓶⓷︷,︸,i⁹,i₉,㌀,¼", "𝔄𝔅ℭ⓵⓶⓷︷,︸,i⁹,i₉,㌀,¼"),
        ("\u{00C5}", "\u{00C5}"),
      ];

      for (var (arg, res) in testCases) {
        final normalizer = NFCNormalizer(Config({}));
        expect(normalizer.normalize(arg), res);
      }

      final config = Config({'type': NormalizerType.nfc.name});
      expect(NormalizerFactory.fromConfig(config) is NFCNormalizer, true);
    });

    test('testNFKDNormalizer', () {
      final testCases = [
        ("café", "cafe\u{301}"),
        ("François", "François"),
        ("Ωmega", "Ωmega"),
        ("über", "über"),
        ("háček", "háček"),
        ("Häagen-Dazs", "Häagen-Dazs"),
        ("你好!", "你好!"),
        ("𝔄𝔅ℭ⓵⓶⓷︷,︸,i⁹,i₉,㌀,¼", "ABC⓵⓶⓷{,},i9,i9,アパート,1⁄4"),
        ("\u{00C5}", "Å"),
      ];

      for (var (arg, res) in testCases) {
        final normalizer = NFKDNormalizer(Config({}));
        expect(normalizer.normalize(arg), res);
      }

      final config = Config({'type': NormalizerType.nfkd.name});
      expect(NormalizerFactory.fromConfig(config) is NFKDNormalizer, true);
    });

    test('testNFKCNormalizer', () {
      final testCases = [
        ("café", "café"),
        ("François", "François"),
        ("Ωmega", "Ωmega"),
        ("über", "über"),
        ("háček", "háček"),
        ("Häagen-Dazs", "Häagen-Dazs"),
        ("你好!", "你好!"),
        ("𝔄𝔅ℭ⓵⓶⓷︷,︸,i⁹,i₉,㌀,¼", "ABC⓵⓶⓷{,},i9,i9,アパート,1⁄4"),
        ("\u{00C5}", "\u{00C5}"),
      ];

      for (var (arg, res) in testCases) {
        final normalizer = NFKCNormalizer(Config({}));
        expect(normalizer.normalize(arg), res);
      }

      final config = Config({'type': NormalizerType.nfkc.name});
      expect(NormalizerFactory.fromConfig(config) is NFKCNormalizer, true);
    });

    test('testStripAccents', () {
      final testCases = [
        ("département", "departement"),
      ];

      final normalizer = BertNormalizer(Config({"stripAccents": true}));
      for (var (arg, res) in testCases) {
        expect(normalizer.normalize(arg), res);
      }
    });

    test('testBertNormalizer', () {
      final testCases = [
        ("Café", "café"),
        ("François", "françois"),
        ("Ωmega", "ωmega"),
        ("über", "über"),
        ("háček", "háček"),
        ("Häagen\tDazs", "häagen dazs"),
        ("你好!", " 你  好 !"),
        ("𝔄𝔅ℭ⓵⓶⓷︷,︸,i⁹,i₉,㌀,¼", "𝔄𝔅ℭ⓵⓶⓷︷,︸,i⁹,i₉,㌀,¼"),
        ("\u{00C5}", "\u{00E5}"),
      ];

      for (var (arg, res) in testCases) {
        final normalizer = BertNormalizer(Config({"stripAccents": false}));
        expect(normalizer.normalize(arg), res);
      }

      final config = Config({'type': NormalizerType.bert.name});
      expect(NormalizerFactory.fromConfig(config) is BertNormalizer, true);
    });

    test('testBertNormalizerDefaults', () {
      final testCases = [
        ("Café", "cafe"),
        ("François", "francois"),
        ("Ωmega", "ωmega"),
        ("über", "uber"),
        ("háček", "hacek"),
        ("Häagen\tDazs", "haagen dazs"),
        ("你好!", " 你  好 !"),
        ("𝔄𝔅ℭ⓵⓶⓷︷,︸,i⁹,i₉,㌀,¼", "𝔄𝔅ℭ⓵⓶⓷︷,︸,i⁹,i₉,㌀,¼"),
        ("Å", "a"),
      ];

      for (var (arg, res) in testCases) {
        final normalizer = BertNormalizer(Config({}));
        expect(normalizer.normalize(arg), res);
      }

      final config = Config({'type': NormalizerType.bert.name});
      expect(NormalizerFactory.fromConfig(config) is BertNormalizer, true);
    });

    test('testPrecompiledNormalizer', () {
      final testCases = [
        ("café", "café"),
        ("François", "François"),
        ("Ωmega", "Ωmega"),
        ("über", "über"),
        ("háček", "háček"),
        ("Häagen-Dazs", "Häagen-Dazs"),
        ("你好!", "你好!"),
        ("𝔄𝔅ℭ⓵⓶⓷︷,︸,i⁹,i₉,㌀,¼", "ABC⓵⓶⓷{,},i9,i9,アパート,1⁄4"),
        ("\u00C5", "\u00C5"),
        ("™\u001eg", "TMg"),
        ("full-width～tilde", "full-width～tilde"),
      ];

      for (var (arg, res) in testCases) {
        final normalizer = PrecompiledNormalizer(Config({}));
        expect(normalizer.normalize(arg), res);
      }

      final config = Config({'type': NormalizerType.precompiled.name});
      expect(NormalizerFactory.fromConfig(config) is PrecompiledNormalizer, true);
    });

    test('testStripAccentsINormalizer', () {
      final testCases = [
        ("café", "café"),
        ("François", "François"),
        ("Ωmega", "Ωmega"),
        ("über", "über"),
        ("háček", "háček"),
        ("Häagen-Dazs", "Häagen-Dazs"),
        ("你好!", "你好!"),
        ("𝔄𝔅ℭ⓵⓶⓷︷,︸,i⁹,i₉,㌀,¼", "ABC⓵⓶⓷{,},i9,i9,アパート,1⁄4"),
        ("\u00C5", "\u00C5"),
      ];

      for (var (arg, res) in testCases) {
        final normalizer = StripAccentsNormalizer(Config({}));
        expect(normalizer.normalize(arg), res);
      }

      final config = Config({'type': NormalizerType.stripAccents.name});
      expect(NormalizerFactory.fromConfig(config) is StripAccentsNormalizer, true);
    });

    test('testStripNormalizer', () {
      final testCases = [
        ("  hello  ", "hello", true, true),
        ("  hello  ", "hello  ", true, false),
        ("  hello  ", "  hello", false, true),
        ("  hello  ", "  hello  ", false, false),
        ("\t\nHello\t\n", "Hello", true, true),
        ("   ", "", true, true),
        ("", "", true, true),
      ];

      for (var (arg, res, left, right) in testCases) {
        final normalizer = StripNormalizer(Config({
          "type": NormalizerType.strip.name,
          "stripLeft": left,
          "stripRight": right,
        }));
        expect(normalizer.normalize(arg), res);
      }

      final config = Config({'type': NormalizerType.strip.name});
      expect(NormalizerFactory.fromConfig(config) is StripNormalizer, true);
    });
  });
}
