import 'package:dart_transformers/hub/config.dart';
import 'package:dart_transformers/tokenizers/decoder.dart';
import 'package:flutter_test/flutter_test.dart';

void main() {
  group('DecoderTests', () {
    // https://github.com/huggingface/tokenizers/pull/1357
    test('testMetaspaceDecoder', () {
      final decoder = MetaspaceDecoder(Config({
        'add_prefix_space': true,
        'replacement': '▁',
      }));

      final tokens = ['▁Hey', '▁my', '▁friend', '▁', '▁<s>', '▁how', '▁are', '▁you'];
      final decoded = decoder.decode(tokens);

      expect(
        decoded,
        equals(['Hey', ' my', ' friend', ' ', ' <s>', ' how', ' are', ' you']),
      );
    });

    test('testWordPieceDecoder', () {
      final config = Config({'prefix': '##', 'cleanup': true});
      final decoder = WordPieceDecoder(config);

      final testCases = [
        (['##inter', '##national', '##ization'], '##internationalization'),
        (['##auto', '##mat', '##ic', 'transmission'], '##automatic transmission'),
        (['who', 'do', "##n't", 'does', "n't", "can't"], "who don't doesn't can't"),
        (['##un', '##believ', '##able', '##fa', '##ntastic'], '##unbelievablefantastic'),
        (['this', 'is', 'un', '##believ', '##able', 'fa', '##ntastic'], 'this is unbelievable fantastic'),
        (['The', '##quick', '##brown', 'fox'], 'Thequickbrown fox'),
      ];

      for (var (tokens, result) in testCases) {
        final output = decoder.decode(tokens);
        expect(output.join(), equals(result));
      }
    });
  });
}
