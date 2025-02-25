import 'package:dart_transformers/tokenizers/trie.dart';
import 'package:flutter_test/flutter_test.dart';

void main() {
  test('testTrieBuilding', () {
    final trie = Trie();
    trie.insert("cat");
    trie.insert("carp");
    trie.insert("car");

    expect(trie.root.children.length, 1);

    final c = trie.get("c");
    expect(c, isNotNull);
    expect(c!.children.length, 1);

    final ca = trie.get("ca");
    expect(ca, isNotNull);
    expect(ca!.children.length, 2);

    final car = trie.get("car");
    expect(car, isNotNull);
    expect(car!.isEndOfWord, isTrue);
    expect(ca.isEndOfWord, isFalse);
    expect(trie.get("card"), isNull);
  });

  test('testTrieCommonPrefixSearch', () {
    final trie = Trie();
    trie.insert("cat");
    trie.insert("carp");
    trie.insert("car");

    final leaves = trie.commonPrefixSearch("carpooling");
    expect(leaves, ["car", "carp"]);
  });

  test('testTrieCommonPrefixSearchIterator', () {
    final trie = Trie();
    trie.insert("cat");
    trie.insert("carp");
    trie.insert("car");

    final expected = {"car", "carp"};
    final leaves = trie.commonPrefixSearch("carpooling");
    for (final leaf in leaves) {
      expect(expected.contains(leaf), isTrue);
      expected.remove(leaf);
    }
    expect(expected.isEmpty, isTrue);
  });
}
