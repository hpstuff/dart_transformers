import 'dart:convert';

import 'package:dart_transformers/tokenizers/utils.dart';

class TrieNode {
  Map<String, TrieNode> children = {};
  bool isEndOfWord = false;

  toJson() {
    return {
      "children": children,
      "isEndOfWord": isEndOfWord,
    };
  }

  @override
  String toString() {
    return jsonEncode(children, toEncodable: (dynamic object) {
      if (object is TrieNode) {
        return object.toJson();
      }
      return object;
    });
  }
}

class Trie {
  final TrieNode root = TrieNode();

  void insert(String word) {
    TrieNode current = root;
    for (final char in word.toIterable()) {
      if (!current.children.containsKey(char)) {
        current.children[char] = TrieNode();
      }
      current = current.children[char]!;
    }
    current.isEndOfWord = true;
  }

  void append(Iterable<String> container) {
    for (var sequence in container) {
      insert(sequence);
    }
  }

  TrieNode? get(String word) {
    TrieNode? node = root;
    for (final char in word.toIterable()) {
      if (node == null || !node.children.containsKey(char)) {
        return null;
      }
      node = node.children[char];
    }
    return node;
  }

  List<String> commonPrefixSearch(String word) {
    final List<String> leaves = [];
    TrieNode? node = root;
    for (var i = 0; i < word.length; i++) {
      final char = word[i];
      if (node == null || !node.children.containsKey(char)) {
        break;
      }
      node = node.children[char];
      if (node!.isEndOfWord) {
        leaves.add(word.substring(0, i + 1));
      }
    }
    return leaves;
  }

  @override
  String toString() {
    return jsonEncode({
      "root": root,
    }, toEncodable: (dynamic object) {
      if (object is TrieNode) {
        return object.toJson();
      }
      return object;
    });
  }
}
