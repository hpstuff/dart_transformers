import 'dart:convert';

import 'package:dart_transformers/hub/config.dart';
import 'package:dart_transformers/tokenizers/tokenizer.dart';
import 'package:dart_transformers/tokenizers/utils.dart';

typedef BytePair = (String, String);

class BPETokenizer extends PreTrainedTokenizerModel {
  final Map<int, String> bytes = bytesChar();
  final Map<BytePair, int> bpeRanks;
  final Map<String, int> tokensToIds;
  Map<int, String> get idsToTokens {
    return tokensToIds.invert();
  }

  int get vocabCount => tokensToIds.length;

  @override
  final String? bosToken;
  @override
  int? get bosTokenId {
    if (bosToken == null) {
      return null;
    }
    return tokensToIds[bosToken];
  }

  @override
  final String? eosToken;
  @override
  int? get eosTokenId {
    if (eosToken == null) {
      return null;
    }
    return tokensToIds[eosToken];
  }

  @override
  final String? unknownToken;
  @override
  int? get unknownTokenId => (unknownToken == null ? null : tokensToIds[unknownToken]);

  @override
  final bool fuseUnknownTokens;

  BPETokenizer.fromConfig(
    Config tokenizerConfig,
    Config tokenizerData,
    Map<String, int> addedTokens,
  )   : bpeRanks = _buildBpeRanks(_mergeFromConfig(tokenizerData['model']?['merges'])),
        tokensToIds = addedTokens.mergeWithConfig(tokenizerData['model']?['vocab']),
        bosToken = tokenizerConfig['bosToken']?.stringValue,
        eosToken = tokenizerConfig['eosToken']?.stringValue,
        unknownToken = tokenizerConfig['unkToken']?['content']?.stringValue ?? tokenizerConfig['unkToken']?.stringValue,
        fuseUnknownTokens = tokenizerConfig['fuseUnk']?.boolValue ?? false;

  static List<BytePair> _mergeFromConfig(Config? config) {
    try {
      return (config?.arrayValue?.map((e) => List.from(e).toList().cast<String>()).toList() ?? []).map((merge) => (merge[0], merge[1])).toList();
    } catch (e) {
      return (config?.arrayValue?.cast<String>() ?? []).map((e) => e.split(" ").toTuple).toList();
    }
  }

  static Map<BytePair, int> _buildBpeRanks(List<BytePair> merges) {
    final bpeRanks = <BytePair, int>{};
    for (var i = 0; i < merges.length; i++) {
      final merge = merges[i];
      final bp = merge;
      bpeRanks[bp] = i;
    }
    return bpeRanks;
  }

  @override
  int? convertTokenToId(String token) {
    return tokensToIds[token] ?? unknownTokenId;
  }

  @override
  String? convertIdToToken(int id) {
    return idsToTokens[id];
  }

  List<String> byteEncode(String text) {
    final regex = RegExp(r"'s|'t|'re|'ve|'m|'ll|'d| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+");
    final tokens = regex.allMatches(text).map((match) => match.group(0)!).toList();
    return tokens.map((token) {
      return token.codeUnits.map((unit) => bytes[unit]).join();
    }).toList();
  }

  List<String> hexaEncode(String text) {
    final regex = RegExp(r"'s|'t|'re|'ve|'m|'ll|'d| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+");
    final tokens = regex.allMatches(text).map((match) => match.group(0)!).toList();
    return tokens.expand((token) {
      return utf8.encode(token).map((byte) => '<0x${byte.toRadixString(16).toUpperCase().padLeft(2, "0")}>').toList();
    }).toList();
  }

  Set<BytePair> _getPairs(List<String> word) {
    final pairs = <BytePair>{};
    for (var i = 0; i < word.length - 1; i++) {
      pairs.add((word[i], word[i + 1]));
    }
    return pairs;
  }

  String bpe(String token) {
    if (token.length <= 1) {
      return token;
    }

    var word = token.split('');
    var pairs = _getPairs(word).toList();

    while (true) {
      final bigrams = pairs.where((bp) => bpeRanks.containsKey(bp)).toList();
      if (bigrams.isEmpty) {
        break;
      }
      final bigram = bigrams.reduce((bp1, bp2) => bpeRanks[bp1]! < bpeRanks[bp2]! ? bp1 : bp2);
      final first = bigram.$1;
      final second = bigram.$2;
      final newWord = <String>[];
      var i = 0;
      while (i < word.length) {
        final j = word.sublist(i).indexOf(first);
        if (j == -1) {
          newWord.addAll(word.sublist(i));
          break;
        }
        newWord.addAll(word.sublist(i, i + j));
        i += j;
        if (i < word.length - 1 && word[i] == first && word[i + 1] == second) {
          newWord.add(first + second);
          i += 2;
        } else {
          newWord.add(word[i]);
          i += 1;
        }
      }
      word = newWord;
      if (word.length == 1) {
        break;
      } else {
        pairs = _getPairs(word).toList();
      }
    }
    return word.join(' ');
  }

  @override
  List<String> tokenize(String text) {
    final tokens = <String>[];
    final bpeTokens = bpe(text).split(' ');
    for (final token in bpeTokens) {
      if (convertTokenToId(token) != unknownTokenId) {
        tokens.add(token);
      } else {
        tokens.addAll(hexaEncode(token));
      }
    }
    return tokens;
  }
}
