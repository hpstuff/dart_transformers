import 'package:dart_transformers/hub/config.dart';
import 'package:dart_transformers/tokenizers/token_lattice.dart';
import 'package:dart_transformers/tokenizers/tokenizer.dart';
import 'package:dart_transformers/tokenizers/tokenizer_error.dart';
import 'package:dart_transformers/tokenizers/trie.dart';

class SentencePieceToken {
  final String token;
  final double score;

  SentencePieceToken({required this.token, required this.score});
}

var c = 0;

class UnigramTokenizer extends PreTrainedTokenizerModel {
  final List<SentencePieceToken> vocab;
  SentencePieceToken get unknownPiece => SentencePieceToken(token: vocab[unknownTokenId].token, score: minScore - 10);
  double get minScore => vocab.fold<double>(999, (partial, token) => partial < token.score ? partial : token.score);
  final Map<String, int> tokensToIds;
  @override
  final String bosToken = " ";
  @override
  int? get bosTokenId => tokensToIds[bosToken];
  @override
  final String? eosToken;
  @override
  int? get eosTokenId => eosToken != null ? tokensToIds[eosToken] : null;
  @override
  final bool fuseUnknownTokens = true;
  final Trie trie;

  @override
  String? get unknownToken => unknownPiece.token;
  double? get unknownTokenScore => unknownPiece.score;

  @override
  int unknownTokenId;

  UnigramTokenizer.fromConfig(
    Config tokenizerConfig,
    Config tokenizerData,
    Map<String, int> addedTokens,
  )   : vocab = _extractVocab(tokenizerData),
        tokensToIds = {},
        unknownTokenId = tokenizerData['model']?['unkId']?.intValue ?? 0,
        eosToken = tokenizerConfig['eosToken']?.stringValue,
        trie = Trie() {
    tokensToIds.addAll(_extractTokensToIds(vocab));
    trie.append(vocab.map((piece) => piece.token));
  }

  static List<SentencePieceToken> _extractVocab(Config tokenizerData) {
    final vocab = tokenizerData['model']?['vocab']?.arrayValue;
    if (vocab == null) {
      throw TokenizerError.malformedVocab();
    }
    return vocab.map((piece) {
      final token = piece[0] as String;
      final scoreValue = piece[1];
      final score = scoreValue is double ? scoreValue : (scoreValue as num).toDouble();
      return SentencePieceToken(token: token, score: score);
    }).toList();
  }

  static Map<String, int> _extractTokensToIds(List<SentencePieceToken> vocab) {
    return {for (var i = 0; i < vocab.length; i++) vocab[i].token: i};
  }

  @override
  int? convertTokenToId(String token) {
    return tokensToIds[token] ?? unknownTokenId;
  }

  @override
  String? convertIdToToken(int id) {
    return vocab[id].token;
  }

  @override
  List<String> tokenize(String text) {
    final lattice = TokenLattice(sentence: text, bosTokenId: bosTokenId ?? 0, eosTokenId: eosTokenId ?? 0);

    var beginPos = 0;
    while (beginPos < text.length) {
      final mblen = 1;
      var hasSingleNode = false;

      for (final token in trie.commonPrefixSearch(text.substring(beginPos))) {
        final tokenId = tokensToIds[token];
        if (tokenId == null) {
          throw TokenizerError("Fatal error", "Token $token not found in the vocabulary");
        }
        final tokenScore = vocab[tokenId].score;
        lattice.insert(pos: beginPos, length: token.length, score: tokenScore, tokenId: tokenId);
        if (!hasSingleNode && token.length == mblen) {
          hasSingleNode = true;
        }
      }
      if (!hasSingleNode) {
        lattice.insert(pos: beginPos, length: mblen, score: unknownPiece.score, tokenId: unknownTokenId);
      }
      beginPos += mblen;
    }

    return lattice.tokens;
  }
}
