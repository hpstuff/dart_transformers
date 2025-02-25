import 'package:dart_transformers/hub/config.dart';
import 'package:dart_transformers/tokenizers/tokenizer.dart';
import 'package:dart_transformers/tokenizers/tokenizer_error.dart';
import 'package:dart_transformers/tokenizers/utils.dart';
import 'package:unicode/unicode.dart' as unicode;
import 'package:unicode/decomposer.dart';
import 'package:unicode/decomposers/canonical.dart';

class BertTokenizer extends PreTrainedTokenizerModel {
  final BasicTokenizer basicTokenizer;
  final WordpieceTokenizer wordpieceTokenizer;
  final int maxLen = 512;
  final bool tokenizeChineseChars;

  final Map<String, int> vocab;
  final Map<int, String> idsToTokens;

  @override
  String? bosToken;
  @override
  int? bosTokenId;
  @override
  String? eosToken;
  @override
  int? eosTokenId;
  @override
  final bool fuseUnknownTokens;

  BertTokenizer({
    required this.vocab,
    this.tokenizeChineseChars = true,
    this.bosToken,
    this.eosToken,
    this.fuseUnknownTokens = false,
    bool doLowerCase = true,
  })  : idsToTokens = vocab.invert(),
        basicTokenizer = BasicTokenizer(doLowerCase: doLowerCase),
        wordpieceTokenizer = WordpieceTokenizer(vocab) {
    bosTokenId = bosToken != null ? vocab[bosToken!] : null;
    eosTokenId = eosToken != null ? vocab[eosToken!] : null;
  }

  BertTokenizer.fromConfig(Config tokenizerConfig, Config tokenizerData, Map<String, int> addedTokens)
      : vocab = tokenizerData["model"]!["vocab"]!.dictionary.cast<String, int>(),
        idsToTokens = tokenizerData["model"]!["vocab"]!.dictionary.cast<String, int>().invert(),
        basicTokenizer = BasicTokenizer(doLowerCase: tokenizerConfig["doLowerCase"]?.boolValue ?? true),
        wordpieceTokenizer = WordpieceTokenizer(tokenizerData["model"]!["vocab"]!.dictionary.cast<String, int>()),
        tokenizeChineseChars = tokenizerConfig["handleChineseChars"]?.boolValue ?? true,
        bosToken = tokenizerConfig["bosToken"]?.stringValue,
        eosToken = tokenizerConfig["eosToken"]?.stringValue,
        fuseUnknownTokens = tokenizerConfig["fuseUnk"]?.boolValue ?? false {
    bosTokenId = bosToken != null ? vocab[bosToken!] : null;
    eosTokenId = eosToken != null ? vocab[eosToken!] : null;
  }

  @override
  List<String> tokenize(String text) {
    text = tokenizeChineseCharsIfNeed(text);
    List<String> tokens = [];
    for (var token in basicTokenizer.tokenize(text)) {
      for (var subToken in wordpieceTokenizer.tokenize(token)) {
        tokens.add(subToken);
      }
    }
    return tokens;
  }

  @override
  List<int> convertTokensToIds(List<String> tokens) {
    if (tokens.length > maxLen) {
      throw TokenizerError.tooLong(
        "Token indices sequence length is longer than the specified maximum sequence length for this BERT model (${tokens.length} > $maxLen). Running this sequence through BERT will result in indexing errors.",
      );
    }
    return tokens.map((token) => vocab[token]!).toList();
  }

  List<int> tokenizeToIds(String text) {
    return convertTokensToIds(tokenize(text));
  }

  int tokenToId(String token) {
    return vocab[token]!;
  }

  List<String> unTokenize(List<int> tokens) {
    return tokens.map((id) => idsToTokens[id]!).toList();
  }

  String convertWordpieceToBasicTokenList(List<String> wordpieceTokenList) {
    List<String> tokenList = [];
    String individualToken = "";

    for (var token in wordpieceTokenList) {
      if (token.startsWith("##")) {
        individualToken += token.substring(2);
      } else {
        if (individualToken.isNotEmpty) {
          tokenList.add(individualToken);
        }
        individualToken = token;
      }
    }

    tokenList.add(individualToken);

    return tokenList.join(" ");
  }

  String tokenizeChineseCharsIfNeed(String text) {
    if (!tokenizeChineseChars) {
      return text;
    }

    return text.split('').map((c) {
      if (c.isChineseChar) {
        return " $c ";
      } else {
        return c;
      }
    }).join();
  }

  @override
  String? convertIdToToken(int id) => idsToTokens[id];

  @override
  int? convertTokenToId(String token) => vocab[token] ?? unknownTokenId;

  @override
  String? get unknownToken => wordpieceTokenizer.unkToken;

  @override
  int? get unknownTokenId => vocab[unknownToken!];
}

extension BertTokenizerExtension on BertTokenizer {
  String? get unknownToken => wordpieceTokenizer.unkToken;
  int? get unknownTokenId => vocab[unknownToken!];

  List<int> encode(String text) => tokenizeToIds(text);

  String decode(List<int> tokens) {
    final tokenStrings = unTokenize(tokens);
    return convertWordpieceToBasicTokenList(tokenStrings);
  }

  int? convertTokenToId(String token) {
    return vocab[token] ?? unknownTokenId;
  }

  String? convertIdToToken(int id) {
    return idsToTokens[id];
  }
}

class BasicTokenizer {
  final bool doLowerCase;

  BasicTokenizer({this.doLowerCase = true});

  final List<String> neverSplit = ["[UNK]", "[SEP]", "[PAD]", "[CLS]", "[MASK]"];
  String maybeStripAccents(String text) {
    if (!doLowerCase) return text;

    final result = decompose(text, [CanonicalDecomposer()]);
    return String.fromCharCodes(
      result.runes.where((c) => !unicode.isNonspacingMark(c)),
    );
  }

  String maybeLowercase(String text) {
    if (!doLowerCase) return text;
    return text.toLowerCase();
  }

  List<String> tokenize(String text) {
    final splitTokens = maybeStripAccents(text).split(RegExp(r'\s+'));
    final tokens = splitTokens.expand((token) {
      if (neverSplit.contains(token)) {
        return [token];
      }
      List<String> toks = [];
      String currentTok = '';
      for (var c in maybeLowercase(token).split('')) {
        if (!c.isExtendedPunctuation) {
          currentTok += c;
        } else if (currentTok.isNotEmpty) {
          toks.add(currentTok);
          toks.add(c);
          currentTok = '';
        } else {
          toks.add(c);
        }
      }
      if (currentTok.isNotEmpty) {
        toks.add(currentTok);
      }
      return toks;
    }).toList();
    return tokens;
  }
}

class WordpieceTokenizer {
  final String unkToken = "[UNK]";
  final int maxInputCharsPerWord = 100;
  final Map<String, int> vocab;

  WordpieceTokenizer(this.vocab);

  /// `word`: A single token.
  /// Warning: this differs from the `pytorch-transformers` implementation.
  /// This should have already been passed through `BasicTokenizer`.
  List<String> tokenize(String word) {
    if (word.length > maxInputCharsPerWord) {
      return [unkToken];
    }
    List<String> outputTokens = [];
    bool isBad = false;
    int start = 0;
    List<String> subTokens = [];

    while (start < word.length) {
      int end = word.length;
      String? curSubstr;

      while (start < end) {
        String substr = word.substring(start, end);
        if (start > 0) {
          substr = "##$substr";
        }
        if (vocab.containsKey(substr)) {
          curSubstr = substr;
          break;
        }
        end -= 1;
      }
      if (curSubstr == null) {
        isBad = true;
        break;
      }
      subTokens.add(curSubstr);
      start = end;
    }
    if (isBad) {
      outputTokens.add(unkToken);
    } else {
      outputTokens.addAll(subTokens);
    }
    return outputTokens;
  }
}
