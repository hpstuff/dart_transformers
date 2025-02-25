import 'dart:convert';

import 'package:dart_transformers/hub/config.dart';
import 'package:dart_transformers/tokenizers/utils.dart';

enum PreTokenizerOption { firstSection }

typedef PreTokenizerOptions = Set<PreTokenizerOption>;

abstract class PreTokenizer {
  PreTokenizer(Config config);

  List<String> preTokenize(String text, [PreTokenizerOptions options]);

  List<String> preTokenizeMultiple(
    List<String> texts, [
    PreTokenizerOptions options = const {PreTokenizerOption.firstSection},
  ]) {
    return texts.expand((text) => preTokenize(text, options)).toList();
  }

  List<String> call(
    List<String> texts, [
    PreTokenizerOptions options = const {PreTokenizerOption.firstSection},
  ]) {
    return preTokenizeMultiple(texts, options);
  }

  List<String> callSingle(
    String text, [
    PreTokenizerOptions options = const {PreTokenizerOption.firstSection},
  ]) {
    return preTokenize(text, options);
  }
}

enum PreTokenizerType {
  sequence,
  byteLevel,
  punctuation,
  digits,
  split,
  whitespace,
  whitespaceSplit,
  metaspace,
  bertPreTokenizer,
  unknown;

  const PreTokenizerType();

  static PreTokenizerType fromString(String value) {
    return PreTokenizerType.values.firstWhere(
      (e) => e.name.toLowerCase() == value.toLowerCase(),
      orElse: () => PreTokenizerType.unknown,
    );
  }
}

class PreTokenizerFactory {
  static PreTokenizer? fromConfig(Config? config) {
    if (config == null) return null;
    final typeName = config["type"]?.stringValue;
    if (typeName == null) return null;

    switch (PreTokenizerType.fromString(typeName)) {
      case PreTokenizerType.sequence:
        return PreTokenizerSequence(config);
      case PreTokenizerType.byteLevel:
        return ByteLevelPreTokenizer(config);
      case PreTokenizerType.punctuation:
        return PunctuationPreTokenizer(config);
      case PreTokenizerType.digits:
        return DigitsPreTokenizer(config);
      case PreTokenizerType.split:
        return SplitPreTokenizer(config);
      case PreTokenizerType.whitespace:
      case PreTokenizerType.whitespaceSplit:
        return WhitespacePreTokenizer(config);
      case PreTokenizerType.metaspace:
        return MetaspacePreTokenizer(config);
      case PreTokenizerType.bertPreTokenizer:
        return BertPreTokenizer(config);
      default:
        throw Exception('Unsupported PreTokenizer type: $typeName');
    }
  }
}

class BertPreTokenizer extends PreTokenizer {
  final String re;

  BertPreTokenizer(super.config) : re = "[^\\s${Constants.PUNCTUATION_REGEX}]+|[${Constants.PUNCTUATION_REGEX}]";

  @override
  List<String> preTokenize(
    String text, [
    PreTokenizerOptions options = const {PreTokenizerOption.firstSection},
  ]) {
    final regex = RegExp(re, unicode: true);
    return regex.allMatches(text).map((match) => match.group(0)!).toList();
  }
}

class PreTokenizerSequence extends PreTokenizer {
  final List<PreTokenizer> preTokenizers;

  PreTokenizerSequence(super.config)
      : preTokenizers = (config["pretokenizers"]?.arrayValue ?? []).map((cfg) => PreTokenizerFactory.fromConfig(Config(cfg))!).toList();

  @override
  List<String> preTokenize(
    String text, [
    PreTokenizerOptions options = const {PreTokenizerOption.firstSection},
  ]) {
    return preTokenizers.fold([text], (current, preTokenizer) {
      return preTokenizer.call(current, options);
    });
  }
}

class WhitespacePreTokenizer extends PreTokenizer {
  final String re;

  WhitespacePreTokenizer(super.config) : re = r'\S+';

  @override
  List<String> preTokenize(String text, [PreTokenizerOptions? options]) {
    final regex = RegExp(re, unicode: true);
    return regex.allMatches(text).map((match) => match.group(0)!).toList();
  }
}

class MetaspacePreTokenizer extends PreTokenizer {
  final bool addPrefixSpace;
  final String replacement;
  final String stringReplacement;
  final PrependScheme prependScheme;

  MetaspacePreTokenizer(super.config)
      : addPrefixSpace = config["addPrefixSpace"]?.boolValue ?? false,
        replacement = config["replacement"]?.stringValue ?? " ",
        stringReplacement = config["strRep"]?.stringValue ?? config["replacement"]?.stringValue ?? " ",
        prependScheme = PrependScheme.from(config["prependScheme"]?.stringValue);

  @override
  List<String> preTokenize(
    String text, [
    PreTokenizerOptions options = const {PreTokenizerOption.firstSection},
  ]) {
    final normalized = text.replaceAll(" ", stringReplacement);

    var prepend = "";
    if (addPrefixSpace && !normalized.startsWith(replacement)) {
      if (prependScheme == PrependScheme.always) {
        prepend = stringReplacement;
      }
      if (prependScheme == PrependScheme.first && options.contains(PreTokenizerOption.firstSection)) {
        prepend = stringReplacement;
      }
    }

    return (prepend + normalized).splitWithDelimiterBehavior(
      replacement,
      behavior: DelimiterBehavior.mergeWithNext,
    );
  }
}

enum PrependScheme {
  first,
  never,
  always;

  const PrependScheme();

  static PrependScheme from(String? value) {
    for (final v in PrependScheme.values) {
      if (v.name == value) {
        return v;
      }
    }
    return PrependScheme.always;
  }
}

class ByteLevelPreTokenizer extends PreTokenizer {
  final Map<int, String> bytes = bytesChar();
  final bool addPrefixSpace;
  final bool trimOffsets;
  final bool useRegex;
  final String re = r"'s|'t|'re|'ve|'m|'ll|'d| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+";

  ByteLevelPreTokenizer(super.config)
      : addPrefixSpace = config["addPrefixSpace"]?.boolValue ?? false,
        trimOffsets = config["trimOffsets"]?.boolValue ?? true,
        useRegex = config["useRegex"]?.boolValue ?? true;

  @override
  List<String> preTokenize(
    String text, [
    PreTokenizerOptions options = const {PreTokenizerOption.firstSection},
  ]) {
    final regex = RegExp(re, unicode: true);
    final tokens = useRegex ? regex.allMatchesWithValue(text).map((e) => e.$1).toList() : [text];

    return tokens.map((token) {
      return addPrefixSpace && !token.startsWith(" ") ? " $token" : token;
    }).map((token) {
      return utf8.encode(token).map((unit) => bytes[unit]).join();
    }).toList();
  }
}

class PunctuationPreTokenizer extends PreTokenizer {
  final String re;

  PunctuationPreTokenizer(super.config) : re = "[^${Constants.PUNCTUATION_REGEX}]+|[${Constants.PUNCTUATION_REGEX}]+";

  @override
  List<String> preTokenize(
    String text, [
    PreTokenizerOptions options = const {PreTokenizerOption.firstSection},
  ]) {
    final regex = RegExp(re, unicode: true);
    return regex.allMatches(text).map((match) => match.group(0)!).toList();
  }
}

class DigitsPreTokenizer extends PreTokenizer {
  final String re;

  DigitsPreTokenizer(super.config) : re = config["individualDigits"]?.boolValue ?? false ? r'[^\d]+|\d' : r'[^\d]+|\d+';

  @override
  List<String> preTokenize(
    String text, [
    PreTokenizerOptions options = const {PreTokenizerOption.firstSection},
  ]) {
    final regex = RegExp(re, unicode: true);
    return regex.allMatches(text).map((match) => match.group(0)!).toList();
  }
}

class SplitPreTokenizer extends PreTokenizer {
  final StringSplitPatternConfig? pattern;
  final bool invert;

  SplitPreTokenizer(super.config)
      : pattern = StringSplitPatternConfig.fromConfig(config),
        invert = config["invert"]?.boolValue ?? false;

  @override
  List<String> preTokenize(
    String text, [
    PreTokenizerOptions options = const {PreTokenizerOption.firstSection},
  ]) {
    if (pattern == null) return [text];
    return pattern!.split(text, invert: invert);
  }
}

enum StringSplitPattern {
  regexp,
  string,
}

class StringSplitPatternConfig {
  final String pattern;
  final StringSplitPattern type;

  StringSplitPatternConfig.regexp(this.pattern) : type = StringSplitPattern.regexp;
  StringSplitPatternConfig.string(this.pattern) : type = StringSplitPattern.string;

  static StringSplitPatternConfig? fromConfig(Config config) {
    final sPattern = config["pattern"]?["String"]?.stringValue;
    if (sPattern != null) {
      return StringSplitPatternConfig.string(sPattern);
    }

    final rPattern = config["pattern"]?["Regex"]?.stringValue;
    if (rPattern != null) {
      return StringSplitPatternConfig.regexp(rPattern);
    }

    return null;
  }
}

extension StringSplitPatternExtension on StringSplitPatternConfig {
  List<String> split(String text, {bool invert = true}) {
    switch (type) {
      case StringSplitPattern.regexp:
        final regex = RegExp(pattern.replaceAll("?i:", "i:"), unicode: true);
        return text.splitBy(regex, includeSeparators: true);
      case StringSplitPattern.string:
        return text.splitBy(pattern, includeSeparators: !invert);
    }
  }
}

extension StringSplitExtension on String {
  List<String> splitBy(
    Pattern pattern, {
    bool includeSeparators = false,
    bool omittingEmptySubsequences = true,
  }) {
    final matches = pattern.allMatches(this);
    final result = <String>[];
    var start = 0;

    for (final match in matches) {
      if (omittingEmptySubsequences && start < match.start) {
        result.add(substring(start, match.start));
      }
      if (includeSeparators) {
        result.add(match.group(0)!);
      }
      start = match.end;
    }

    if (omittingEmptySubsequences && start < length) {
      result.add(substring(start));
    }

    return result;
  }
}

typedef MatchesWithValue = (String, bool);

extension MatchedGroup on Match {
  String? matched() {
    if (groupCount == 0) {
      return group(0);
    }
    return groups(List.from(
      Iterable.generate(groupCount).map((i) => i + 1),
    )).firstWhere(
      (g) => g != null,
      orElse: () => group(0),
    );
  }
}

extension SplitMatchesExtension on Pattern {
  List<MatchesWithValue> allMatchesWithValue(String text) {
    final matches = allMatches(text);
    final result = <MatchesWithValue>[];
    var start = 0;

    for (final match in matches) {
      if (start < match.start) {
        result.add((text.substring(start, match.start), false));
      }

      final group = match.matched();

      result.add((group!, true));
      start = match.end;
    }
    if (start <= text.length - 1 || start == 0) {
      result.add((text.substring(start), false));
    }
    return result;
  }
}

extension SplitDelimiterBehaviorExtension on String {
  List<String> splitWithDelimiterBehavior(
    Pattern pattern, {
    DelimiterBehavior behavior = DelimiterBehavior.remove,
  }) {
    final matches = pattern.allMatchesWithValue(this);

    switch (behavior) {
      case DelimiterBehavior.isolate:
        return matches.map((e) => e.$1).toList();
      case DelimiterBehavior.mergeWithPrevious:
        var previousMatch = false;
        return matches
            .fold<List<MatchesWithValue>>([], (acc, match) {
              if (match.$2 && !previousMatch) {
                if (acc.isNotEmpty) {
                  acc.last = (acc.last.$1 + match.$1, false);
                } else {
                  acc.add((match.$1, false));
                }
              } else {
                acc.add((match.$1, false));
              }
              previousMatch = match.$2;
              return acc;
            })
            .map((e) => e.$1)
            .toList();
      case DelimiterBehavior.mergeWithNext:
        var previousMatch = false;
        return matches.reversed
            .fold<List<MatchesWithValue>>([], (acc, match) {
              if (match.$2 && !previousMatch) {
                if (acc.isNotEmpty) {
                  acc.last = (match.$1 + acc.last.$1, false);
                } else {
                  acc.add((match.$1, false));
                }
              } else {
                acc.add((match.$1, false));
              }
              previousMatch = match.$2;
              return acc;
            })
            .reversed
            .map((e) => e.$1)
            .toList();
      default:
        return matches.where((match) => !match.$2).map((match) => match.$1).toList();
    }
  }
}

enum DelimiterBehavior {
  remove,
  isolate,
  mergeWithNext,
  mergeWithPrevious,
}
