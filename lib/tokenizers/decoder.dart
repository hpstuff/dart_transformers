import 'dart:convert';
import 'dart:core';

import 'package:dart_transformers/hub/config.dart';
import 'package:dart_transformers/tokenizers/normalizer.dart';
import 'package:dart_transformers/tokenizers/utils.dart';

abstract class Decoder {
  List<String> decode(List<String> tokens);
  List<String> call(List<String> tokens) => decode(tokens);

  Decoder(Config config);
}

enum DecoderType {
  sequence,
  wordPiece,
  byteLevel,
  replace,
  byteFallback,
  fuse,
  strip,
  metaspace,
  unknown;

  const DecoderType();

  static DecoderType fromString(String value) {
    return DecoderType.values.firstWhere(
      (e) => e.name.toLowerCase() == value.toLowerCase(),
      orElse: () => DecoderType.unknown,
    );
  }
}

class DecoderFactory {
  static Decoder? fromConfig(Config? config, [Set<String>? addedTokens]) {
    if (config == null) return null;
    final typeName = config["type"]?.stringValue;

    if (typeName == null) return null;
    switch (DecoderType.fromString(typeName)) {
      case DecoderType.sequence:
        return DecoderSequence(config);
      case DecoderType.byteLevel:
        return ByteLevelDecoder(config, addedTokens);
      case DecoderType.replace:
        return ReplaceDecoder(config);
      case DecoderType.byteFallback:
        return ByteFallbackDecoder(config);
      case DecoderType.fuse:
        return FuseDecoder(config);
      case DecoderType.strip:
        return StripDecoder(config);
      case DecoderType.metaspace:
        return MetaspaceDecoder(config);
      case DecoderType.wordPiece:
        return WordPieceDecoder(config);
      default:
        throw Exception('Unsupported Decoder type: $typeName');
    }
  }
}

class WordPieceDecoder extends Decoder {
  final String prefix;
  final bool cleanup;
  final RegExp re = RegExp(r"\s(\.|\?|\!|\,|'\s|n't|'m|'s|'ve|'re)");

  WordPieceDecoder(super.config)
      : prefix = config["prefix"]?.stringValue ?? '',
        cleanup = config["cleanup"]?.boolValue ?? false;

  @override
  List<String> decode(List<String> tokens) {
    final firstToken = cleanup ? cleanUpTokenization(tokens.first) : tokens.first;
    return [firstToken] +
        tokens.skip(1).map((token) {
          token = token.startsWith(prefix) ? token.replaceFirst(prefix, '') : ' $token';
          return cleanup ? cleanUpTokenization(token) : token;
        }).toList();
  }

  String cleanUpTokenization(String token) {
    return token.replaceAllMapped(re, (match) => match.group(1)!).replaceAll(' do not', " don't");
  }
}

class DecoderSequence extends Decoder {
  final List<Decoder> decoders;

  DecoderSequence(super.config)
      : decoders = config["decoders"]
                ?.arrayValue
                ?.map(
                  (c) => DecoderFactory.fromConfig(Config(c))!,
                )
                .toList() ??
            [];

  @override
  List<String> decode(List<String> tokens) {
    return decoders.fold(tokens, (current, decoder) => decoder.call(current));
  }
}

class ByteLevelDecoder extends Decoder {
  final Set<String> addedTokens;
  final Map<String, int> byteDecoder = bytesChar().invert();

  ByteLevelDecoder(super.config, [Set<String>? addedTokens]) : addedTokens = addedTokens ?? {};

  @override
  List<String> decode(List<String> tokens) {
    final subTexts = <String>[];
    var currentSubText = <String>[];

    String convertTokensToString(List<String> tokens) {
      final text = tokens.join('');
      final List<int> utfCodepoints = text.toIterable().map((c) => byteDecoder[c]).where((c) => c != null).toList().cast();
      return utf8.decode(utfCodepoints);
    }

    for (final token in tokens) {
      if (addedTokens.contains(token)) {
        if (currentSubText.isNotEmpty) {
          subTexts.add(convertTokensToString(currentSubText));
          currentSubText = [];
        }
        subTexts.add(token);
      } else {
        currentSubText.add(token);
      }
    }

    if (currentSubText.isNotEmpty) {
      subTexts.add(convertTokensToString(currentSubText));
    }

    return subTexts;
  }
}

class ReplaceDecoder extends Decoder {
  final StringReplacePattern? pattern;

  ReplaceDecoder(super.config) : pattern = StringReplacePattern.from(config);

  @override
  List<String> decode(List<String> tokens) {
    if (pattern == null) return tokens;
    return tokens.map((token) => pattern!.replace(token)).toList();
  }
}

class ByteFallbackDecoder extends Decoder {
  ByteFallbackDecoder(super.config);

  @override
  List<String> decode(List<String> tokens) {
    final newTokens = <String>[];
    var byteTokens = <int>[];

    int? parseByte(String token) {
      if (token.length == 6 && token.startsWith('<0x') && token.endsWith('>')) {
        final byteStr = token.substring(3, 5);
        return int.tryParse(byteStr, radix: 16);
      }
      return null;
    }

    for (final token in tokens) {
      final byte = parseByte(token);
      if (byte != null) {
        byteTokens.add(byte);
      } else {
        if (byteTokens.isNotEmpty) {
          final codeUnits = byteTokens.map((b) => b).toList();
          newTokens.add(utf8.decode(codeUnits));
          byteTokens = [];
        }
        newTokens.add(token);
      }
    }
    return newTokens;
  }
}

class FuseDecoder extends Decoder {
  FuseDecoder(super.config);

  @override
  List<String> decode(List<String> tokens) {
    return [tokens.join('')];
  }
}

class StripDecoder extends Decoder {
  final String content;
  final int start;
  final int stop;

  StripDecoder(super.config)
      : content = config["content"]?.stringValue ?? '',
        start = config["start"]?.intValue ?? 0,
        stop = config["stop"]?.intValue ?? 0;

  @override
  List<String> decode(List<String> tokens) {
    return tokens.map((token) {
      return token.trimFromStart(start).trimFromEnd(stop);
    }).toList();
  }
}

class MetaspaceDecoder extends Decoder {
  final bool addPrefixSpace;
  final String replacement;

  MetaspaceDecoder(super.config)
      : addPrefixSpace = config["addPrefixSpace"]?.boolValue ?? false,
        replacement = config["replacement"]?.stringValue ?? '_';

  @override
  List<String> decode(List<String> tokens) {
    final replaced = tokens.map((token) {
      return token.replaceAll(replacement, ' ');
    }).toList();
    if (addPrefixSpace && replaced.first.startsWith(' ')) {
      replaced[0] = replaced[0].substring(1);
    }
    return replaced;
  }
}

extension StringExtensions on String {
  String trimFromStart(int upto, [String character = ' ']) {
    var result = this;
    var trimmed = 0;
    while (trimmed < upto && result.startsWith(character)) {
      result = result.substring(1);
      trimmed++;
    }
    return result;
  }

  String trimFromEnd(int upto, [String character = ' ']) {
    var result = this;
    var trimmed = 0;
    while (trimmed < upto && result.endsWith(character)) {
      result = result.substring(0, result.length - 1);
      trimmed++;
    }
    return result;
  }
}
