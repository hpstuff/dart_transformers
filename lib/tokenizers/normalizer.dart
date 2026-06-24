import 'package:dart_transformers/hub/config.dart';
import "package:unorm_dart/unorm_dart.dart" as unorm;
import 'package:dart_transformers/tokenizers/utils.dart';

abstract class Normalizer {
  String normalize(String text);
  String call(String text) => normalize(text);

  Normalizer(Config config);
}

enum NormalizerType {
  sequence,
  prepend,
  replace,
  lowercase,
  nfd,
  nfc,
  nfkd,
  nfkc,
  bert,
  bertNormalizer,
  precompiled,
  stripAccents,
  strip,
  unknown;

  const NormalizerType();

  static NormalizerType fromString(String value) {
    return NormalizerType.values.firstWhere(
      (e) => e.name.toLowerCase() == value.toLowerCase(),
      orElse: () => NormalizerType.unknown,
    );
  }
}

class NormalizerFactory {
  static Normalizer? fromConfig(Config? config) {
    if (config == null) return null;
    final typeName = config["type"]?.stringValue;
    if (typeName == null) return null;

    switch (NormalizerType.fromString(typeName)) {
      case NormalizerType.sequence:
        return NormalizerSequence(config);
      case NormalizerType.prepend:
        return PrependNormalizer(config);
      case NormalizerType.replace:
        return ReplaceNormalizer(config);
      case NormalizerType.lowercase:
        return LowercaseNormalizer(config);
      case NormalizerType.nfd:
        return NFDNormalizer(config);
      case NormalizerType.nfc:
        return NFCNormalizer(config);
      case NormalizerType.nfkd:
        return NFKDNormalizer(config);
      case NormalizerType.nfkc:
        return NFKCNormalizer(config);
      case NormalizerType.bert:
      case NormalizerType.bertNormalizer:
        return BertNormalizer(config);
      case NormalizerType.precompiled:
        return PrecompiledNormalizer(config);
      case NormalizerType.stripAccents:
        return StripAccentsNormalizer(config);
      case NormalizerType.strip:
        return StripNormalizer(config);
      default:
        throw Exception('Unsupported Normalizer type: $typeName');
    }
  }
}

class NormalizerSequence extends Normalizer {
  final List<Normalizer> normalizers;

  NormalizerSequence(super.config)
      : normalizers = config["normalizers"]
                ?.arrayValue
                ?.map(
                  (c) => NormalizerFactory.fromConfig(Config(c))!,
                )
                .toList() ??
            [];

  @override
  String normalize(String text) {
    return normalizers.fold(text, (current, normalizer) => normalizer.call(current));
  }
}

class PrependNormalizer extends Normalizer {
  final String prepend;

  PrependNormalizer(super.config) : prepend = config["prepend"]?.stringValue ?? '';

  @override
  String normalize(String text) {
    return prepend + text;
  }
}

class ReplaceNormalizer extends Normalizer {
  final StringReplacePattern? pattern;

  ReplaceNormalizer(super.config) : pattern = StringReplacePattern.from(config);

  @override
  String normalize(String text) {
    return pattern?.replace(text) ?? text;
  }
}

class LowercaseNormalizer extends Normalizer {
  LowercaseNormalizer(super.config);

  @override
  String normalize(String text) {
    return text.toLowerCase();
  }
}

class NFDNormalizer extends Normalizer {
  NFDNormalizer(super.config);

  @override
  String normalize(String text) {
    return unorm.nfd(text);
  }
}

class NFCNormalizer extends Normalizer {
  NFCNormalizer(super.config);

  @override
  String normalize(String text) {
    return unorm.nfc(text);
  }
}

class NFKDNormalizer extends Normalizer {
  NFKDNormalizer(super.config);

  @override
  String normalize(String text) {
    return unorm.nfkd(text);
  }
}

class NFKCNormalizer extends Normalizer {
  NFKCNormalizer(super.config);

  @override
  String normalize(String text) {
    return unorm.nfkc(text);
  }
}

class BertNormalizer extends Normalizer {
  final bool shouldCleanText;
  final bool shouldHandleChineseChars;
  final bool shouldStripAccents;
  final bool shouldLowercase;

  BertNormalizer(super.config)
      : shouldCleanText = config["cleanText"]?.boolValue ?? true,
        shouldHandleChineseChars = config["handleChineseChars"]?.boolValue ?? true,
        shouldLowercase = config["lowercase"]?.boolValue ?? true,
        shouldStripAccents = config["stripAccents"]?.boolValue ?? (config["lowercase"]?.boolValue ?? true);

  @override
  String normalize(String text) {
    var output = text;
    if (shouldCleanText) {
      output = cleanText(output);
    }
    if (shouldHandleChineseChars) {
      output = output.handleChineseChars();
    }
    if (shouldStripAccents) {
      output = stripAccents(output);
    }
    if (shouldLowercase) {
      output = output.toLowerCase();
    }
    return output;
  }

  String cleanText(String text) {
    return text.runes.map((int codeUnit) {
      if (codeUnit == 0x0 || codeUnit == 0xFFFD || isControl(codeUnit)) {
        return String.fromCharCode(codeUnit);
      }

      // Replace whitespace: \t, \n, \r
      if (codeUnit == 0x009 || codeUnit == 0x00A || codeUnit == 0x000D) {
        return ' ';
      } else {
        return String.fromCharCode(codeUnit);
      }
    }).join();
  }

  bool isControl(int codeUnit) {
    if (codeUnit == 0x009 || codeUnit == 0x00A || codeUnit == 0x000D) {
      // Except \t, \n, \r that will be spaces.
      return false;
    } else {
      // https://unicode.org/reports/tr44/#GC_Values_Table
      // Other Cc | Cf | Cs | Co | Cn
      return isOther(codeUnit);
    }
  }

  bool isOther(int codeUnit) {
    return (codeUnit >= 0x00 && codeUnit <= 0x1F) || (codeUnit >= 0x7F && codeUnit <= 0x9F);
  }

  String stripAccents(String text) {
    return unorm.nfd(text).replaceAll(RegExp(r'[\u0300-\u036F]'), '');
  }
}

class PrecompiledNormalizer extends Normalizer {
  PrecompiledNormalizer(super.config);

  @override
  String normalize(String text) {
    // var output = StringBuffer();
    // var hasFullwidthTilde = false;

    // for (var codeUnit in utf8.encode(text)) {
    //   if ((codeUnit >= 0x0001 && codeUnit <= 0x0008) ||
    //       codeUnit == 0x000B ||
    //       (codeUnit >= 0x000E && codeUnit <= 0x001F) ||
    //       codeUnit == 0x007F ||
    //       codeUnit == 0x008F ||
    //       codeUnit == 0x009F) {
    //     // Non-printing control characters
    //     continue;
    //   } else if (codeUnit == 0x0009 ||
    //       codeUnit == 0x000A ||
    //       codeUnit == 0x000C ||
    //       codeUnit == 0x000D ||
    //       codeUnit == 0x1680 ||
    //       (codeUnit >= 0x200B && codeUnit <= 0x200F) ||
    //       codeUnit == 0x2028 ||
    //       codeUnit == 0x2029 ||
    //       codeUnit == 0x2581 ||
    //       codeUnit == 0xFEFF ||
    //       codeUnit == 0xFFFD) {
    //     // Separators
    //     output.write(' ');
    //   } else if (codeUnit == 0xFF5E) {
    //     hasFullwidthTilde = true;
    //     output.write(String.fromCharCode(codeUnit));
    //   } else {
    //     output.write(String.fromCharCode(codeUnit));
    //   }
    // }

    var result = text.replaceAll(RegExp("r[\u0001-\u0008\u000B\u000E-\u001F\u007F\u008F\u009F]", unicode: true), ''); // Remove control characters
    text = text.replaceAll(
        RegExp(r"[\u0009\u000A\u000C\u000D\u00A0\u1680\u2000-\u200F\u2028\u2029\u202F\u205F\u2581\u3000\uFEFF\uFFFD]", unicode: true), '\u0020');

    if (text.contains('\uFF5E')) {
      return result.split('\uFF5E').map((part) => unorm.nfkc(part)).join('\uFF5E');
    } else {
      return unorm.nfkc(result);
    }
  }
}

class StripAccentsNormalizer extends Normalizer {
  StripAccentsNormalizer(super.config);

  @override
  String normalize(String text) {
    return unorm.nfkc(text);
  }
}

class StripNormalizer extends Normalizer {
  final bool leftStrip;
  final bool rightStrip;

  StripNormalizer(super.config)
      : leftStrip = config["stripLeft"]?.boolValue ?? true,
        rightStrip = config["stripRight"]?.boolValue ?? true;

  @override
  String normalize(String text) {
    var result = text;
    if (leftStrip) {
      result = result.trimLeft();
    }
    if (rightStrip) {
      result = result.trimRight();
    }
    return result;
  }
}

class StringReplacePattern {
  final RegExp? regexp;
  final String? pattern;
  final String replacement;

  StringReplacePattern.regexp(this.regexp, this.replacement) : pattern = null;
  StringReplacePattern.string(this.pattern, this.replacement) : regexp = null;

  String replace(String text) {
    if (regexp != null) {
      return text.replaceAll(regexp!, replacement);
    } else if (pattern != null) {
      return text.replaceAll(pattern!, replacement);
    }
    return text;
  }

  static StringReplacePattern? from(Config config) {
    final replacement = config["content"]?.stringValue;
    if (replacement == null) return null;
    if (config["pattern"]?["String"]?.stringValue != null) {
      return StringReplacePattern.string(config["pattern"]!["String"]!.stringValue!, replacement);
    }
    if (config["pattern"]?["Regex"]?.stringValue != null) {
      return StringReplacePattern.regexp(RegExp(config["pattern"]!["Regex"]!.stringValue!), replacement);
    }
    return null;
  }

  @override
  String toString() {
    return 'StringReplacePattern(regexp: $regexp, pattern: $pattern, replacement: $replacement)';
  }
}
