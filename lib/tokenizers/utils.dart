import 'package:dart_transformers/hub/config.dart';
import 'package:http/http.dart' as http;

extension ChineseCharExtension on String {
  /// https://en.wikipedia.org/wiki/CJK_Unified_Ideographs_(Unicode_block)
  bool get isChineseChar {
    assert(length == 1);
    final c = codeUnitAt(0);
    return (c >= 0x4E00 && c <= 0x9FFF) ||
        (c >= 0x3400 && c <= 0x4DBF) ||
        (c >= 0x20000 && c <= 0x2A6DF) ||
        (c >= 0x2A700 && c <= 0x2B73F) ||
        (c >= 0x2B740 && c <= 0x2B81F) ||
        (c >= 0x2B820 && c <= 0x2CEAF) ||
        (c >= 0xF900 && c <= 0xFAFF) ||
        (c >= 0x2F800 && c <= 0x2FA1F);
  }

  String handleChineseChars() {
    return toIterable().map((char) => char.isChineseChar ? " $char " : char).join();
  }
}

class Constants {
  // ignore: non_constant_identifier_names
  static final PUNCTUATION_REGEX = "\\p{P}\u0021-\u002F\u003A-\u0040\u005B-\u0060\u007B-\u007E";
}

class Range {
  final int start;
  final int end;

  Range(this.start, this.end);
}

extension ListTuple<T> on List<T> {
  (T, T) get toTuple {
    return (this[0], this[1]);
  }
}

Map<int, String> bytesChar() {
  final List<int> bs = [];

  // Add range: '!' (33) to '~' (126)
  for (int i = 33; i <= 126; i++) {
    bs.add(i);
  }

  // Add range: 0xA1 (161) to 0xAC (172)
  for (int i = 0xA1; i <= 0xAC; i++) {
    bs.add(i);
  }

  // Add range: 0xAE (174) to 0xFF (255)
  for (int i = 0xAE; i <= 0xFF; i++) {
    bs.add(i);
  }

  // Create corresponding list of codepoints
  final List<int> cs = bs.map((i) => i).toList();
  int n = 0;

  // For bytes not present in bs, add them with a codepoint of 256+n.
  for (int b = 0; b <= 255; b++) {
    if (!bs.contains(b)) {
      bs.add(b);
      cs.add(256 + n);
      n++;
    }
  }

  // Zip bs and cs into a Map<int, String>.
  final Map<int, String> mapping = {};
  for (int i = 0; i < bs.length; i++) {
    mapping[bs[i]] = String.fromCharCode(cs[i]);
  }
  return mapping;
}

extension StringExtension on String {
  String get capitalize {
    return "${this[0].toUpperCase()}${substring(1).toLowerCase()}";
  }

  bool get isExtendedPunctuation {
    assert(length == 1, 'Input must be a single character.');
    int cp = codeUnitAt(0);
    // We treat all non-letter/number ASCII as punctuation.
    // Characters such as "^", "$", and "`" are not in the Unicode
    // Punctuation class but we treat them as punctuation anyways, for
    // consistency.
    if ((cp >= 33 && cp <= 47) || (cp >= 58 && cp <= 64) || (cp >= 91 && cp <= 96) || (cp >= 123 && cp <= 126)) {
      return true;
    }
    String cat = String.fromCharCode(cp).runes.map((rune) => String.fromCharCode(rune)).first;
    if (RegExp(r'[\p{P}]', unicode: true).hasMatch(cat)) {
      return true;
    }
    return false;
  }
}

extension StringMatching on String {
  bool matchesGlob(String glob) {
    final escaped = glob.replaceAllMapped(
      RegExp(r'([.+^$(){}|\[\]\\])'),
      (match) => '\\${match[0]}',
    );
    final pattern = '^${escaped.replaceAll('*', '.*').replaceAll('?', '.')}\$';
    final regex = RegExp(pattern);
    return regex.hasMatch(this);
  }
}

extension ListStringMatching on List<String> {
  List<String> matching(String glob) {
    return where((str) => str.matchesGlob(glob)).toList();
  }
}

Future<http.Response> head(Uri url, {Map<String, String>? headers}) {
  http.Request req = http.Request("HEAD", url)
    ..followRedirects = false
    ..headers.addAll(headers ?? {});
  return _withClient((client) async => http.Response.fromStream(await client.send(req)));
}

Future<T> _withClient<T>(Future<T> Function(http.Client) fn) async {
  var client = http.Client();
  try {
    return await fn(client);
  } finally {
    client.close();
  }
}

extension InvertMapExtension<K, V> on Map<K, V> {
  Map<V, K> invert() {
    final inverted = <V, K>{};
    for (final entry in entries) {
      inverted[entry.value] = entry.key;
    }
    return inverted;
  }
}

extension MergeWithConfig<K, V> on Map<K, V> {
  Map<K, V> mergeWithConfig(Config? other) {
    return {
      ...other?.dictionary.cast<K, V>() ?? {},
      ...this,
    };
  }
}

extension AddedTokenExtension on Config {
  String? get addedTokenAsString {
    if (stringValue != null) {
      return stringValue;
    }
    // This is possibly a serialization of the AddedToken class
    // TODO: support lstrip, rstrip, normalized, etc.
    return this["content"]?.stringValue;
  }
}

extension Flatten<T extends Object> on Iterable<T> {
  Iterable<T> flatten() {
    Iterable<T> innerFlatten(Iterable<T> list) sync* {
      for (final value in list) {
        if (value is List<T>) {
          yield* innerFlatten(value);
        } else {
          yield value;
        }
      }
    }

    return innerFlatten(this);
  }
}

extension IterableString on String {
  Iterable<String> toIterable() sync* {
    for (var i = 0; i < length; i++) {
      yield substring(i, i + 1);
    }
  }
}
