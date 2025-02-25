import 'dart:convert';

class Config {
  final Map<String, dynamic> _dictionary;

  Config(this._dictionary);

  String camelCase(String string) {
    return string
        .split('_')
        .asMap()
        .map((index, element) => MapEntry(index, index == 0 ? element.toLowerCase() : element[0].toUpperCase() + element.substring(1).toLowerCase()))
        .values
        .join();
  }

  String uncamelCase(String string) {
    final buffer = StringBuffer();
    bool previousCharacterIsLowercase = false;

    for (var i = 0; i < string.length; i++) {
      var char = string[i];
      if (char.toUpperCase() == char && previousCharacterIsLowercase) {
        buffer.write('_');
      }
      buffer.write(char.toLowerCase());
      previousCharacterIsLowercase = char.toLowerCase() == char;
    }

    return buffer.toString();
  }

  Config? operator [](String member) {
    var key = _dictionary.containsKey(member) ? member : uncamelCase(member);
    var value = _dictionary[key];
    if (value is Map<String, dynamic>) {
      return Config(value);
    } else if (value != null) {
      return Config({'value': value});
    }
    return null;
  }

  dynamic get value => _dictionary['value'];

  Map get dictionary => _dictionary;

  int? get intValue => value != null ? BigInt.from(value).toInt() : null;
  bool? get boolValue => value as bool?;
  String? get stringValue => value as String?;

  List<dynamic>? get arrayValue {
    var list = value as List<dynamic>?;
    return list?.map((item) => item).toList();
  }

  (int, String)? get tokenValue {
    var tuple = value as List<dynamic>?;
    if (tuple != null && tuple.length == 2 && tuple[0] is int && tuple[1] is String) {
      return (tuple[0] as int, tuple[1] as String);
    }
    return null;
  }

  Config merge(Config other) {
    return Config({
      ..._dictionary,
      ...other._dictionary,
    });
  }

  @override
  String toString() {
    return "Config<${jsonEncode(_dictionary)}>";
  }
}
