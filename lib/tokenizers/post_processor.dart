import 'package:dart_transformers/hub/config.dart';

abstract class PostProcessor {
  List<String> postProcess(List<String> tokens, {List<String>? tokensPair, bool addSpecialTokens = true});
  List<String> call(List<String> tokens, {List<String>? tokensPair, bool addSpecialTokens = true}) {
    return postProcess(tokens, tokensPair: tokensPair, addSpecialTokens: addSpecialTokens);
  }

  PostProcessor();

  PostProcessor.fromConfig(Config config);
}

enum PostProcessorType {
  templateProcessing,
  byteLevel,
  robertaProcessing,
  bertProcessing,
  sequence;

  const PostProcessorType();

  static PostProcessorType fromString(String value) {
    return PostProcessorType.values.firstWhere(
      (e) => e.name.toLowerCase() == value.toLowerCase(),
      orElse: () => throw Exception('Unsupported PostProcessor type: $value'),
    );
  }
}

class PostProcessorFactory {
  static PostProcessor? fromConfig(Config? config) {
    if (config == null) return null;
    final typeName = config["type"]?.stringValue;
    if (typeName == null) return null;

    switch (PostProcessorType.fromString(typeName)) {
      case PostProcessorType.templateProcessing:
        return TemplateProcessing.fromConfig(config);
      case PostProcessorType.byteLevel:
        return ByteLevelPostProcessor.fromConfig(config);
      case PostProcessorType.robertaProcessing:
        return RobertaProcessing.fromConfig(config);
      case PostProcessorType.bertProcessing:
        return BertProcessing.fromConfig(config);
      case PostProcessorType.sequence:
        return SequenceProcessing.fromConfig(config);
    }
  }
}

class TemplateProcessing extends PostProcessor {
  final List<Config> single;
  final List<Config> pair;

  TemplateProcessing({required this.single, required this.pair});

  @override
  List<String> postProcess(List<String> tokens, {List<String>? tokensPair, bool addSpecialTokens = true}) {
    final config = tokensPair == null ? single : pair;

    final toReturn = <String>[];
    for (final item in config) {
      if (item["SpecialToken"] != null) {
        if (addSpecialTokens) {
          toReturn.add(item["SpecialToken"]!["id"]!.stringValue!);
        }
      } else if (item["Sequence"] != null) {
        if (item["Sequence"]!["id"]?.stringValue == 'A') {
          toReturn.addAll(tokens);
        } else if (item["Sequence"]!["id"]?.stringValue == 'B') {
          toReturn.addAll(tokensPair!);
        }
      }
    }
    return toReturn;
  }

  factory TemplateProcessing.fromConfig(Config config) {
    final List<Config> single = config["single"]?.arrayValue?.map((e) => Config(e)).toList() ?? (throw Exception('Missing `single` processor configuration'));
    final List<Config> pair = config["pair"]?.arrayValue?.map((e) => Config(e)).toList() ?? (throw Exception('Missing `pair` processor configuration'));
    return TemplateProcessing(single: single, pair: pair);
  }
}

class ByteLevelPostProcessor extends PostProcessor {
  ByteLevelPostProcessor();

  @override
  List<String> postProcess(List<String> tokens, {List<String>? tokensPair, bool addSpecialTokens = true}) {
    return tokens;
  }

  factory ByteLevelPostProcessor.fromConfig(Config config) {
    return ByteLevelPostProcessor();
  }
}

class RobertaProcessing extends PostProcessor {
  final (int, String) sep;
  final (int, String) cls;
  final bool trimOffset;
  final bool addPrefixSpace;

  RobertaProcessing({required this.sep, required this.cls, this.trimOffset = true, this.addPrefixSpace = true});

  @override
  List<String> postProcess(List<String> tokens, {List<String>? tokensPair, bool addSpecialTokens = true}) {
    var outTokens = tokens;
    var tokensPairCopy = tokensPair;
    if (trimOffset) {
      if (addPrefixSpace) {
        outTokens = outTokens.map(trimExtraSpaces).toList();
        tokensPairCopy = tokensPairCopy?.map(trimExtraSpaces).toList();
      } else {
        outTokens = outTokens.map((token) => token.trim()).toList();
        tokensPairCopy = tokensPairCopy?.map((token) => token.trim()).toList();
      }
    }

    outTokens = [cls.$2] + outTokens + [sep.$2];
    if (tokensPairCopy != null && tokensPairCopy.isNotEmpty) {
      outTokens += [sep.$2] + tokensPairCopy + [sep.$2];
    }

    return outTokens;
  }

  String trimExtraSpaces(String token) {
    final prefixOffset = findPrefixIndex(token);
    final suffixOffset = findSuffixIndex(token);
    return token.substring(prefixOffset, token.length - suffixOffset);
  }

  int findPrefixIndex(String text) {
    if (text.isEmpty || !text.startsWith(' ')) return 0;
    return text.indexOf(RegExp(r'[^ ]')) - 1;
  }

  int findSuffixIndex(String text) {
    if (text.isEmpty || !text.endsWith(' ')) return 0;
    return text.length - text.lastIndexOf(RegExp(r'[^ ]')) - 2;
  }

  factory RobertaProcessing.fromConfig(Config config) {
    final sep = config["sep"]?.tokenValue ?? (throw Exception('Missing `sep` processor configuration'));
    final cls = config["cls"]?.tokenValue ?? (throw Exception('Missing `cls` processor configuration'));
    final trimOffset = config["trimOffset"]?.boolValue ?? true;
    final addPrefixSpace = config["addPrefixSpace"]?.boolValue ?? true;
    return RobertaProcessing(sep: sep, cls: cls, trimOffset: trimOffset, addPrefixSpace: addPrefixSpace);
  }
}

class BertProcessing extends PostProcessor {
  final (int, String) sep;
  final (int, String) cls;

  BertProcessing({required this.sep, required this.cls});

  @override
  List<String> postProcess(List<String> tokens, {List<String>? tokensPair, bool addSpecialTokens = true}) {
    if (!addSpecialTokens) return tokens + (tokensPair ?? []);
    var outTokens = [cls.$2] + tokens + [sep.$2];
    if (tokensPair != null && tokensPair.isNotEmpty) {
      outTokens += tokensPair + [sep.$2];
    }
    return outTokens;
  }

  factory BertProcessing.fromConfig(Config config) {
    final sep = config["sep"]?.tokenValue ?? (throw Exception('Missing `sep` processor configuration'));
    final cls = config["cls"]?.tokenValue ?? (throw Exception('Missing `cls` processor configuration'));
    return BertProcessing(sep: sep, cls: cls);
  }
}

class SequenceProcessing extends PostProcessor {
  final List<PostProcessor> processors;

  SequenceProcessing({required this.processors});

  @override
  List<String> postProcess(List<String> tokens, {List<String>? tokensPair, bool addSpecialTokens = true}) {
    var currentTokens = tokens;
    var currentTokensPair = tokensPair;

    for (final processor in processors) {
      final processed = processor.postProcess(currentTokens, tokensPair: currentTokensPair, addSpecialTokens: addSpecialTokens);
      currentTokens = processed;
      currentTokensPair = null;
    }

    return currentTokens;
  }

  factory SequenceProcessing.fromConfig(Config config) {
    final processorConfigs = config["processors"]?.arrayValue ?? (throw Exception('Missing `processors` configuration'));
    final processors = processorConfigs.map((config) => PostProcessorFactory.fromConfig(Config(config))!).toList();
    return SequenceProcessing(processors: processors);
  }
}
