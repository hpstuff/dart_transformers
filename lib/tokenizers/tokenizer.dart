import 'dart:convert';
import 'dart:math';

import 'package:dart_transformers/hub/config.dart';
import 'package:dart_transformers/hub/hub.dart';
import 'package:dart_transformers/hub/hub_api.dart';
import 'package:dart_transformers/tokenizers/bert_tokenizer.dart';
import 'package:dart_transformers/tokenizers/bpe_tokenizer.dart';
import 'package:dart_transformers/tokenizers/decoder.dart';
import 'package:dart_transformers/tokenizers/normalizer.dart';
import 'package:dart_transformers/tokenizers/post_processor.dart';
import 'package:dart_transformers/tokenizers/pre_tokenizer.dart';
import 'package:dart_transformers/tokenizers/tokenizer_error.dart';
import 'package:dart_transformers/tokenizers/unigram_tokenizer.dart';
import 'package:dart_transformers/tokenizers/utils.dart';
import 'package:jinja/jinja.dart';

abstract class TokenizingModel {
  List<String> tokenize(String text);

  int? convertTokenToId(String token);

  String? convertIdToToken(int id);

  List<String> call(String text) {
    return tokenize(text);
  }

  List<int?> convertTokensToIds(List<String> tokens) {
    return tokens.map((token) => convertTokenToId(token)).toList();
  }

  List<String?> convertIdsToTokens(List<int> ids) {
    return ids.map((id) => convertIdToToken(id)).toList();
  }

  String? get bosToken;
  int? get bosTokenId;
  String? get eosToken;
  int? get eosTokenId;
  String? get unknownToken;
  int? get unknownTokenId;

  bool get fuseUnknownTokens;
}

abstract class PreTrainedTokenizerModel extends TokenizingModel {
  PreTrainedTokenizerModel();
  PreTrainedTokenizerModel.fromConfig(Config tokenizerConfig, Config tokenizerData, Map<String, int> addedTokens) {
    throw TokenizerError.unsupportedTokenizer("PreTrainedTokenizer");
  }
}

class TokenizerModel {
  static String? unknownToken(Config tokenizerConfig) {
    return tokenizerConfig["unkToken"]?["content"]?.stringValue ?? tokenizerConfig["unkToken"]?.stringValue;
  }

  static TokenizingModel from(Config tokenizerConfig, Config tokenizerData, Map<String, int> addedTokens) {
    final tokenizerClassName = tokenizerConfig["tokenizerClass"]?.stringValue;
    if (tokenizerClassName == null) {
      throw TokenizerError.missingTokenizerClassInConfig();
    }

    final tokenizerName = tokenizerClassName.replaceAll("Fast", "");

    switch (tokenizerName) {
      case "BertTokenizer":
      case "DistilbertTokenizer":
      case "DistilBertTokenizer":
        return BertTokenizer.fromConfig(tokenizerConfig, tokenizerData, addedTokens);
      case "CodeGenTokenizer":
        return CodeGenTokenizer.fromConfig(tokenizerConfig, tokenizerData, addedTokens);
      case "CodeLlamaTokenizer":
        return CodeLlamaTokenizer.fromConfig(tokenizerConfig, tokenizerData, addedTokens);
      case "FalconTokenizer":
        return FalconTokenizer.fromConfig(tokenizerConfig, tokenizerData, addedTokens);
      case "GemmaTokenizer":
        return GemmaTokenizer.fromConfig(tokenizerConfig, tokenizerData, addedTokens);
      case "GPT2Tokenizer":
      case "Gpt2Tokenizer":
        return GPT2Tokenizer.fromConfig(tokenizerConfig, tokenizerData, addedTokens);
      case "LlamaTokenizer":
        return LlamaTokenizer.fromConfig(tokenizerConfig, tokenizerData, addedTokens);
      case "T5Tokenizer":
        return T5Tokenizer.fromConfig(tokenizerConfig, tokenizerData, addedTokens);
      case "WhisperTokenizer":
        return WhisperTokenizer.fromConfig(tokenizerConfig, tokenizerData, addedTokens);
      case "CohereTokenizer":
        return CohereTokenizer.fromConfig(tokenizerConfig, tokenizerData, addedTokens);
      case "Qwen2Tokenizer":
        return Qwen2Tokenizer.fromConfig(tokenizerConfig, tokenizerData, addedTokens);
      case "PreTrainedTokenizer":
        return BPETokenizer.fromConfig(tokenizerConfig, tokenizerData, addedTokens);
      default:
        throw TokenizerError.unsupportedTokenizer(tokenizerName);
    }
  }
}

enum ChatTemplateArgumentType { literal, name }

class ChatTemplateArgument {
  final ChatTemplateArgumentType type;
  final String value;

  ChatTemplateArgument.literal(this.value) : type = ChatTemplateArgumentType.literal;
  ChatTemplateArgument.name(this.value) : type = ChatTemplateArgumentType.name;
}

typedef Message = Map<String, dynamic>;
typedef ToolSpec = Map<String, dynamic>;

abstract class Tokenizer {
  List<String> tokenize(String text);

  /// Main entry point
  List<int> encode(String text, {bool addSpecialTokens = true});
  List<int> call(String text, bool addSpecialTokens) {
    return encode(text, addSpecialTokens: addSpecialTokens);
  }

  /// Decode
  String decode(List<int> tokens, {bool skipSpecialTokens = false});

  int? convertTokenToId(String token);
  List<int?> convertTokensToIds(List<String> tokens) {
    return tokens.map((token) => convertTokenToId(token)).toList();
  }

  String? convertIdToToken(int id);
  List<String?> convertIdsToTokens(List<int> ids) {
    return ids.map((id) => convertIdToToken(id)).toList();
  }

  String? get bosToken;
  int? get bosTokenId;
  String? get eosToken;
  int? get eosTokenId;
  String? get unknownToken;
  int? get unknownTokenId;
  bool get fuseUnknownTokens;

  bool hasChatTemplate = false;

  List<int> applyChatTemplate(
    List<Message> messages, {
    ChatTemplateArgument? chatTemplate,
    List<ToolSpec>? tools,
    Map<String, dynamic>? additionalContext,
    bool addGenerationPrompt,
    bool truncation,
    int? maxLength,
  });
}

const List<String> specialTokenAttributes = [
  "bos_token",
  "eos_token",
  "unk_token",
  "sep_token",
  "pad_token",
  "cls_token",
  "mask_token",
  "additional_special_tokens"
];

class PreTrainedTokenizer extends Tokenizer {
  late final TokenizingModel model;

  @override
  String? get bosToken => model.bosToken;
  @override
  int? get bosTokenId => model.bosTokenId;
  @override
  String? get eosToken => model.eosToken;
  @override
  int? get eosTokenId => model.eosTokenId;
  @override
  String? get unknownToken => model.unknownToken;
  @override
  int? get unknownTokenId => model.unknownTokenId;
  @override
  bool get fuseUnknownTokens => model.fuseUnknownTokens;

  final Set<String> addedTokens;
  final Map<String, int> specialTokens;
  RegExp? addedTokensRegex;

  PreTokenizer? preTokenizer;
  Normalizer? normalizer;
  PostProcessor? postProcessor;
  Decoder? decoder;
  final Config tokenizerConfig;

  final bool cleanUpTokenizationSpaces;

  PreTrainedTokenizer({
    required this.tokenizerConfig,
    required Config tokenizerData,
  })  : addedTokens = {},
        specialTokens = {},
        addedTokensRegex = null,
        preTokenizer = null,
        normalizer = null,
        postProcessor = null,
        decoder = null,
        cleanUpTokenizationSpaces = tokenizerConfig["cleanUpTokenizationSpaces"]?.boolValue ?? true {
    // Initialize addedTokens and specialTokens
    final addedTokensMap = <String, int>{};
    final specialTokensMap = <String, int>{};
    for (final addedToken in tokenizerData["addedTokens"]?.arrayValue ?? []) {
      final int? id = addedToken["id"];
      final String? content = addedToken["content"];
      if (id != null && content != null) {
        addedTokensMap[content] = id;
        if (addedToken["special"] ?? false) {
          specialTokensMap[content] = id;
        }
      }
    }
    addedTokens.addAll(addedTokensMap.keys);
    specialTokens.addAll(specialTokensMap);

    // Create addedTokensRegex
    final unwrappedAddedTokens = (tokenizerData["addedTokens"]?.arrayValue ?? [])
        .map((addedToken) {
          final String? content = addedToken["content"];
          final bool prefix = addedToken["lstrip"] ?? false;
          final bool suffix = addedToken["rstrip"] ?? false;
          return content != null ? (content, prefix, suffix) : null;
        })
        .where((token) => token != null)
        .map((token) => token!)
        .toList()
      ..sort((a, b) => b.$1.length.compareTo(a.$1.length));

    final addedTokensRegexString = unwrappedAddedTokens.map((token) {
      final escapedToken = RegExp.escape(token.$1);
      final prefix = token.$2 ? r'\s*' : '';
      final suffix = token.$3 ? r'\s*' : '';
      return '$prefix($escapedToken)$suffix';
    }).join('|');
    addedTokensRegex = addedTokensRegexString.isNotEmpty ? RegExp(addedTokensRegexString) : null;

    // Initialize other components
    preTokenizer = PreTokenizerFactory.fromConfig(tokenizerData["preTokenizer"]);
    normalizer = NormalizerFactory.fromConfig(tokenizerData["normalizer"]);
    postProcessor = PostProcessorFactory.fromConfig(tokenizerData["postProcessor"]);
    decoder = DecoderFactory.fromConfig(tokenizerData["decoder"], addedTokens);
    model = TokenizerModel.from(tokenizerConfig, tokenizerData, addedTokensMap);
  }

  List<String> preTokenize(String text, PreTokenizerOptions options) {
    return preTokenizer?.callSingle(text, options) ?? [text];
  }

  String normalize(String text) {
    return normalizer?.call(text) ?? text;
  }

  List<String> postProcess(List<String> tokens, {bool addSpecialTokens = true}) {
    return postProcessor?.call(tokens, addSpecialTokens: addSpecialTokens) ?? tokens;
  }

  List<String> decodeTokens(List<String> tokens) {
    return decoder?.call(tokens) ?? tokens;
  }

  String cleanUp(String text) {
    if (!cleanUpTokenizationSpaces) return text;
    return text
        .replaceAll(' .', '.')
        .replaceAll(' ?', '?')
        .replaceAll(' !', '!')
        .replaceAll(' ,', ',')
        .replaceAll(" ' ", "'")
        .replaceAll(" n't", "n't")
        .replaceAll(" 'm", "'m")
        .replaceAll(" 's", "'s")
        .replaceAll(" 've", "'ve")
        .replaceAll(" 're", "'re");
  }

  List<String> fuseUnknown(List<String> tokens) {
    if (!fuseUnknownTokens) return tokens;
    final fused = <String>[];
    var previousIsUnknown = false;
    for (final token in tokens) {
      final isUnknown = model.convertTokenToId(token) == model.unknownTokenId;
      if (isUnknown) {
        if (!previousIsUnknown) fused.add(token);
      } else {
        fused.add(token);
      }
      previousIsUnknown = isUnknown;
    }
    return fused;
  }

  @override
  List<String> tokenize(String text) {
    final sections = addedTokensRegex != null
        ? text.splitWithDelimiterBehavior(
            addedTokensRegex!,
            behavior: DelimiterBehavior.isolate,
          )
        : [text];

    final List<List<String>> tokens = [];
    for (var i = 0; i < sections.length; i++) {
      final section = sections[i];
      if (addedTokens.contains(section)) {
        tokens.add([section]);
      } else {
        tokens.addAll(preTokenize(
          normalize(section),
          i == 0 ? const {PreTokenizerOption.firstSection} : const {},
        ).map((token) => model.call(token)).toList());
      }
    }
    return tokens.expand((tokens) => fuseUnknown(tokens)).toList();
  }

  @override
  List<int> encode(String text, {bool addSpecialTokens = true}) {
    final tokenized = tokenize(text);
    final processed = postProcess(tokenized, addSpecialTokens: addSpecialTokens);
    return processed.map((token) => convertTokenToId(token)!).toList();
  }

  @override
  String decode(List<int> tokens, {bool skipSpecialTokens = false}) {
    final tokenStrings = skipSpecialTokens
        ? tokens.where((id) => !specialTokens.values.contains(id)).map((id) => model.convertIdToToken(id)).toList()
        : tokens.map((id) => model.convertIdToToken(id)).toList();
    final decoded = decodeTokens(tokenStrings.where((token) => token != null).toList().cast<String>());
    return cleanUp(decoded.join(''));
  }

  @override
  int? convertTokenToId(String token) {
    return model.convertTokenToId(token);
  }

  @override
  String? convertIdToToken(int id) {
    return model.convertIdToToken(id);
  }

  @override
  bool get hasChatTemplate => tokenizerConfig["chatTemplate"] != null;

  @override
  List<int> applyChatTemplate(List<Message> messages,
      {ChatTemplateArgument? chatTemplate,
      List<ToolSpec>? tools,
      Map<String, dynamic>? additionalContext,
      bool addGenerationPrompt = true,
      bool truncation = false,
      int? maxLength}) {
    String? selectedChatTemplate;

    if (chatTemplate?.type == ChatTemplateArgumentType.literal) {
      selectedChatTemplate = chatTemplate?.value;
    } else if (tokenizerConfig["chatTemplate"] != null) {
      final valueFromConfig = tokenizerConfig["chatTemplate"]?.value;
      if (valueFromConfig is List) {
        final templateDict = Map.fromEntries((valueFromConfig).map((item) {
          final String? name = item["name"];
          final String? template = item["template"];
          return MapEntry(name, template);
        }).where((entry) => entry.key != null && entry.value != null));

        if (chatTemplate?.type == ChatTemplateArgumentType.name) {
          selectedChatTemplate = templateDict[chatTemplate?.value];
          if (selectedChatTemplate == null) {
            throw TokenizerError.chatTemplate('No chat template named "${chatTemplate?.value}" was found in the tokenizer config');
          }
        } else if (tools != null && tools.isNotEmpty && templateDict.containsKey("tool_use")) {
          selectedChatTemplate = templateDict["tool_use"];
        } else if (templateDict.containsKey("default")) {
          selectedChatTemplate = templateDict["default"];
        }
      } else if (valueFromConfig is String) {
        selectedChatTemplate = valueFromConfig;
      }
    }

    if (selectedChatTemplate == null) {
      throw TokenizerError.chatTemplate('This tokenizer does not have a chat template, and no template was passed.');
    }

    Template template = Template(utf8.decode(utf8.encode(selectedChatTemplate)), keepTrailingNewLine: false);

    final context = {
      "messages": messages,
      "add_generation_prompt": addGenerationPrompt,
      if (tools != null) "tools": tools,
      if (additionalContext != null) ...additionalContext,
    };

    final specialTokenAttributesConfig = tokenizerConfig.dictionary.entries.where(
      (entry) => specialTokenAttributes.contains(entry.key) && entry.value != null,
    );

    for (var entry in specialTokenAttributesConfig) {
      final key = entry.key;
      final value = entry.value;
      if (value is String) {
        context[key] = value;
      } else if (value is Map) {
        context[key] = Config(value.cast()).addedTokenAsString;
      } else {
        context[key] = value;
      }
    }

    final rendered = template.render(context);

    var encodedTokens = encode(rendered, addSpecialTokens: false);
    maxLength ??= encodedTokens.length;
    maxLength = min(maxLength, tokenizerConfig["modelMaxLength"]?.intValue ?? maxLength);
    if (encodedTokens.length > maxLength) {
      if (truncation) {
        encodedTokens = encodedTokens.sublist(0, maxLength);
      }
    }

    return encodedTokens;
  }

  // List<int> applyChatTemplate(
  //   List<Message> messages, {
  //   ChatTemplateArgument? chatTemplate,
  //   List<ToolSpec>? tools,
  //   Map<String, dynamic>? additionalContext,
  //   bool addGenerationPrompt = false,
  //   bool truncation = false,
  //   int? maxLength,
  // }) {
  //   String? selectedChatTemplate;
  //   if (chatTemplate != null && chatTemplate.type == ChatTemplateArgumentType.literal) {
  //     selectedChatTemplate = chatTemplate.value;
  //   } else if (tokenizerConfig["chatTemplate"] != null) {
  //     final valueFromConfig = tokenizerConfig["chatTemplate"];
  //     if (valueFromConfig is List) {
  //       final templateDict = Map.fromEntries(valueFromConfig.map((item) {
  //         final name = item["name"]?.stringValue;
  //         final template = item["template"]?.stringValue;
  //         return MapEntry(name, template);
  //       }).where((entry) => entry.key != null && entry.value != null));
  //       if (chatTemplate != null && chatTemplate.type == ChatTemplateArgumentType.name) {
  //         selectedChatTemplate = templateDict[chatTemplate.value];
  //         if (selectedChatTemplate == null) {
  //           throw TokenizerError.chatTemplate('No chat template named "${chatTemplate.value}" was found in the tokenizer config');
  //         }
  //       } else if (tools != null && tools.isNotEmpty && templateDict.containsKey("tool_use")) {
  //         selectedChatTemplate = templateDict["tool_use"];
  //       } else if (templateDict.containsKey("default")) {
  //         selectedChatTemplate = templateDict["default"];
  //       }
  //     } else if (valueFromConfig is String) {
  //       selectedChatTemplate = valueFromConfig;
  //     }
  //   }

  //   if (selectedChatTemplate == null) {
  //     throw TokenizerError.chatTemplate('This tokenizer does not have a chat template, and no template was passed.');
  //   }

  //   final template = Template(selectedChatTemplate);
  //   final context = {
  //     "messages": messages,
  //     "add_generation_prompt": addGenerationPrompt,
  //     if (tools != null) "tools": tools,
  //     if (additionalContext != null) ...additionalContext,
  //     ...tokenizerConfig.dictionary.entries.where((entry) => specialTokenAttributes.contains(entry.key) && entry.value != null).map((entry) {
  //       final key = entry.key;
  //       final value = entry.value;
  //       if (value is String) {
  //         return MapEntry(key, value);
  //       } else if (value is Map) {
  //         return MapEntry(key, addedTokenAsString(Config(value)));
  //       } else {
  //         return MapEntry(key, value);
  //       }
  //     }).toMap(),
  //   };

  //   final rendered = template.render(context);
  //   var encodedTokens = encode(rendered, addSpecialTokens: false);
  //   maxLength ??= encodedTokens.length;
  //   maxLength = min(maxLength, tokenizerConfig["modelMaxLength"]?.intValue ?? maxLength);
  //   if (encodedTokens.length > maxLength) {
  //     if (truncation) {
  //       encodedTokens = encodedTokens.sublist(0, maxLength);
  //     }
  //   }

  //   return encodedTokens;
  // }
}

class GPT2Tokenizer extends BPETokenizer {
  GPT2Tokenizer.fromConfig(super.tokenizerConfig, super.tokenizerData, super.addedTokens) : super.fromConfig();
}

class FalconTokenizer extends BPETokenizer {
  FalconTokenizer.fromConfig(super.tokenizerConfig, super.tokenizerData, super.addedTokens) : super.fromConfig();
}

class LlamaTokenizer extends BPETokenizer {
  LlamaTokenizer.fromConfig(super.tokenizerConfig, super.tokenizerData, super.addedTokens) : super.fromConfig();
}

class CodeGenTokenizer extends BPETokenizer {
  CodeGenTokenizer.fromConfig(super.tokenizerConfig, super.tokenizerData, super.addedTokens) : super.fromConfig();
}

class WhisperTokenizer extends BPETokenizer {
  WhisperTokenizer.fromConfig(super.tokenizerConfig, super.tokenizerData, super.addedTokens) : super.fromConfig();
}

class GemmaTokenizer extends BPETokenizer {
  GemmaTokenizer.fromConfig(super.tokenizerConfig, super.tokenizerData, super.addedTokens) : super.fromConfig();
}

class CodeLlamaTokenizer extends BPETokenizer {
  CodeLlamaTokenizer.fromConfig(super.tokenizerConfig, super.tokenizerData, super.addedTokens) : super.fromConfig();
}

class CohereTokenizer extends BPETokenizer {
  CohereTokenizer.fromConfig(super.tokenizerConfig, super.tokenizerData, super.addedTokens) : super.fromConfig();
}

class Qwen2Tokenizer extends BPETokenizer {
  Qwen2Tokenizer.fromConfig(super.tokenizerConfig, super.tokenizerData, super.addedTokens) : super.fromConfig();
}

class T5Tokenizer extends UnigramTokenizer {
  T5Tokenizer.fromConfig(super.tokenizerConfig, super.tokenizerData, super.addedTokens) : super.fromConfig();
}

class AutoTokenizer {
  static Tokenizer from(Config tokenizerConfig, Config tokenizerData) {
    final tokenizerName = tokenizerConfig["tokenizerClass"]?.stringValue?.replaceAll("Fast", "");
    switch (tokenizerName) {
      case "LlamaTokenizer":
        return LlamaPreTrainedTokenizer(tokenizerConfig: tokenizerConfig, tokenizerData: tokenizerData);
      default:
        return PreTrainedTokenizer(tokenizerConfig: tokenizerConfig, tokenizerData: tokenizerData);
    }
  }

  static Future<Tokenizer> fromPretrained(String model, {HubApi? hubApi}) async {
    hubApi ??= HubApi.shared;
    final config = LanguageModelConfigurationFromHub.fromName(modelName: model, hubApi: hubApi);
    final tokenizerConfig = await config.tokenizerConfig;
    if (tokenizerConfig == null) {
      throw TokenizerError.missingConfig();
    }
    final tokenizerData = await config.tokenizerData;

    return AutoTokenizer.from(tokenizerConfig, tokenizerData);
  }

  static Future<Tokenizer> fromModelFolder(Uri modelFolder, {HubApi? hubApi}) async {
    hubApi ??= HubApi.shared;
    final config = LanguageModelConfigurationFromHub.fromFolder(modelFolder: modelFolder, hubApi: hubApi);
    final tokenizerConfig = await config.tokenizerConfig;
    if (tokenizerConfig == null) {
      throw TokenizerError.missingConfig();
    }
    final tokenizerData = await config.tokenizerData;

    return AutoTokenizer.from(tokenizerConfig, tokenizerData);
  }
}

Config? maybeUpdatePostProcessor({
  required Config tokenizerConfig,
  Config? processorConfig,
}) {
  // If it's already a Template processor (instead of a ByteLevel one), assume it's correct
  final postProcessor = PostProcessorFactory.fromConfig(processorConfig);
  if (postProcessor is TemplateProcessing) return null;

  final addBosToken = tokenizerConfig["addBosToken"]?.boolValue ?? false;
  final bosToken = tokenizerConfig["bosToken"]?.addedTokenAsString;
  if (addBosToken && bosToken == null) {
    throw TokenizerError.mismatchedConfig("add_bos_token is True but bos_token is null");
  }

  final addEosToken = tokenizerConfig["addEosToken"]?.boolValue ?? false;
  final eosToken = tokenizerConfig["eosToken"]?.addedTokenAsString;
  if (addEosToken && eosToken == null) {
    throw TokenizerError.mismatchedConfig("add_eos_token is True but eos_token is null");
  }

  // alt implementation
  var single = <Map<String, dynamic>>[];
  if (addBosToken) {
    single.add({
      "SpecialToken": {"id": bosToken!, "type_id": 0}
    });
  }
  single.add({
    "Sequence": {"id": "A", "type_id": 0}
  });
  if (addEosToken) {
    single.add({
      "SpecialToken": {"id": eosToken!, "type_id": 0}
    });
  }

  var pair = List<Map<String, dynamic>>.from(single);
  if (addBosToken) {
    pair.add({
      "SpecialToken": {"id": bosToken!, "type_id": 1}
    });
  }
  pair.add({
    "Sequence": {"id": "B", "type_id": 1}
  });
  if (addEosToken) {
    pair.add({
      "SpecialToken": {"id": eosToken!, "type_id": 1}
    });
  }

  final postProcessorConfig = Config({
    "type": "TemplateProcessing",
    "single": single,
    "pair": pair,
  });
  return postProcessorConfig;
}

class LlamaPreTrainedTokenizer extends PreTrainedTokenizer {
  final bool isLegacy;

  LlamaPreTrainedTokenizer({
    required super.tokenizerConfig,
    required Config tokenizerData,
  })  : isLegacy = tokenizerConfig["legacy"]?.boolValue ?? true,
        super(
          tokenizerData: _updateTokenizerData(tokenizerConfig, tokenizerData),
        );

  static Config _updateTokenizerData(Config tokenizerConfig, Config tokenizerData) {
    final configDictionary = Map<String, dynamic>.from(tokenizerData.dictionary);
    if (!(tokenizerConfig["legacy"]?.boolValue ?? true)) {
      configDictionary.remove("normalizer");
      configDictionary["pre_tokenizer"] = {
        "type": "Metaspace",
        "replacement": "▁",
        "add_prefix_space": true,
        "prepend_scheme": "first",
      };
    }

    final postProcessorConfig = maybeUpdatePostProcessor(
      tokenizerConfig: tokenizerConfig,
      processorConfig: tokenizerData["post_processor"],
    );
    if (postProcessorConfig != null) {
      configDictionary["post_processor"] = postProcessorConfig.dictionary;
    }

    return Config(configDictionary);
  }
}
