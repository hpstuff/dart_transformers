import 'dart:convert';

import 'package:dart_transformers/hub/config.dart';
import 'package:dart_transformers/hub/hub_api.dart';
import 'package:dart_transformers/tokenizers/utils.dart';
import 'package:flutter/services.dart';

enum RepoType {
  models,
  datasets,
  spaces,
}

class Repo {
  final String id;
  final RepoType type;

  Repo({required this.id, this.type = RepoType.models});
}

class HubClientError extends Error {
  final String _type;
  final Object? message;
  HubClientError(this._type, [this.message]);

  HubClientError.parse()
      : _type = 'HubClientError.parse',
        message = null;
  HubClientError.authorizationRequired()
      : _type = 'HubClientError.authorizationRequired',
        message = null;
  HubClientError.unexpectedError()
      : _type = 'HubClientError.unexpectedError',
        message = null;
  HubClientError.httpStatusCode(this.message) : _type = 'HubClientError.missingVocab';

  @override
  String toString() {
    if (message != null) {
      return "$_type: ${Error.safeToString(message)}";
    }
    return _type;
  }
}

class Hub {
  static Future<List<String>> getFilenames({required Repo repo, List<String> globs = const []}) async {
    return await HubApi.shared.getFilenames(repo, globs: globs);
  }

  static Future<Uri> snapshot({required Repo repo, List<String> globs = const [], void Function(int, double)? progressHandler}) async {
    return await HubApi.shared.snapshot(repo, globs: globs, progressHandler: progressHandler);
  }

  static Future<Config> whoami({required String token}) async {
    return await HubApi(hfToken: token).whoami();
  }

  static Future<FileMetadata> getFileMetadata({required Uri fileURL}) async {
    return await HubApi.shared.getFileMetadata(fileURL);
  }

  static Future<List<FileMetadata>> getFileMetadataFromRepo({required Repo repo, List<String> globs = const []}) async {
    return await HubApi.shared.getFileMetadataFromRepo(repo, globs: globs);
  }
}

class Configurations {
  final Config modelConfig;
  final Config? tokenizerConfig;
  final Config tokenizerData;

  Configurations({
    required this.modelConfig,
    this.tokenizerConfig,
    required this.tokenizerData,
  });
}

class LanguageModelConfigurationFromHub {
  Future<Configurations>? configPromise;

  LanguageModelConfigurationFromHub.fromName({
    required String modelName,
    HubApi? hubApi,
  }) {
    configPromise = loadConfig(modelName: modelName, hubApi: hubApi ?? HubApi.shared);
  }

  LanguageModelConfigurationFromHub.fromFolder({
    required Uri modelFolder,
    HubApi? hubApi,
  }) {
    configPromise = loadConfigFromFolder(modelFolder: modelFolder, hubApi: hubApi ?? HubApi.shared);
  }

  Future<Config> get modelConfig async {
    return (await configPromise)!.modelConfig;
  }

  Future<Config?> get tokenizerConfig async {
    final hubConfig = (await configPromise)!.tokenizerConfig;
    if (hubConfig != null) {
      if (hubConfig["tokenizerClass"] != null) return hubConfig;
      final modelType = await this.modelType;
      if (modelType != null) {
        final fallbackConfig = await fallbackTokenizerConfig(modelType);
        if (fallbackConfig != null) {
          return fallbackConfig.merge(hubConfig);
        }
        return hubConfig.merge(Config({'tokenizer_class': '${modelType.capitalize}Tokenizer'}));
      }
    }
    final modelType = await this.modelType;
    return modelType != null ? await fallbackTokenizerConfig(modelType) : null;
  }

  Future<Config> get tokenizerData async {
    return (await configPromise)!.tokenizerData;
  }

  Future<String?> get modelType async {
    return (await modelConfig)["modelType"]?.stringValue;
  }

  Future<Configurations> loadConfig({
    required String modelName,
    HubApi? hubApi,
  }) async {
    hubApi ??= HubApi.shared;
    final filesToDownload = ['config.json', 'tokenizer_config.json', 'tokenizer.json'];
    final repo = Repo(id: modelName);
    final downloadedModelFolder = await hubApi.snapshot(repo, globs: filesToDownload);
    return loadConfigFromFolder(modelFolder: downloadedModelFolder, hubApi: hubApi);
  }

  Future<Configurations> loadConfigFromFolder({
    required Uri modelFolder,
    HubApi? hubApi,
  }) async {
    hubApi ??= HubApi.shared;
    Config? tokenizerConfig;
    final modelConfig = await hubApi.configuration(modelFolder.resolve('config.json'));
    try {
      tokenizerConfig = await hubApi.configuration(modelFolder.resolve('tokenizer_config.json'));
    } catch (e) {
      tokenizerConfig = null;
    }
    final tokenizerVocab = await hubApi.configuration(modelFolder.resolve('tokenizer.json'));

    return Configurations(
      modelConfig: modelConfig,
      tokenizerConfig: tokenizerConfig,
      tokenizerData: tokenizerVocab,
    );
  }

  static Future<Config?> fallbackTokenizerConfig(String modelType) async {
    final manifest = await AssetManifest.loadFromAssetBundle(rootBundle);
    try {
      final files = manifest.listAssets().where((path) => path.contains("/fallback_config/"));
      final file = files.firstWhere((path) => path.endsWith("fallback_config/${modelType}_tokenizer_config.json"));
      String jsonRaw = await rootBundle.loadString(file);
      jsonRaw = jsonRaw.replaceAll("\\\\n", "\\n"); //Find and replace undesirable instances here
      final parsed = json.decode(jsonRaw);
      if (parsed is Map<String, dynamic>) {
        return Config(parsed);
      }
    } catch (e) {
      return null;
    }
    return null;
  }
}
