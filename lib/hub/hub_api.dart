import 'dart:convert';
import 'dart:io';
import 'package:crypto/crypto.dart';
import 'package:dart_transformers/hub/config.dart';
import 'package:dart_transformers/hub/hub.dart';
import 'package:dart_transformers/tokenizers/utils.dart';
import 'package:http/http.dart' as http;

class HubApi {
  final Uri downloadBase;
  final String? hfToken;
  final String endpoint;
  final bool useBackgroundSession;

  HubApi({
    Uri? downloadBase,
    String? hfToken,
    this.endpoint = 'https://huggingface.co',
    this.useBackgroundSession = false,
  })  : hfToken = hfToken ?? _hfTokenFromEnv(),
        downloadBase = downloadBase ?? Directory.systemTemp.createTempSync('huggingface').uri;

  static final HubApi shared = HubApi();

  static String? _hfTokenFromEnv() {
    final possibleTokens = [
      () => Platform.environment['HF_TOKEN'],
      () => Platform.environment['HUGGING_FACE_HUB_TOKEN'],
      () {
        final path = Platform.environment['HF_TOKEN_PATH'];
        if (path != null) {
          return File(path).readAsStringSync();
        }
        return null;
      },
      () {
        final home = Platform.environment['HF_HOME'];
        if (home != null) {
          return File('$home/token').readAsStringSync();
        }
        return null;
      },
      () => File('${Platform.environment['HOME']}/.cache/huggingface/token').readAsStringSync(),
      () => File('${Platform.environment['HOME']}/.huggingface/token').readAsStringSync(),
    ];
    try {
      return possibleTokens.map((f) => f()).firstWhere((token) => token != null && token.isNotEmpty, orElse: () => null);
    } catch (e) {
      return null;
    }
  }

  Uri localRepoLocation(Repo repo) {
    return downloadBase.resolve('${repo.type.name}/${repo.id}/');
  }

  Future<Map<String, dynamic>> httpGet(Uri url) async {
    final Map<String, String> headers = hfToken != null ? {'Authorization': 'Bearer $hfToken'} : {};
    final response = await http.get(url, headers: headers);
    if (response.statusCode >= 200 && response.statusCode < 300) {
      String jsonRaw = utf8.decode(response.bodyBytes);
      jsonRaw = jsonRaw.replaceAll("\\\\n", "\\n"); //Find and replace undesirable instances here
      return json.decode(jsonRaw);
    } else if (response.statusCode >= 400 && response.statusCode < 500) {
      throw HubClientError.authorizationRequired();
    } else {
      throw HubClientError.httpStatusCode(response.statusCode);
    }
  }

  Future<Map<String, dynamic>> httpHead(Uri url) async {
    final Map<String, String> headers = hfToken != null ? {'Authorization': 'Bearer $hfToken'} : {};
    final response = await head(url, headers: headers);
    if (response.statusCode >= 200 && response.statusCode < 400) {
      return {...response.headers, 'statusCode': response.statusCode.toString()};
    } else if (response.statusCode >= 400 && response.statusCode < 500) {
      throw HubClientError.authorizationRequired();
    } else {
      throw HubClientError.httpStatusCode(response.statusCode);
    }
  }

  Future<List<String>> getFilenames(Repo repo, {List<String> globs = const []}) async {
    final url = Uri.parse('$endpoint/api/models/${repo.id}');
    final response = await httpGet(url);
    final filenames = (response['siblings'] as List).map((e) => e['rfilename'] as String).toList();
    if (globs.isEmpty) {
      return filenames;
    }
    Set<String> selected = {};

    for (final glob in globs) {
      selected.addAll(filenames.matching(glob));
    }

    return selected.toList();
  }

  Future<Config> whoami() async {
    if (hfToken == null) {
      throw HubClientError.authorizationRequired();
    }
    final url = Uri.parse('$endpoint/api/whoami-v2');
    final result = await httpGet(url);
    return Config(result);
  }

  String? normalizeEtag(String? etag) {
    if (etag == null) return null;
    return etag.replaceAll(RegExp(r'^W/'), '').replaceAll('"', '');
  }

  Future<FileMetadata> getFileMetadata(Uri url) async {
    final response = await httpHead(url);
    final location = response['statusCode'] == '302' ? response['location'] : url.toString();

    return FileMetadata(
      commitHash: response['x-repo-commit'],
      etag: normalizeEtag(response['x-linked-etag'] ?? response['etag']),
      location: location,
      size: int.tryParse(response['x-linked-size'] ?? response['content-length'] ?? ''),
    );
  }

  Future<List<FileMetadata>> getFileMetadataFromRepo(Repo repo, {List<String> globs = const []}) async {
    final files = await getFilenames(repo, globs: globs);
    final url = Uri.parse('$endpoint/${repo.id}/resolve/main/'); // TODO: revisions
    List<FileMetadata> selectedMetadata = [];
    for (final file in files) {
      final fileURL = url.resolve(file);
      selectedMetadata.add(await getFileMetadata(fileURL));
    }
    return selectedMetadata;
  }

  Future<Uri> snapshot(Repo repo, {List<String> globs = const [], void Function(int, double)? progressHandler}) async {
    final filenames = await getFilenames(repo, globs: globs);
    final totalFiles = filenames.length;
    final repoDestination = localRepoLocation(repo);

    for (int i = 0; i < totalFiles; i++) {
      final filename = filenames[i];
      final downloader = HubFileDownloader(
        repo: repo,
        repoDestination: repoDestination,
        relativeFilename: filename,
        hfToken: hfToken,
        endpoint: endpoint,
        backgroundSession: useBackgroundSession,
      );
      await downloader.download((fractionDownloaded) {
        if (progressHandler != null) {
          progressHandler((i + fractionDownloaded).floor(), (i + fractionDownloaded) / totalFiles);
        }
      });
    }

    if (progressHandler != null) {
      progressHandler(totalFiles, 1.0);
    }

    return repoDestination;
  }

  Future<Config> configuration(Uri fileURL) async {
    final data = await File.fromUri(fileURL).readAsBytes();
    String jsonRaw = utf8.decode(data);
    jsonRaw = jsonRaw.replaceAll("\\\\n", "\\n"); //Find and replace undesirable instances here
    final parsed = json.decode(jsonRaw);

    if (parsed is Map<String, dynamic>) {
      return Config(parsed);
    } else {
      throw HubClientError.parse();
    }
  }
}

class FileMetadata {
  /// The commit hash related to the file
  final String? commitHash;

  /// Etag of the file on the server
  final String? etag;

  /// Location where to download the file. Can be a Hub url or not (CDN).
  final String location;

  /// Size of the file. In case of an LFS file, contains the size of the actual LFS file, not the pointer.
  final int? size;

  FileMetadata({
    required this.commitHash,
    required this.etag,
    required this.location,
    required this.size,
  });

  @override
  toString() {
    return """FileMetadata(
      commitHash: $commitHash,
      etag: $etag,
      location: $location,
      size: $size
    )""";
  }
}

/// Metadata about a file in the local directory related to a download process
class LocalDownloadFileMetadata {
  /// Commit hash of the file in the repo
  final String commitHash;

  /// ETag of the file in the repo. Used to check if the file has changed.
  /// For LFS files, this is the sha256 of the file. For regular files, it corresponds to the git hash.
  final String etag;

  /// Path of the file in the repo
  final String filename;

  /// The timestamp of when the metadata was saved i.e. when the metadata was accurate
  final DateTime timestamp;

  LocalDownloadFileMetadata({
    required this.commitHash,
    required this.etag,
    required this.filename,
    required this.timestamp,
  });

  @override
  toString() {
    return """LocalDownloadFileMetadata(
      commitHash: $commitHash,
      etag: $etag,
      filename: $filename,
      timestamp: $timestamp
    )""";
  }
}

class HubFileDownloader {
  final Repo repo;
  final Uri repoDestination;
  final String relativeFilename;
  final String? hfToken;
  final String endpoint;
  final bool backgroundSession;

  final String sha256Pattern = r"^[0-9a-f]{64}$";
  final String commitHashPattern = r"^[0-9a-f]{40}$";

  HubFileDownloader({
    required this.repo,
    required this.repoDestination,
    required this.relativeFilename,
    this.hfToken,
    this.endpoint = 'https://huggingface.co',
    this.backgroundSession = false,
  });

  Uri get source {
    var url = Uri.parse(endpoint);
    if (repo.type != RepoType.models) {
      url = url.resolve(repo.type.name);
    }
    url = url.resolve('${repo.id}/resolve/main/$relativeFilename');
    return url;
  }

  File get destination {
    return File('${repoDestination.path}/$relativeFilename');
  }

  Directory get metadataDestination {
    return Directory('${repoDestination.path}/.cache/huggingface/download');
  }

  bool get downloaded {
    return destination.existsSync();
  }

  Future<void> prepareDestination() async {
    final directory = destination.parent;
    if (!directory.existsSync()) {
      directory.createSync(recursive: true);
    }
  }

  Future<void> prepareMetadataDestination() async {
    if (!metadataDestination.existsSync()) {
      metadataDestination.createSync(recursive: true);
    }
  }

  Future<LocalDownloadFileMetadata?> readDownloadMetadata(Directory localDir, String filePath) async {
    final metadataPath = File('${localDir.path}/$filePath');
    if (metadataPath.existsSync()) {
      try {
        final contents = await metadataPath.readAsString();
        final lines = contents.split('\n');
        if (lines.length < 3) {
          throw Exception('Metadata file is missing required fields.');
        }
        final commitHash = lines[0].trim();
        final etag = lines[1].trim();
        final timestamp = double.tryParse(lines[2].trim());
        if (timestamp == null) {
          throw Exception('Missing or invalid timestamp.');
        }
        final timestampDate = DateTime.fromMillisecondsSinceEpoch((timestamp * 1000).toInt());
        return LocalDownloadFileMetadata(
          commitHash: commitHash,
          etag: etag,
          filename: filePath,
          timestamp: timestampDate,
        );
      } catch (e) {
        metadataPath.deleteSync();
        return null;
      }
    }
    return null;
  }

  bool isValidHash(String hash, String pattern) {
    final regex = RegExp(pattern);
    return regex.hasMatch(hash);
  }

  Future<void> writeDownloadMetadata(String commitHash, String etag, String metadataRelativePath) async {
    final metadataContent = '$commitHash\n$etag\n${DateTime.now().millisecondsSinceEpoch / 1000}\n';
    final metadataPath = File('${metadataDestination.path}/$metadataRelativePath');
    if (!metadataPath.parent.existsSync()) {
      metadataPath.parent.createSync(recursive: true);
    }
    await metadataPath.writeAsString(metadataContent);
  }

  Future<String> computeFileHash(File file) async {
    final bytes = await file.readAsBytes();
    final digest = sha256.convert(bytes);
    return digest.toString();
  }

  Future<File> download(void Function(double) progressHandler) async {
    final metadataRelativePath = '$relativeFilename.metadata';
    final localMetadata = await readDownloadMetadata(metadataDestination, metadataRelativePath);
    final remoteMetadata = await HubApi.shared.getFileMetadata(source);

    final localCommitHash = localMetadata?.commitHash ?? '';
    final remoteCommitHash = remoteMetadata.commitHash ?? '';

    if (isValidHash(remoteCommitHash, commitHashPattern) && downloaded && localMetadata != null && localCommitHash == remoteCommitHash) {
      return destination;
    }

    if (downloaded) {
      if (localMetadata?.etag == remoteMetadata.etag) {
        await writeDownloadMetadata(remoteCommitHash, remoteMetadata.etag!, metadataRelativePath);
        return destination;
      }

      if (isValidHash(remoteMetadata.etag!, sha256Pattern)) {
        final fileHash = await computeFileHash(destination);
        if (fileHash == remoteMetadata.etag) {
          await writeDownloadMetadata(remoteCommitHash, remoteMetadata.etag!, metadataRelativePath);
          return destination;
        }
      }
    }

    await prepareDestination();
    await prepareMetadataDestination();

    final request = http.Request('GET', source);
    if (hfToken != null) {
      request.headers['Authorization'] = 'Bearer $hfToken';
    }
    final response = await request.send();

    if (response.statusCode != 200) {
      throw HubClientError.httpStatusCode(response.statusCode);
    }

    final totalBytes = response.contentLength ?? 0;
    int downloadedBytes = 0;
    final sink = destination.openWrite();

    await for (final chunk in response.stream) {
      downloadedBytes += chunk.length;
      sink.add(chunk);
      if (totalBytes != 0) {
        progressHandler(downloadedBytes / totalBytes);
      }
    }
    await sink.close();

    await writeDownloadMetadata(remoteCommitHash, remoteMetadata.etag!, metadataRelativePath);
    return destination;
  }
}
