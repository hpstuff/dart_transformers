import 'dart:io';

import 'package:dart_transformers/hub/hub.dart';
import 'package:dart_transformers/hub/hub_api.dart';
import 'package:flutter_test/flutter_test.dart';

void main() {
  setUp(() {
    // Put setup code here. This method is called before the invocation of each test method in the class.
  });

  tearDown(() {
    // Put teardown code here. This method is called after the invocation of each test method in the class.
  });

  group('HubApiTests', () {
    test('testFilenameRetrieval', () async {
      try {
        final filenames = await Hub.getFilenames(repo: Repo(id: 'coreml-projects/Llama-2-7b-chat-coreml'));
        expect(filenames.length, 13);
      } catch (error) {
        fail('$error');
      }
    });

    test('testFilenameRetrievalWithGlob', () async {
      try {
        final filenames1 = await Hub.getFilenames(
          repo: Repo(id: 'coreml-projects/Llama-2-7b-chat-coreml'),
          globs: ['*.json'],
        );
        expect(
          filenames1.toSet(),
          {
            'config.json',
            'tokenizer.json',
            'tokenizer_config.json',
            'llama-2-7b-chat.mlpackage/Manifest.json',
            'llama-2-7b-chat.mlpackage/Data/com.apple.CoreML/FeatureDescriptions.json',
            'llama-2-7b-chat.mlpackage/Data/com.apple.CoreML/Metadata.json',
          }.toSet(),
        );

        // Glob patterns are case sensitive
        final filenames2 = await Hub.getFilenames(
          repo: Repo(id: 'coreml-projects/Llama-2-7b-chat-coreml'),
          globs: ['*.JSON'],
        );
        expect(filenames2, []);
      } catch (error) {
        fail('$error');
      }
    });

    test('testFilenameRetrievalFromDirectories', () async {
      try {
        // Contents of all directories matching a pattern
        final filenames = await Hub.getFilenames(
          repo: Repo(id: 'coreml-projects/Llama-2-7b-chat-coreml'),
          globs: ['*.mlpackage/*'],
        );
        expect(
          filenames.toSet(),
          {
            'llama-2-7b-chat.mlpackage/Manifest.json',
            'llama-2-7b-chat.mlpackage/Data/com.apple.CoreML/FeatureDescriptions.json',
            'llama-2-7b-chat.mlpackage/Data/com.apple.CoreML/Metadata.json',
            'llama-2-7b-chat.mlpackage/Data/com.apple.CoreML/model.mlmodel',
            'llama-2-7b-chat.mlpackage/Data/com.apple.CoreML/weights/weight.bin',
          }.toSet(),
        );
      } catch (error) {
        fail('$error');
      }
    });

    test('testFilenameRetrievalWithMultiplePatterns', () async {
      try {
        final patterns = ['config.json', 'tokenizer.json', 'tokenizer_*.json'];
        final filenames = await Hub.getFilenames(
          repo: Repo(id: 'coreml-projects/Llama-2-7b-chat-coreml'),
          globs: patterns,
        );
        expect(
          filenames.toSet(),
          {'config.json', 'tokenizer.json', 'tokenizer_config.json'}.toSet(),
        );
      } catch (error) {
        fail('$error');
      }
    });

    test('testGetFileMetadata', () async {
      try {
        final url = Uri.parse('https://huggingface.co/coreml-projects/Llama-2-7b-chat-coreml/resolve/main/config.json');
        final metadata = await Hub.getFileMetadata(fileURL: url);

        expect(metadata.commitHash, isNotNull);
        expect(metadata.etag, isNotNull);
        expect(metadata.location, url.toString());
        expect(metadata.size, 163);
      } catch (error) {
        fail('$error');
      }
    });

    test('testGetFileMetadataBlobPath', () async {
      try {
        final url = Uri.parse('https://huggingface.co/coreml-projects/Llama-2-7b-chat-coreml/resolve/main/config.json');
        final metadata = await Hub.getFileMetadata(fileURL: url);

        expect(metadata.commitHash, isNotNull);
        expect(metadata.etag, isNotNull);
        expect(metadata.etag!.startsWith('d6ceb9'), isTrue);
        expect(metadata.location, url.toString());
        expect(metadata.size, 163);
      } catch (error) {
        fail('$error');
      }
    });

    test('testGetFileMetadataWithRevision', () async {
      try {
        final revision = 'f2c752cfc5c0ab6f4bdec59acea69eefbee381c2';
        final url = Uri.parse('https://huggingface.co/julien-c/dummy-unknown/resolve/$revision/config.json');
        final metadata = await Hub.getFileMetadata(fileURL: url);

        expect(metadata.commitHash, revision);
        expect(metadata.etag, isNotNull);
        expect(metadata.etag!.isNotEmpty, isTrue);
        expect(metadata.location, url.toString());
        expect(metadata.size, 851);
      } catch (error) {
        fail('$error');
      }
    });

    test('testGetFileMetadataWithBlobSearch', () async {
      try {
        final repo = Repo(id: 'coreml-projects/Llama-2-7b-chat-coreml');
        final metadataFromBlob = (await Hub.getFileMetadataFromRepo(repo: repo, globs: ['*.json'])).toList()..sort((a, b) => a.location.compareTo(b.location));
        final files = (await Hub.getFilenames(repo: repo, globs: ['*.json'])).toList()..sort();
        for (var i = 0; i < metadataFromBlob.length; i++) {
          final metadata = metadataFromBlob[i];
          final file = files[i];
          expect(metadata.commitHash, isNotNull);
          expect(metadata.etag, isNotNull);
          expect(metadata.etag!.isNotEmpty, isTrue);
          expect(metadata.location.contains(file), isTrue);
          expect(metadata.size! > 0, isTrue);
        }
      } catch (error) {
        fail('$error');
      }
    });

    test('testGetLargeFileMetadata', () async {
      try {
        final revision = 'eaf97358a37d03fd48e5a87d15aff2e8423c1afb';
        final etag = 'fc329090bfbb2570382c9af997cffd5f4b78b39b8aeca62076db69534e020107';
        final location =
            'https://cdn-lfs.hf.co/repos/4a/4e/4a4e587f66a2979dcd75e1d7324df8ee9ef74be3582a05bea31c2c26d0d467d0/fc329090bfbb2570382c9af997cffd5f4b78b39b8aeca62076db69534e020107?response-content-disposition=inline%3B+filename*%3DUTF-8%27%27model.mlmodel%3B+filename%3D%22model.mlmodel';
        final size = 504766;

        final url = Uri.parse(
            'https://huggingface.co/coreml-projects/Llama-2-7b-chat-coreml/resolve/main/llama-2-7b-chat.mlpackage/Data/com.apple.CoreML/model.mlmodel');
        final metadata = await Hub.getFileMetadata(fileURL: url);

        expect(metadata.commitHash, revision);
        expect(metadata.etag, etag);
        expect(metadata.location.contains(location), isTrue);
        expect(metadata.size, size);
      } catch (error) {
        fail('$error');
      }
    });
  });

  group("SnapshotDownloadTests", () {
    final repo = Repo(id: 'coreml-projects/Llama-2-7b-chat-coreml');
    final lfsRepo = Repo(id: 'pcuenq/smol-lfs');
    final downloadDestination = Directory.systemTemp.createTempSync('huggingface-tests');

    setUp(() {});

    tearDown(() {
      try {
        downloadDestination.deleteSync(recursive: true);
      } catch (error) {
        rethrow;
      }
    });

    List<String> getRelativeFiles(Directory dir, String repo, {bool skipHidden = true}) {
      final filenames = <String>[];
      final prefix = '${downloadDestination.path}/models/$repo/';

      for (var entity in dir.listSync(recursive: true)) {
        if (entity is File) {
          if (entity.path.contains(".cache") && skipHidden) {
            continue;
          }
          final relativePath = entity.path.substring(prefix.length);
          filenames.add(relativePath);
        }
      }
      return filenames;
    }

    test('testDownload', () async {
      final hubApi = HubApi(downloadBase: downloadDestination.uri);
      double? totalCompleted;
      int? completedUnitCount;

      final downloadedTo = await hubApi.snapshot(
        repo,
        globs: ['*.json'],
        progressHandler: (count, progress) {
          totalCompleted = progress;
          completedUnitCount = count;
        },
      );

      final downloadedFilenames = getRelativeFiles(downloadDestination, repo.id);

      expect(totalCompleted, 1);
      expect(completedUnitCount, 6);
      expect(downloadedTo.path, '${downloadDestination.path}/models/${repo.id}/');

      expect(
        downloadedFilenames.toSet(),
        {
          'config.json',
          'tokenizer.json',
          'tokenizer_config.json',
          'llama-2-7b-chat.mlpackage/Manifest.json',
          'llama-2-7b-chat.mlpackage/Data/com.apple.CoreML/FeatureDescriptions.json',
          'llama-2-7b-chat.mlpackage/Data/com.apple.CoreML/Metadata.json',
        }.toSet(),
      );
    });

    test('testDownloadInBackground', () async {
      final hubApi = HubApi(downloadBase: downloadDestination.uri, useBackgroundSession: true);
      double? totalCompleted;
      int? completedUnitCount;

      final downloadedTo = await hubApi.snapshot(
        repo,
        globs: ['llama-2-7b-chat.mlpackage/Data/com.apple.CoreML/Metadata.json'],
        progressHandler: (count, progress) {
          totalCompleted = progress;
          completedUnitCount = count;
        },
      );

      expect(totalCompleted, 1);
      expect(completedUnitCount, 1);
      expect(downloadedTo.path, '${downloadDestination.path}/models/${repo.id}/');

      final downloadedFilenames = getRelativeFiles(downloadDestination, repo.id);
      expect(
        downloadedFilenames.toSet(),
        {'llama-2-7b-chat.mlpackage/Data/com.apple.CoreML/Metadata.json'}.toSet(),
      );
    });

    test('testCustomEndpointDownload', () async {
      final hubApi = HubApi(downloadBase: downloadDestination.uri, endpoint: "https://hf-mirror.com");
      double? totalCompleted;
      int? completedUnitCount;

      final downloadedTo = await hubApi.snapshot(
        repo,
        globs: ['*.json'],
        progressHandler: (count, progress) {
          totalCompleted = progress;
          completedUnitCount = count;
        },
      );

      expect(totalCompleted, 1);
      expect(completedUnitCount, 6);
      expect(downloadedTo.path, '${downloadDestination.path}/models/${repo.id}/');

      final downloadedFilenames = getRelativeFiles(downloadDestination, repo.id);
      expect(
        downloadedFilenames.toSet(),
        {
          'config.json',
          'tokenizer.json',
          'tokenizer_config.json',
          'llama-2-7b-chat.mlpackage/Manifest.json',
          'llama-2-7b-chat.mlpackage/Data/com.apple.CoreML/FeatureDescriptions.json',
          'llama-2-7b-chat.mlpackage/Data/com.apple.CoreML/Metadata.json',
        }.toSet(),
      );
    });

    test('testDownloadFileMetadata', () async {
      final hubApi = HubApi(downloadBase: downloadDestination.uri);
      double? totalCompleted;
      int? completedUnitCount;

      final downloadedTo = await hubApi.snapshot(
        repo,
        globs: ['*.json'],
        progressHandler: (count, progress) {
          totalCompleted = progress;
          completedUnitCount = count;
        },
      );

      expect(totalCompleted, 1);
      expect(completedUnitCount, 6);
      expect(downloadedTo.path, '${downloadDestination.path}/models/${repo.id}/');

      final downloadedFilenames = getRelativeFiles(downloadDestination, repo.id);
      expect(
        downloadedFilenames.toSet(),
        {
          'config.json',
          'tokenizer.json',
          'tokenizer_config.json',
          'llama-2-7b-chat.mlpackage/Manifest.json',
          'llama-2-7b-chat.mlpackage/Data/com.apple.CoreML/FeatureDescriptions.json',
          'llama-2-7b-chat.mlpackage/Data/com.apple.CoreML/Metadata.json',
        }.toSet(),
      );

      final metadataDestination = Directory('${downloadedTo.path}.cache/huggingface/download');
      final downloadedMetadataFilenames = getRelativeFiles(metadataDestination, repo.id, skipHidden: false);
      expect(
        downloadedMetadataFilenames.toSet(),
        {
          '.cache/huggingface/download/config.json.metadata',
          '.cache/huggingface/download/tokenizer.json.metadata',
          '.cache/huggingface/download/tokenizer_config.json.metadata',
          '.cache/huggingface/download/llama-2-7b-chat.mlpackage/Manifest.json.metadata',
          '.cache/huggingface/download/llama-2-7b-chat.mlpackage/Data/com.apple.CoreML/FeatureDescriptions.json.metadata',
          '.cache/huggingface/download/llama-2-7b-chat.mlpackage/Data/com.apple.CoreML/Metadata.json.metadata',
        }.toSet(),
      );
    });

    test('testDownloadFileMetadataExists', () async {
      final hubApi = HubApi(downloadBase: downloadDestination.uri);
      double? totalCompleted;
      int? completedUnitCount;

      final downloadedTo = await hubApi.snapshot(
        repo,
        globs: ['*.json'],
        progressHandler: (count, progress) {
          totalCompleted = progress;
          completedUnitCount = count;
        },
      );

      expect(totalCompleted, 1);
      expect(completedUnitCount, 6);
      expect(downloadedTo.path, '${downloadDestination.path}/models/${repo.id}/');

      final downloadedFilenames = getRelativeFiles(downloadDestination, repo.id);
      expect(
        downloadedFilenames.toSet(),
        {
          'config.json',
          'tokenizer.json',
          'tokenizer_config.json',
          'llama-2-7b-chat.mlpackage/Manifest.json',
          'llama-2-7b-chat.mlpackage/Data/com.apple.CoreML/FeatureDescriptions.json',
          'llama-2-7b-chat.mlpackage/Data/com.apple.CoreML/Metadata.json',
        }.toSet(),
      );

      final metadataDestination = Directory('${downloadedTo.path}.cache/huggingface/download');

      final configPath = File('${downloadedTo.path}config.json');
      var attributes = configPath.statSync();
      final originalTimestamp = attributes.modified;

      final downloadedMetadataFilenames = getRelativeFiles(metadataDestination, repo.id, skipHidden: false);
      expect(
        downloadedMetadataFilenames.toSet(),
        {
          '.cache/huggingface/download/config.json.metadata',
          '.cache/huggingface/download/tokenizer.json.metadata',
          '.cache/huggingface/download/tokenizer_config.json.metadata',
          '.cache/huggingface/download/llama-2-7b-chat.mlpackage/Manifest.json.metadata',
          '.cache/huggingface/download/llama-2-7b-chat.mlpackage/Data/com.apple.CoreML/FeatureDescriptions.json.metadata',
          '.cache/huggingface/download/llama-2-7b-chat.mlpackage/Data/com.apple.CoreML/Metadata.json.metadata',
        }.toSet(),
      );

      await hubApi.snapshot(
        repo,
        globs: ['*.json'],
        progressHandler: (count, progress) {
          totalCompleted = progress;
          completedUnitCount = count;
        },
      );

      attributes = configPath.statSync();
      final secondDownloadTimestamp = attributes.modified;

      // File will not be downloaded again thus last modified date will remain unchanged
      expect(originalTimestamp, secondDownloadTimestamp);
    });

    test('testDownloadFileMetadataSame', () async {
      final hubApi = HubApi(downloadBase: downloadDestination.uri);
      double? totalCompleted;
      int? completedUnitCount;

      final downloadedTo = await hubApi.snapshot(
        repo,
        globs: ['tokenizer.json'],
        progressHandler: (count, progress) {
          totalCompleted = progress;
          completedUnitCount = count;
        },
      );

      expect(totalCompleted, 1);
      expect(completedUnitCount, 1);
      expect(downloadedTo.path, '${downloadDestination.path}/models/${repo.id}/');

      final downloadedFilenames = getRelativeFiles(downloadDestination, repo.id);
      expect(downloadedFilenames.toSet(), {'tokenizer.json'}.toSet());

      final metadataDestination = Directory('${downloadedTo.path}.cache/huggingface/download');
      final metadataPath = File('${metadataDestination.path}/tokenizer.json.metadata');

      final downloadedMetadataFilenames = getRelativeFiles(metadataDestination, repo.id, skipHidden: false);
      expect(downloadedMetadataFilenames.toSet(), {'.cache/huggingface/download/tokenizer.json.metadata'}.toSet());

      final originalMetadata = metadataPath.readAsStringSync();

      await hubApi.snapshot(
        repo,
        globs: ['tokenizer.json'],
        progressHandler: (count, progress) {
          totalCompleted = progress;
          completedUnitCount = count;
        },
      );

      final secondDownloadMetadata = metadataPath.readAsStringSync();

      // File hasn't changed so commit hash and etag will be identical
      final originalArr = originalMetadata.split('\n');
      final secondDownloadArr = secondDownloadMetadata.split('\n');

      expect(originalArr[0], secondDownloadArr[0]);
      expect(originalArr[1], secondDownloadArr[1]);
    });

    test('testDownloadFileMetadataCorrupted', () async {
      final hubApi = HubApi(downloadBase: downloadDestination.uri);
      double? totalCompleted;
      int? completedUnitCount;

      final downloadedTo = await hubApi.snapshot(
        repo,
        globs: ['*.json'],
        progressHandler: (count, progress) {
          totalCompleted = progress;
          completedUnitCount = count;
        },
      );

      expect(totalCompleted, 1);
      expect(completedUnitCount, 6);
      expect(downloadedTo.path, '${downloadDestination.path}/models/${repo.id}/');

      final downloadedFilenames = getRelativeFiles(downloadDestination, repo.id);
      expect(
        downloadedFilenames.toSet(),
        {
          'config.json',
          'tokenizer.json',
          'tokenizer_config.json',
          'llama-2-7b-chat.mlpackage/Manifest.json',
          'llama-2-7b-chat.mlpackage/Data/com.apple.CoreML/FeatureDescriptions.json',
          'llama-2-7b-chat.mlpackage/Data/com.apple.CoreML/Metadata.json',
        }.toSet(),
      );

      final metadataDestination = Directory('${downloadedTo.path}.cache/huggingface/download');

      final configPath = File('${downloadedTo.path}config.json');
      var attributes = configPath.statSync();
      final originalTimestamp = attributes.modified;

      final downloadedMetadataFilenames = getRelativeFiles(metadataDestination, repo.id, skipHidden: false);
      expect(
        downloadedMetadataFilenames.toSet(),
        {
          '.cache/huggingface/download/config.json.metadata',
          '.cache/huggingface/download/tokenizer.json.metadata',
          '.cache/huggingface/download/tokenizer_config.json.metadata',
          '.cache/huggingface/download/llama-2-7b-chat.mlpackage/Manifest.json.metadata',
          '.cache/huggingface/download/llama-2-7b-chat.mlpackage/Data/com.apple.CoreML/FeatureDescriptions.json.metadata',
          '.cache/huggingface/download/llama-2-7b-chat.mlpackage/Data/com.apple.CoreML/Metadata.json.metadata',
        }.toSet(),
      );

      // Corrupt config.json.metadata
      await File('${metadataDestination.path}/config.json.metadata').writeAsString('a');

      await hubApi.snapshot(
        repo,
        globs: ['*.json'],
        progressHandler: (count, progress) {
          totalCompleted = progress;
          completedUnitCount = count;
        },
      );

      attributes = configPath.statSync();
      final secondDownloadTimestamp = attributes.modified;

      // File will be downloaded again thus last modified date will change
      expect(originalTimestamp != secondDownloadTimestamp, isTrue);

      // Corrupt config.metadata again
      await File('${metadataDestination.path}/config.json.metadata').writeAsString('a\nb\nc\n');

      await hubApi.snapshot(
        repo,
        globs: ['*.json'],
        progressHandler: (count, progress) {
          totalCompleted = progress;
          completedUnitCount = count;
        },
      );

      attributes = configPath.statSync();
      final thirdDownloadTimestamp = attributes.modified;

      // File will be downloaded again thus last modified date will change
      expect(originalTimestamp != thirdDownloadTimestamp, isTrue);
    });

    test('testDownloadLargeFileMetadataCorrupted', () async {
      final hubApi = HubApi(downloadBase: downloadDestination.uri);
      double? totalCompleted;
      int? completedUnitCount;

      final downloadedTo = await hubApi.snapshot(
        repo,
        globs: ['*.mlmodel'],
        progressHandler: (count, progress) {
          totalCompleted = progress;
          completedUnitCount = count;
        },
      );

      expect(totalCompleted, 1);
      expect(completedUnitCount, 1);
      expect(downloadedTo.path, '${downloadDestination.path}/models/${repo.id}/');

      final downloadedFilenames = getRelativeFiles(downloadDestination, repo.id);
      expect(
        downloadedFilenames.toSet(),
        {'llama-2-7b-chat.mlpackage/Data/com.apple.CoreML/model.mlmodel'}.toSet(),
      );

      final metadataDestination = Directory('${downloadedTo.path}.cache/huggingface/download');

      final modelPath = File('${downloadedTo.path}llama-2-7b-chat.mlpackage/Data/com.apple.CoreML/model.mlmodel');
      var attributes = modelPath.statSync();
      final originalTimestamp = attributes.modified;

      final downloadedMetadataFilenames = getRelativeFiles(metadataDestination, repo.id, skipHidden: false);
      expect(
        downloadedMetadataFilenames.toSet(),
        {'.cache/huggingface/download/llama-2-7b-chat.mlpackage/Data/com.apple.CoreML/model.mlmodel.metadata'}.toSet(),
      );

      // Corrupt model.metadata etag
      final corruptedMetadataString = 'a\nfc329090bfbb2570382c9af997cffd5f4b78b39b8aeca62076db69534e020108\n0\n';
      final metadataFile = File('${metadataDestination.path}/llama-2-7b-chat.mlpackage/Data/com.apple.CoreML/model.mlmodel.metadata');
      await metadataFile.writeAsString(corruptedMetadataString);

      await hubApi.snapshot(
        repo,
        globs: ['*.mlmodel'],
        progressHandler: (count, progress) {
          totalCompleted = progress;
          completedUnitCount = count;
        },
      );

      attributes = modelPath.statSync();
      final thirdDownloadTimestamp = attributes.modified;

      // File will not be downloaded again because this is an LFS file.
      // While downloading LFS files, we first check if local file ETag is the same as remote ETag.
      // If that's the case we just update the metadata and keep the local file.
      expect(originalTimestamp, thirdDownloadTimestamp);

      final metadataString = await metadataFile.readAsString();

      // Updated metadata file needs to have the correct commit hash, etag and timestamp.
      // This is being updated because the local etag (SHA256 checksum) matches the remote etag
      expect(metadataString, isNot(corruptedMetadataString));
    });

    test('testDownloadLargeFile', () async {
      final hubApi = HubApi(downloadBase: downloadDestination.uri);
      double? totalCompleted;
      int? completedUnitCount;

      final downloadedTo = await hubApi.snapshot(
        repo,
        globs: ['*.mlmodel'],
        progressHandler: (count, progress) {
          totalCompleted = progress;
          completedUnitCount = count;
        },
      );

      expect(totalCompleted, 1);
      expect(completedUnitCount, 1);
      expect(downloadedTo.path, '${downloadDestination.path}/models/${repo.id}/');

      final downloadedFilenames = getRelativeFiles(downloadDestination, repo.id);
      expect(
        downloadedFilenames.toSet(),
        {'llama-2-7b-chat.mlpackage/Data/com.apple.CoreML/model.mlmodel'}.toSet(),
      );

      final metadataDestination = Directory('${downloadedTo.path}.cache/huggingface/download');
      final downloadedMetadataFilenames = getRelativeFiles(metadataDestination, repo.id, skipHidden: false);
      expect(
        downloadedMetadataFilenames.toSet(),
        {'.cache/huggingface/download/llama-2-7b-chat.mlpackage/Data/com.apple.CoreML/model.mlmodel.metadata'}.toSet(),
      );

      final metadataFile = File('${metadataDestination.path}/llama-2-7b-chat.mlpackage/Data/com.apple.CoreML/model.mlmodel.metadata');
      final metadataString = await metadataFile.readAsString();

      final expected = 'eaf97358a37d03fd48e5a87d15aff2e8423c1afb\nfc329090bfbb2570382c9af997cffd5f4b78b39b8aeca62076db69534e020107';
      expect(metadataString.contains(expected), isTrue);
    });

    test('testDownloadSmolLargeFile', () async {
      final hubApi = HubApi(downloadBase: downloadDestination.uri);
      double? totalCompleted;
      int? completedUnitCount;

      final downloadedTo = await hubApi.snapshot(
        lfsRepo,
        globs: ['x.bin'],
        progressHandler: (count, progress) {
          totalCompleted = progress;
          completedUnitCount = count;
        },
      );

      expect(totalCompleted, 1);
      expect(completedUnitCount, 1);
      expect(downloadedTo.path, '${downloadDestination.path}/models/${lfsRepo.id}/');

      final downloadedFilenames = getRelativeFiles(downloadDestination, lfsRepo.id);
      expect(downloadedFilenames.toSet(), {'x.bin'}.toSet());

      final metadataDestination = Directory('${downloadedTo.path}.cache/huggingface/download');
      final downloadedMetadataFilenames = getRelativeFiles(metadataDestination, lfsRepo.id, skipHidden: false);
      expect(downloadedMetadataFilenames.toSet(), {'.cache/huggingface/download/x.bin.metadata'}.toSet());

      final metadataFile = File('${metadataDestination.path}/x.bin.metadata');
      final metadataString = await metadataFile.readAsString();

      final expected = '77b984598d90af6143d73d5a2d6214b23eba7e27\n98ea6e4f216f2fb4b69fff9b3a44842c38686ca685f3f55dc48c5d3fb1107be4';
      expect(metadataString.contains(expected), isTrue);
    });

    test('testRegexValidation', () async {
      final hubApi = HubApi(downloadBase: downloadDestination.uri);
      double? totalCompleted;
      int? completedUnitCount;

      final downloadedTo = await hubApi.snapshot(
        lfsRepo,
        globs: ['x.bin'],
        progressHandler: (count, progress) {
          totalCompleted = progress;
          completedUnitCount = count;
        },
      );

      expect(totalCompleted, 1);
      expect(completedUnitCount, 1);
      expect(downloadedTo.path, '${downloadDestination.path}/models/${lfsRepo.id}/');

      final downloadedFilenames = getRelativeFiles(downloadDestination, lfsRepo.id);
      expect(downloadedFilenames.toSet(), {'x.bin'}.toSet());

      final metadataDestination = Directory('${downloadedTo.path}.cache/huggingface/download');
      final downloadedMetadataFilenames = getRelativeFiles(metadataDestination, lfsRepo.id, skipHidden: false);
      expect(downloadedMetadataFilenames.toSet(), {'.cache/huggingface/download/x.bin.metadata'}.toSet());

      final metadataFile = File('${metadataDestination.path}/x.bin.metadata');
      final metadataString = await metadataFile.readAsString();
      final metadataArr = metadataString.split('\n');

      final commitHash = metadataArr[0];
      final etag = metadataArr[1];

      // Not needed for the downloads, just to test validation function
      final downloader = HubFileDownloader(
        repo: lfsRepo,
        repoDestination: downloadedTo,
        relativeFilename: 'x.bin',
        backgroundSession: false,
      );

      expect(downloader.isValidHash(commitHash, downloader.commitHashPattern), isTrue);
      expect(downloader.isValidHash(etag, downloader.sha256Pattern), isTrue);

      expect(downloader.isValidHash('$commitHash a', downloader.commitHashPattern), isFalse);
      expect(downloader.isValidHash('$etag a', downloader.sha256Pattern), isFalse);
    });

    test('testLFSFileNoMetadata', () async {
      final hubApi = HubApi(downloadBase: downloadDestination.uri);
      double? totalCompleted;
      int? completedUnitCount;

      final downloadedTo = await hubApi.snapshot(
        lfsRepo,
        globs: ['x.bin'],
        progressHandler: (count, progress) {
          totalCompleted = progress;
          completedUnitCount = count;
        },
      );

      expect(totalCompleted, 1);
      expect(completedUnitCount, 1);
      expect(downloadedTo.path, '${downloadDestination.path}/models/${lfsRepo.id}/');

      final downloadedFilenames = getRelativeFiles(downloadDestination, lfsRepo.id);
      expect(downloadedFilenames.toSet(), {'x.bin'}.toSet());

      final metadataDestination = Directory('${downloadedTo.path}.cache/huggingface/download');

      final filePath = File('${downloadedTo.path}x.bin');
      var attributes = filePath.statSync();
      final originalTimestamp = attributes.modified;

      final downloadedMetadataFilenames = getRelativeFiles(metadataDestination, lfsRepo.id, skipHidden: false);
      expect(downloadedMetadataFilenames.toSet(), {'.cache/huggingface/download/x.bin.metadata'}.toSet());

      final metadataFile = File('${metadataDestination.path}/x.bin.metadata');
      await metadataFile.delete();

      await hubApi.snapshot(
        lfsRepo,
        globs: ['x.bin'],
        progressHandler: (count, progress) {
          totalCompleted = progress;
          completedUnitCount = count;
        },
      );

      attributes = filePath.statSync();
      final secondDownloadTimestamp = attributes.modified;

      // File will not be downloaded again thus last modified date will remain unchanged
      expect(originalTimestamp, secondDownloadTimestamp);
      expect(metadataFile.existsSync(), isTrue);

      final metadataString = await metadataFile.readAsString();
      final expected = '77b984598d90af6143d73d5a2d6214b23eba7e27\n98ea6e4f216f2fb4b69fff9b3a44842c38686ca685f3f55dc48c5d3fb1107be4';

      expect(metadataString.contains(expected), isTrue);
    });

    test('testLFSFileCorruptedMetadata', () async {
      final hubApi = HubApi(downloadBase: downloadDestination.uri);
      double? totalCompleted;
      int? completedUnitCount;

      final downloadedTo = await hubApi.snapshot(
        lfsRepo,
        globs: ['x.bin'],
        progressHandler: (count, progress) {
          totalCompleted = progress;
          completedUnitCount = count;
        },
      );

      expect(totalCompleted, 1);
      expect(completedUnitCount, 1);
      expect(downloadedTo.path, '${downloadDestination.path}/models/${lfsRepo.id}/');

      final downloadedFilenames = getRelativeFiles(downloadDestination, lfsRepo.id);
      expect(downloadedFilenames.toSet(), {'x.bin'}.toSet());

      final metadataDestination = Directory('${downloadedTo.path}.cache/huggingface/download');

      final filePath = File('${downloadedTo.path}x.bin');
      var attributes = filePath.statSync();
      final originalTimestamp = attributes.modified;

      final downloadedMetadataFilenames = getRelativeFiles(metadataDestination, lfsRepo.id, skipHidden: false);
      expect(downloadedMetadataFilenames.toSet(), {'.cache/huggingface/download/x.bin.metadata'}.toSet());

      final metadataFile = File('${metadataDestination.path}/x.bin.metadata');
      await metadataFile.writeAsString('a');

      await hubApi.snapshot(
        lfsRepo,
        globs: ['x.bin'],
        progressHandler: (count, progress) {
          totalCompleted = progress;
          completedUnitCount = count;
        },
      );

      attributes = filePath.statSync();
      final secondDownloadTimestamp = attributes.modified;

      // File will not be downloaded again thus last modified date will remain unchanged
      expect(originalTimestamp, secondDownloadTimestamp);
      expect(metadataFile.existsSync(), isTrue);

      final metadataString = await metadataFile.readAsString();
      final expected = '77b984598d90af6143d73d5a2d6214b23eba7e27\n98ea6e4f216f2fb4b69fff9b3a44842c38686ca685f3f55dc48c5d3fb1107be4';

      expect(metadataString.contains(expected), isTrue);
    });

    test('testNonLFSFileRedownload', () async {
      final hubApi = HubApi(downloadBase: downloadDestination.uri);
      double? totalCompleted;
      int? completedUnitCount;

      final downloadedTo = await hubApi.snapshot(
        repo,
        globs: ['config.json'],
        progressHandler: (count, progress) {
          totalCompleted = progress;
          completedUnitCount = count;
        },
      );

      expect(totalCompleted, 1);
      expect(completedUnitCount, 1);
      expect(downloadedTo.path, '${downloadDestination.path}/models/${repo.id}/');

      final downloadedFilenames = getRelativeFiles(downloadDestination, repo.id);
      expect(downloadedFilenames.toSet(), {'config.json'}.toSet());

      final metadataDestination = Directory('${downloadedTo.path}.cache/huggingface/download');

      final filePath = File('${downloadedTo.path}config.json');
      var attributes = filePath.statSync();
      final originalTimestamp = attributes.modified;

      final downloadedMetadataFilenames = getRelativeFiles(metadataDestination, repo.id, skipHidden: false);
      expect(downloadedMetadataFilenames.toSet(), {'.cache/huggingface/download/config.json.metadata'}.toSet());

      final metadataFile = File('${metadataDestination.path}/config.json.metadata');
      await metadataFile.delete();

      await hubApi.snapshot(
        repo,
        globs: ['config.json'],
        progressHandler: (count, progress) {
          totalCompleted = progress;
          completedUnitCount = count;
        },
      );

      attributes = filePath.statSync();
      final secondDownloadTimestamp = attributes.modified;

      // File will be downloaded again thus last modified date will change
      expect(originalTimestamp != secondDownloadTimestamp, isTrue);
      expect(metadataFile.existsSync(), isTrue);

      final metadataString = await metadataFile.readAsString();
      final expected = 'eaf97358a37d03fd48e5a87d15aff2e8423c1afb\nd6ceb92ce9e3c83ab146dc8e92a93517ac1cc66f';

      expect(metadataString.contains(expected), isTrue);
    });
  });
}
