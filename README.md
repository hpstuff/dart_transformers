# Dart Transformers

This library is a work in progress and is not yet ready for production use.

`dart_transformers` is a collection of utilities to help adopt language models in Dart apps.

It's inspired by [`swift-transformers`](https://github.com/huggingface/swift-transformers) and aims to provide similar functionality for Dart developers.

## Modules

- `Tokenizers`: Utilities to convert text to tokens and back, with support for Chat Templates and Tools. Follows the abstractions in [`tokenizers`](https://github.com/huggingface/tokenizers). Usage example:
```dart
import 'package:dart_transformers/transformers.dart';

void testTokenizer() async {
  final tokenizer = await AutoTokenizer.fromPretrained("deepseek-ai/DeepSeek-R1-Distill-Qwen-7B");
  final messages = [{"role": "user", "content": "Describe the Dart programming language."}];
  final encoded = tokenizer.applyChatTemplate(messages);
  final decoded = tokenizer.decode(encoded);
}
```

- `Hub`: Utilities for interacting with the Hugging Face Hub! Download models, tokenizers and other config files. Usage example:
```dart
import 'package:dart_transformers/transformers.dart';

void testHub() async {
  final repo = Repo(id: "mlx-community/Qwen2.5-0.5B-Instruct-2bit-mlx");
  final filesToDownload = ["config.json", "*.safetensors"];
  final modelDirectory = await Hub.snapshot(
    repo: repo,
    globs: filesToDownload,
    progressHandler: (count, progress) {
      print("Download progress: ${progress * 100}%");
    }
  );
  print("Files downloaded to: ${modelDirectory.path}");
}
```

## License

[Apache 2](LICENSE).