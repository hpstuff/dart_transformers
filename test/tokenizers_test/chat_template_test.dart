import 'dart:convert';
import 'dart:io' as io;
import 'package:dart_transformers/tokenizers/tokenizer.dart';
import 'package:dart_transformers/tokenizers/tokenizer_error.dart';
import 'package:flutter_test/flutter_test.dart';

void main() {
  TestWidgetsFlutterBinding.ensureInitialized();
  io.HttpOverrides.global = null;
  group('ChatTemplateTests', () {
    final messages = [
      {
        "role": "user",
        "content": "Describe the Swift programming language.",
      }
    ];
    test('testTemplateFromConfig', () async {
      final tokenizer = await AutoTokenizer.fromPretrained("microsoft/Phi-3-mini-128k-instruct");
      final encoded = tokenizer.applyChatTemplate(messages);
      final encodedTarget = [32010, 4002, 29581, 278, 14156, 8720, 4086, 29889, 32007, 32001];
      expect(encoded, encodedTarget);
      final decoded = tokenizer.decode(encoded);
      final decodedTarget = "<|user|>Describe the Swift programming language.<|end|><|assistant|>";
      expect(decoded, decodedTarget);
    });

    test('testDeepSeekQwenChatTemplate', () async {
      final tokenizer = await AutoTokenizer.fromPretrained("deepseek-ai/DeepSeek-R1-Distill-Qwen-7B");
      final encoded = tokenizer.applyChatTemplate(messages);
      final encodedTarget = [151646, 151644, 74785, 279, 23670, 15473, 4128, 13, 151645, 151648, 198];
      expect(encoded, encodedTarget);
      final decoded = tokenizer.decode(encoded);
      final decodedTarget = "<｜begin▁of▁sentence｜><｜User｜>Describe the Swift programming language.<｜Assistant｜><think>\n";
      expect(decoded, decodedTarget);
    });

    test('testDefaultTemplateFromArrayInConfig', () async {
      final tokenizer = await AutoTokenizer.fromPretrained("mlx-community/Mistral-7B-Instruct-v0.3-4bit");
      final encoded = tokenizer.applyChatTemplate(messages);
      final encodedTarget = [1, 29473, 3, 28752, 1040, 4672, 2563, 17060, 4610, 29491, 29473, 4];
      expect(encoded, encodedTarget);
      final decoded = tokenizer.decode(encoded);
      final decodedTarget = "<s> [INST] Describe the Swift programming language. [/INST]";
      expect(decoded, decodedTarget);
    });

    test('testTemplateFromArgumentWithEnum', () async {
      final tokenizer = await AutoTokenizer.fromPretrained("microsoft/Phi-3-mini-128k-instruct");
      final mistral7BDefaultTemplate =
          "{{bos_token}}{% for message in messages %}{% if (message['role'] == 'user') != (loop.index0 % 2 == 0) %}{{ raise_exception('Conversation roles must alternate user/assistant/user/assistant/...') }}{% endif %}{% if message['role'] == 'user' %}{{ ' [INST] ' + message['content'] + ' [/INST]' }}{% elif message['role'] == 'assistant' %}{{ ' ' + message['content'] + ' ' + eos_token}}{% else %}{{ raise_exception('Only user and assistant roles are supported!') }}{% endif %}{% endfor %}";
      final encoded = tokenizer.applyChatTemplate(messages, chatTemplate: ChatTemplateArgument.literal(mistral7BDefaultTemplate));
      final encodedTarget = [1, 518, 25580, 29962, 20355, 915, 278, 14156, 8720, 4086, 29889, 518, 29914, 25580, 29962];
      expect(encoded, encodedTarget);
      final decoded = tokenizer.decode(encoded);
      final decodedTarget = "<s> [INST] Describe the Swift programming language. [/INST]";
      expect(decoded, decodedTarget);
    });

    test('testNamedTemplateFromArgument', () async {
      final tokenizer = await AutoTokenizer.fromPretrained("mlx-community/Mistral-7B-Instruct-v0.3-4bit");
      final encoded = tokenizer.applyChatTemplate(messages, chatTemplate: ChatTemplateArgument.name("default"));
      final encodedTarget = [1, 29473, 3, 28752, 1040, 4672, 2563, 17060, 4610, 29491, 29473, 4];
      expect(encoded, encodedTarget);
      final decoded = tokenizer.decode(encoded);
      final decodedTarget = "<s> [INST] Describe the Swift programming language. [/INST]";
      expect(decoded, decodedTarget);
    });

    test("testQwen2_5WithTools", () async {
      final tokenizer = await AutoTokenizer.fromPretrained("mlx-community/Qwen2.5-7B-Instruct-4bit");

      final weatherQueryMessages = [
        {
          "role": "user",
          "content": "What is the weather in Paris today?",
        }
      ];

      final getCurrentWeatherToolSpec = {
        "type": "function",
        "function": {
          "name": "get_current_weather",
          "description": "Get the current weather in a given location",
          "parameters": {
            "type": "object",
            "properties": {
              "location": {"type": "string", "description": "The city and state, e.g. San Francisco, CA"},
              "unit": {
                "type": "string",
                "enum": ["celsius", "fahrenheit"]
              }
            },
            "required": ["location"]
          }
        }
      };

      final encoded = tokenizer.applyChatTemplate(weatherQueryMessages, tools: [getCurrentWeatherToolSpec]);
      final decoded = tokenizer.decode(encoded);

      void assertDictsAreEqual(Map<String, dynamic> actual, Map<String, dynamic> expected) {
        actual.forEach((key, value) {
          if (value is Map<String, dynamic> && expected[key] is Map<String, dynamic>) {
            assertDictsAreEqual(value, expected[key]);
          } else if (value is List<String>) {
            final expectedArrayValue = expected[key] as List<String>?;
            expect(expectedArrayValue, isNotNull);
            expect(Set.from(value), Set.from(expectedArrayValue!));
          } else {
            expect(value, expected[key]);
          }
        });
      }

      final startRange = decoded.indexOf("<tools>\n");
      final endRange = decoded.indexOf("\n</tools>", startRange + "<tools>\n".length);
      if (startRange != -1 && endRange != -1) {
        final toolsSection = decoded.substring(startRange + "<tools>\n".length, endRange);
        final toolsDict = jsonDecode(toolsSection) as Map<String, dynamic>?;
        if (toolsDict != null) {
          assertDictsAreEqual(toolsDict, getCurrentWeatherToolSpec);
        } else {
          fail("Failed to decode tools section");
        }
      } else {
        fail("Failed to find tools section");
      }

      final expectedPromptStart = """<|im_start|>system
You are Qwen, created by Alibaba Cloud. You are a helpful assistant.

# Tools

You may call one or more functions to assist with the user query.

You are provided with function signatures within <tools></tools> XML tags:
<tools>""";

      final expectedPromptEnd = r"""</tools>

For each function call, return a json object with function name and arguments within <tool_call></tool_call> XML tags:
<tool_call>
{\"name\": <function-name>, \"arguments\": <args-json-object>}
</tool_call><|im_end|>
<|im_start|>user
What is the weather in Paris today?<|im_end|>
<|im_start|>assistant
""";

      expect(decoded.startsWith(expectedPromptStart), isTrue, reason: "Prompt should start with expected system message");
      expect(decoded.endsWith(expectedPromptEnd), isTrue, reason: "Prompt should end with expected format");
    });

    test('testHasChatTemplate', () async {
      var tokenizer = await AutoTokenizer.fromPretrained("google-bert/bert-base-uncased");
      expect(tokenizer.hasChatTemplate, isFalse);

      tokenizer = await AutoTokenizer.fromPretrained("deepseek-ai/DeepSeek-R1-Distill-Qwen-7B");
      expect(tokenizer.hasChatTemplate, isTrue);
    });

    test('testApplyTemplateError', () async {
      var tokenizer = await AutoTokenizer.fromPretrained("google-bert/bert-base-uncased");
      expect(tokenizer.hasChatTemplate, isFalse);
      expect(() => tokenizer.applyChatTemplate([]), throwsA(isA<TokenizerError>()));
      try {
        tokenizer.applyChatTemplate([]);
        fail('Expected TokenizerError');
      } catch (e) {
        if (e is TokenizerError) {
          expect(e.message, "This tokenizer does not have a chat template, and no template was passed.");
        } else {
          fail('Unexpected error: $e');
        }
      }
    });
  });
}
