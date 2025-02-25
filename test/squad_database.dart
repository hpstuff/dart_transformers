import 'dart:convert';
import 'dart:io';

class SquadExample {
  final String qaId;
  final String context;
  final String question;
  final String answerText;
  final int startPos;
  final int endPos;

  SquadExample({
    required this.qaId,
    required this.context,
    required this.question,
    required this.answerText,
    required this.startPos,
    required this.endPos,
  });

  @override
  String toString() {
    return """SquadExample{
      qaId: $qaId,
      context: ${context.substring(1, 10)},
      question: ${question.substring(1, 10)},
      answerText: ${answerText.substring(1, 10)},
      startPos: $startPos,
      endPos: $endPos
    }""";
  }
}

class SquadDataset {
  final List<SquadDatum> data;
  final String version;

  SquadDataset({required this.data, required this.version});

  factory SquadDataset.fromJson(Map<String, dynamic> json) {
    return SquadDataset(
      data: List<SquadDatum>.from(
          json['data'].map((x) => SquadDatum.fromJson(x))),
      version: json['version'],
    );
  }
}

class SquadDatum {
  final List<SquadParagraph> paragraphs;
  final String title;

  SquadDatum({required this.paragraphs, required this.title});

  factory SquadDatum.fromJson(Map<String, dynamic> json) {
    return SquadDatum(
      paragraphs: List<SquadParagraph>.from(
          json['paragraphs'].map((x) => SquadParagraph.fromJson(x))),
      title: json['title'],
    );
  }
}

class SquadParagraph {
  final String context;
  final List<SquadQA> qas;

  SquadParagraph({required this.context, required this.qas});

  factory SquadParagraph.fromJson(Map<String, dynamic> json) {
    return SquadParagraph(
      context: json['context'],
      qas: List<SquadQA>.from(json['qas'].map((x) => SquadQA.fromJson(x))),
    );
  }
}

class SquadQA {
  final List<SquadAnswer> answers;
  final String id;
  final String question;

  SquadQA({required this.answers, required this.id, required this.question});

  factory SquadQA.fromJson(Map<String, dynamic> json) {
    return SquadQA(
      answers: List<SquadAnswer>.from(
          json['answers'].map((x) => SquadAnswer.fromJson(x))),
      id: json['id'],
      question: json['question'],
    );
  }
}

class SquadAnswer {
  final int answerStart;
  final String text;

  SquadAnswer({required this.answerStart, required this.text});

  factory SquadAnswer.fromJson(Map<String, dynamic> json) {
    return SquadAnswer(
      answerStart: json['answer_start'],
      text: json['text'],
    );
  }
}

class Squad {
  static Future<List<SquadExample>> getExamples() async {
    final file = File('test_assets/dev-v1.1.json');
    final json = jsonDecode(await file.readAsString());
    final squadDataset = SquadDataset.fromJson(json);

    List<SquadExample> examples = [];
    for (var datum in squadDataset.data) {
      for (var paragraph in datum.paragraphs) {
        for (var qa in paragraph.qas) {
          var example = SquadExample(
            qaId: qa.id,
            context: paragraph.context,
            question: qa.question,
            answerText: qa.answers[0].text,
            startPos: qa.answers[0].answerStart,
            endPos: -1, // todo: remove -1
          );
          examples.add(example);
        }
      }
    }
    return examples;
  }
}
