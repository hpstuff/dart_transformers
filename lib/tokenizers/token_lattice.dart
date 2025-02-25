class TokenLattice {
  final String sentence;
  late final int len;
  final int bosTokenId;
  final int eosTokenId;
  final List<TokenLatticeNode> nodes = [];
  late final List<List<TokenLatticeNode>> beginNodes;
  late final List<List<TokenLatticeNode>> endNodes;

  TokenLattice({required this.sentence, required this.bosTokenId, required this.eosTokenId}) {
    len = sentence.length;
    beginNodes = List.generate(len + 1, (_) => []);
    endNodes = List.generate(len + 1, (_) => []);

    var bos = TokenLatticeNode(bosTokenId, 0, 0, 0, 0.0);
    var eos = TokenLatticeNode(eosTokenId, 1, len, 0, 0.0);
    nodes.add(bos.clone());
    nodes.add(eos.clone());
    beginNodes[len].add(eos);
    endNodes[0].add(bos);
  }

  void insert({required int pos, required int length, required double score, required int tokenId}) {
    var nodeId = nodes.length;
    var node = TokenLatticeNode(tokenId, nodeId, pos, length, score);
    beginNodes[pos].add(node);
    endNodes[pos + length].add(node);
    nodes.add(node);
  }

  List<TokenLatticeNode> viterbi() {
    var len = this.len;
    var pos = 0;
    while (pos <= len) {
      if (beginNodes[pos].isEmpty) {
        return [];
      }
      for (var rnode in beginNodes[pos]) {
        rnode.prev = null;
        var bestScore = 0.0;
        TokenLatticeNode? bestNode;
        for (var lnode in endNodes[pos]) {
          var score = lnode.backtraceScore + rnode.score;
          if (bestNode == null || score > bestScore) {
            bestNode = lnode.clone();
            bestScore = score;
          }
        }

        if (bestNode != null) {
          rnode.prev = bestNode;
          rnode.backtraceScore = bestScore;
        } else {
          return [];
        }
      }
      pos++;
    }

    var results = <TokenLatticeNode>[];
    var root = beginNodes[len][0];
    var prev = root.prev;
    if (prev == null) {
      return [];
    }

    var node = prev.clone();
    while (node.prev != null) {
      results.add(node.clone());
      var n = node.clone();
      node = n.prev!.clone();
    }

    results = results.reversed.toList();
    return results;
  }

  String piece(TokenLatticeNode node) {
    return sentence.substring(node.pos, node.pos + node.length);
  }

  List<String> get tokens {
    var nodes = viterbi();
    return nodes.map((x) => piece(x)).toList();
  }

  List<int> get tokenIds {
    var nodes = viterbi();
    return nodes.map((x) => x.tokenId).toList();
  }
}

class TokenLatticeNode {
  final int tokenId;
  final int nodeId;
  final int pos;
  final int length;
  final double score;
  TokenLatticeNode? prev;
  double backtraceScore = 0.0;

  TokenLatticeNode(this.tokenId, this.nodeId, this.pos, this.length, this.score) {
    prev = null;
  }

  TokenLatticeNode clone() {
    var n = TokenLatticeNode(tokenId, nodeId, pos, length, score);
    n.prev = prev;
    n.backtraceScore = backtraceScore;
    return n;
  }

  TokenLatticeNode copyWith({int? tokenId, int? nodeId, int? pos, int? length, double? score, TokenLatticeNode? prev, double? backtraceScore}) {
    return TokenLatticeNode(
      tokenId ?? this.tokenId,
      nodeId ?? this.nodeId,
      pos ?? this.pos,
      length ?? this.length,
      score ?? this.score,
    );
  }
}
