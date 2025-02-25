class TokenizerError extends Error {
  final String _type;
  final Object? message;
  TokenizerError(this._type, [this.message]);

  TokenizerError.tooLong(this.message) : _type = 'TokenizerError.tooLong';
  TokenizerError.missingConfig()
      : _type = 'TokenizerError.missingConfig',
        message = null;
  TokenizerError.missingTokenizerClassInConfig()
      : _type = 'TokenizerError.missingTokenizerClassInConfig',
        message = null;
  TokenizerError.tokenizerConfigNotFound()
      : _type = 'TokenizerError.tokenizerConfigNotFound',
        message = null;
  TokenizerError.unsupportedTokenizer(this.message) : _type = 'TokenizerError.unsupportedTokenizer';
  TokenizerError.missingVocab()
      : _type = 'TokenizerError.missingVocab',
        message = null;
  TokenizerError.malformedVocab()
      : _type = 'TokenizerError.malformedVocab',
        message = null;
  TokenizerError.chatTemplate(this.message) : _type = 'TokenizerError.chatTemplate';
  TokenizerError.mismatchedConfig(this.message) : _type = 'TokenizerError.mismatchedConfig';

  @override
  String toString() {
    if (message != null) {
      return "$_type: ${Error.safeToString(message)}";
    }
    return _type;
  }
}
