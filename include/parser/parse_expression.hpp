#ifndef KUIPER_INFER_INCLUDE_PARSER_PARSE_EXPRESSION_HPP_
#define KUIPER_INFER_INCLUDE_PARSER_PARSE_EXPRESSION_HPP_
#include <memory>
#include <string>
#include <utility>
#include <vector>

namespace kuiper_infer {

enum class TokenType {
    TokenUnknown = -9,
    TokenInputNumber = -8,
    TokenComma = -7,
    TokenAdd = -6,
    TokenMul = -5,
    TokenLeftBracket = -4,
    TokenRightBracket = -3,
};

/**
 * @brief Token parsed from expression
 *
 * Represents a token extracted from the input expression string.
 */
struct Token {
    /// Type of this token
    TokenType token_type = TokenType::TokenUnknown;

    /// Start position in the input expression
    int32_t start_pos = 0;

    /// End position in the input expression
    int32_t end_pos = 0;

    /**
     * @brief Construct a new Token, 左闭右开[)
     *
     * @param token_type Token type
     * @param start_pos Start position
     * @param end_pos End position
     */
    Token(TokenType token_type, int32_t start_pos, int32_t end_pos)
        : token_type(token_type), start_pos(start_pos), end_pos(end_pos) {
    }
};

/**
 * @brief Node in expression syntax tree
 */
struct TokenNode {
    /***
     * @brief Index of input variable
     * @details 当num_index=x时，表示计算数@(x-1)，
     * 比如当num_index=1时，表示计算数@0，当num_index=2时，表示计算数@1
     * 当num_index为负数时，表示当前节点是一个计算节点，比如add，mul
     * 
    */
    int32_t num_index = -1;

    /// Left child node
    std::shared_ptr<TokenNode> left = nullptr;

    /// Right child node
    std::shared_ptr<TokenNode> right = nullptr;

    /**
     * @brief Construct a new Token Node
     *
     * @param num_index Index of input variable
     * @param left Left child node
     * @param right Right child node
     */
    TokenNode(int32_t num_index, std::shared_ptr<TokenNode> left, std::shared_ptr<TokenNode> right);

    TokenNode() = default;
};

/**
 * @brief Parser for math expressions
 *
 * Parses a math expression string into a syntax tree.
 * Performs lexical and syntax analysis.
 */
class ExpressionParser {
public:
    /**
     * @brief Construct a new Expression Parser
     *
     * @param statement The expression string
     */
    explicit ExpressionParser(std::string statement) : statement_(std::move(statement)) {}

    /**
     * @brief Performs lexical analysis 词法分析
     *
     * Breaks the expression into tokens.
     *
     * @param retokenize Whether to re-tokenize
     */
    void Tokenizer(bool retokenize = false);

    /**
     * @brief Performs syntax analysis 语法分析
     *
     * 首先解析词法分析得到的Token数组tokens_，转换成一棵语法树
     * 然后将语法树转换成逆波兰表达式，并返回
     *
     * @return Vector of root nodes
     */
    std::vector<std::shared_ptr<TokenNode>> Generate();

    /**
     * @brief Gets the tokens
     *
     * @return The tokens
     */
    const std::vector<Token>& tokens() const;

    /**
     * @brief Gets the token strings
     *
     * @return The token strings
     */
    const std::vector<std::string>& token_str_array() const;

private:
    /**
     * @brief 从index位置开始解析词法分析得到的tokens_，转换成一棵语法树，并返回
     * @param index tokens_的下标
    */
    std::shared_ptr<TokenNode> Generate_(int32_t& index);

private:
    std::vector<Token> tokens_; /// 存放解析得到的Token
    std::vector<std::string> token_strs_; /// 表达式中tokens_对应的substr
    std::string statement_; /// input expression 
};
}  // namespace kuiper_infer

#endif  // KUIPER_INFER_INCLUDE_PARSER_PARSE_EXPRESSION_HPP_
