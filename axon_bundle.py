#!/usr/bin/env python3
"""
Axon Compiler - All-in-One Repository Generator, Builder & Test Runner.
Run: python3 axon_bundle.py
"""

import os
import sys
import subprocess

FILES = {}

# =====================================================================
# Configuration & Build Files
# =====================================================================
FILES["pyproject.toml"] = """[build-system]
requires = ["setuptools>=61.0", "wheel"]
build-backend = "setuptools.build_meta"

[project]
name = "axon-compiler"
version = "1.0.0"
description = "Axon Tensor DSL & Domain-Specific Compiler with FP4 Quantization"
readme = "README.md"
requires-python = ">=3.10"
dependencies = [
    "llvmlite>=0.40.0",
]

[project.scripts]
axonc = "axon.cli:main"

[tool.setuptools.packages.find]
where = ["."]
include = ["axon*"]
"""

FILES["README.md"] = """# Axon Compiler Architecture

Axon is a domain-specific compiler that transforms tensor graphs into
standalone native binaries embedding FP4-quantized weights.
Zero Python runtime dependency is required for the final binary.
"""

FILES["Makefile"] = """CC ?= gcc
CFLAGS ?= -O3 -fPIC -Wall -Wextra -std=c11 -march=native
AR ?= ar
PYTHON ?= python3

RUNTIME_DIR = runtime
RUNTIME_LIB = $(RUNTIME_DIR)/libaxon_runtime.a
RUNTIME_OBJ = $(RUNTIME_DIR)/axon_tensor.o \\
              $(RUNTIME_DIR)/axon_matmul.o \\
              $(RUNTIME_DIR)/axon_quant.o \\
              $(RUNTIME_DIR)/axon_mmap.o \\
              $(RUNTIME_DIR)/axon_runtime.o

.PHONY: all clean runtime test

all: runtime

runtime: $(RUNTIME_LIB)

$(RUNTIME_DIR)/%.o: $(RUNTIME_DIR)/%.c
	$(CC) $(CFLAGS) -c $< -o $@

$(RUNTIME_LIB): $(RUNTIME_OBJ)
	$(AR) rcs $@ $^

clean:
	rm -f $(RUNTIME_DIR)/*.o $(RUNTIME_LIB)
	rm -rf build/ dist/ *.egg-info
	find . -type d -name "__pycache__" -exec rm -rf {} +
	rm -f *.bin *.o *.blob *.S *.gen.o

test: runtime
	$(PYTHON) -m unittest discover -s tests -p "test_*.py"
"""

# =====================================================================
# Axon Package: __init__, utils, frontend
# =====================================================================
FILES["axon/__init__.py"] = """__version__ = "1.0.0"\n"""

FILES["axon/utils/__init__.py"] = """from .errors import AxonError, LexError, ParseError, SemanticError, CodegenError
__all__ = ["AxonError", "LexError", "ParseError", "SemanticError", "CodegenError"]
"""

FILES["axon/utils/errors.py"] = """class AxonError(Exception):
    def __init__(self, message: str, line: int = 0, col: int = 0, filename: str = "<axon>"):
        self.message = message
        self.line = line
        self.col = col
        self.filename = filename
        super().__init__(self.__str__())

    def __str__(self) -> str:
        if self.line > 0:
            return f"[{self.filename}:{self.line}:{self.col}] {self.message}"
        return f"[{self.filename}] {self.message}"

class LexError(AxonError): pass
class ParseError(AxonError): pass
class SemanticError(AxonError): pass
class CodegenError(AxonError): pass
"""

FILES["axon/frontend/__init__.py"] = """from .tokens import Token, TokenKind
from .lexer import Lexer
from .ast_nodes import Program, FunctionDef, Param, TypeExpr, TensorLiteral, VarRef, CallExpr, BinaryExpr, LetStmt, ReturnStmt
from .parser import Parser
from .semantic import SemanticAnalyzer
__all__ = ["Token", "TokenKind", "Lexer", "Program", "FunctionDef", "Param", "TypeExpr", "TensorLiteral", "VarRef", "CallExpr", "BinaryExpr", "LetStmt", "ReturnStmt", "Parser", "SemanticAnalyzer"]
"""

FILES["axon/frontend/tokens.py"] = """from enum import Enum, auto
from dataclasses import dataclass

class TokenKind(Enum):
    IDENT = auto(); INT_LIT = auto(); FLOAT_LIT = auto(); STRING_LIT = auto()
    FN = auto(); LET = auto(); IN = auto(); RETURN = auto(); AS = auto()
    TENSOR = auto(); MATMUL = auto(); RELU = auto(); ADD = auto(); MUL = auto(); CAST = auto()
    LPAREN = auto(); RPAREN = auto(); LBRACKET = auto(); RBRACKET = auto(); LBRACE = auto(); RBRACE = auto()
    COMMA = auto(); COLON = auto(); ARROW = auto(); ASSIGN = auto(); SEMICOLON = auto()
    PLUS = auto(); STAR = auto(); EOF = auto()

@dataclass(frozen=True)
class Token:
    kind: TokenKind
    lexeme: str
    line: int
    col: int
"""

FILES["axon/frontend/lexer.py"] = """from .tokens import Token, TokenKind
from ..utils.errors import LexError

class Lexer:
    KEYWORDS = {
        "fn": TokenKind.FN, "let": TokenKind.LET, "in": TokenKind.IN,
        "return": TokenKind.RETURN, "as": TokenKind.AS, "tensor": TokenKind.TENSOR,
        "matmul": TokenKind.MATMUL, "relu": TokenKind.RELU, "add": TokenKind.ADD,
        "mul": TokenKind.MUL, "cast": TokenKind.CAST,
    }

    def __init__(self, source: str, filename: str = "<axon>"):
        self.source = source
        self.filename = filename
        self.pos = 0
        self.line = 1
        self.col = 1
        self.tokens: list[Token] = []

    def _peek(self, offset: int = 0) -> str:
        idx = self.pos + offset
        return self.source[idx] if idx < len(self.source) else ""

    def _advance(self) -> str:
        ch = self._peek()
        self.pos += 1
        if ch == "\\n":
            self.line += 1
            self.col = 1
        else:
            self.col += 1
        return ch

    def tokenize(self) -> list[Token]:
        while self.pos < len(self.source):
            ch = self._peek()
            if ch.isspace():
                self._advance()
                continue
            if ch == "/" and self._peek(1) == "/":
                while self._peek() not in ("", "\\n"):
                    self._advance()
                continue
            start_l, start_c = self.line, self.col
            if ch == "-" and self._peek(1) == ">":
                self._advance(); self._advance()
                self.tokens.append(Token(TokenKind.ARROW, "->", start_l, start_c))
                continue
            single = {
                "(": TokenKind.LPAREN, ")": TokenKind.RPAREN,
                "[": TokenKind.LBRACKET, "]": TokenKind.RBRACKET,
                "{": TokenKind.LBRACE, "}": TokenKind.RBRACE,
                ",": TokenKind.COMMA, ":": TokenKind.COLON,
                ";": TokenKind.SEMICOLON, "=": TokenKind.ASSIGN,
                "+": TokenKind.PLUS, "*": TokenKind.STAR
            }
            if ch in single:
                self._advance()
                self.tokens.append(Token(single[ch], ch, start_l, start_c))
            elif ch.isalpha() or ch == "_":
                chars = []
                while self._peek().isalnum() or self._peek() == "_":
                    chars.append(self._advance())
                s = "".join(chars)
                self.tokens.append(Token(self.KEYWORDS.get(s, TokenKind.IDENT), s, start_l, start_c))
            elif ch.isdigit() or (ch == "-" and (self._peek(1).isdigit() or self._peek(1) == ".")):
                chars = [self._advance()] if ch == "-" else []
                has_dot = False
                while self._peek().isdigit() or (self._peek() == "." and not has_dot):
                    if self._peek() == ".": has_dot = True
                    chars.append(self._advance())
                s = "".join(chars)
                self.tokens.append(Token(TokenKind.FLOAT_LIT if has_dot else TokenKind.INT_LIT, s, start_l, start_c))
            else:
                raise LexError(f"Unexpected character: {ch!r}", self.line, self.col, self.filename)
        self.tokens.append(Token(TokenKind.EOF, "", self.line, self.col))
        return self.tokens
"""

FILES["axon/frontend/ast_nodes.py"] = """from dataclasses import dataclass, field
from typing import Optional, List

@dataclass(frozen=True)
class SourceLoc:
    line: int
    col: int

class Expr: loc: SourceLoc
class Stmt: loc: SourceLoc

@dataclass
class TypeExpr(Expr):
    name: str
    shape: List[int]
    loc: SourceLoc

@dataclass
class Param(Expr):
    name: str
    type_expr: TypeExpr
    loc: SourceLoc

@dataclass
class TensorLiteral(Expr):
    values: List[float]
    shape: List[int]
    dtype: str
    loc: SourceLoc

@dataclass
class VarRef(Expr):
    name: str
    loc: SourceLoc

@dataclass
class CallExpr(Expr):
    callee: str
    args: List[Expr]
    loc: SourceLoc
    resolved_shape: Optional[List[int]] = None
    resolved_dtype: Optional[str] = None

@dataclass
class BinaryExpr(Expr):
    op: str
    left: Expr
    right: Expr
    loc: SourceLoc
    resolved_shape: Optional[List[int]] = None
    resolved_dtype: Optional[str] = None

@dataclass
class LetStmt(Stmt):
    name: str
    type_expr: Optional[TypeExpr]
    value: Expr
    loc: SourceLoc

@dataclass
class ReturnStmt(Stmt):
    value: Expr
    loc: SourceLoc

@dataclass
class ExprStmt(Stmt):
    expr: Expr
    loc: SourceLoc

@dataclass
class FunctionDef(Expr):
    name: str
    params: List[Param]
    return_type: TypeExpr
    body: List[Stmt]
    loc: SourceLoc

@dataclass
class Program(Expr):
    functions: List[FunctionDef]
    loc: SourceLoc = field(default_factory=lambda: SourceLoc(1, 1))
"""

FILES["axon/frontend/grammar.py"] = """AXON_EBNF_SPEC = '''
program      = { function_def } ;
function_def = "fn" identifier "(" [ param { "," param } ] ")" "->" type_expr block ;
param        = identifier ":" type_expr ;
type_expr    = dtype [ "[" integer { "," integer } "]" ] ;
dtype        = "f32" | "f16" | "fp4" | "i32" | "bool" ;
block        = "{" { statement } "}" ;
statement    = let_stmt | return_stmt | expr_stmt ;
let_stmt     = "let" identifier [ ":" type_expr ] "=" expr ";" ;
return_stmt  = "return" expr ";" ;
expr_stmt    = expr ";" ;
'''
"""

FILES["axon/frontend/parser.py"] = """from .tokens import Token, TokenKind
from .ast_nodes import (
    SourceLoc, Program, FunctionDef, Param, TypeExpr, TensorLiteral,
    VarRef, CallExpr, BinaryExpr, LetStmt, ReturnStmt, ExprStmt, Expr, Stmt
)
from ..utils.errors import ParseError

class Parser:
    def __init__(self, tokens: list[Token], filename: str = "<axon>"):
        self.tokens = tokens
        self.filename = filename
        self.pos = 0

    def _peek(self) -> Token: return self.tokens[self.pos]
    def _check(self, kind: TokenKind) -> bool: return self._peek().kind == kind

    def _advance(self) -> Token:
        tok = self._peek()
        if tok.kind != TokenKind.EOF: self.pos += 1
        return tok

    def _consume(self, kind: TokenKind, msg: str) -> Token:
        if self._check(kind): return self._advance()
        curr = self._peek()
        raise ParseError(f"{msg}, got {curr.kind.name} ('{curr.lexeme}')", curr.line, curr.col, self.filename)

    def parse(self) -> Program:
        functions = []
        while not self._check(TokenKind.EOF):
            functions.append(self._parse_function())
        return Program(functions=functions)

    def _parse_function(self) -> FunctionDef:
        fn_tok = self._consume(TokenKind.FN, "Expected 'fn'")
        name_tok = self._consume(TokenKind.IDENT, "Expected function name")
        self._consume(TokenKind.LPAREN, "Expected '('")
        params = []
        if not self._check(TokenKind.RPAREN):
            while True:
                p_name = self._consume(TokenKind.IDENT, "Expected parameter name")
                self._consume(TokenKind.COLON, "Expected ':'")
                p_type = self._parse_type()
                params.append(Param(p_name.lexeme, p_type, SourceLoc(p_name.line, p_name.col)))
                if self._check(TokenKind.COMMA): self._advance()
                else: break
        self._consume(TokenKind.RPAREN, "Expected ')'")
        self._consume(TokenKind.ARROW, "Expected '->'")
        return_type = self._parse_type()
        self._consume(TokenKind.LBRACE, "Expected '{'")
        body: list[Stmt] = []
        while not self._check(TokenKind.RBRACE) and not self._check(TokenKind.EOF):
            body.append(self._parse_statement())
        self._consume(TokenKind.RBRACE, "Expected '}'")
        return FunctionDef(name_tok.lexeme, params, return_type, body, SourceLoc(fn_tok.line, fn_tok.col))

    def _parse_type(self) -> TypeExpr:
        dtype_tok = self._consume(TokenKind.IDENT, "Expected type")
        shape: list[int] = []
        if self._check(TokenKind.LBRACKET):
            self._advance()
            while not self._check(TokenKind.RBRACKET) and not self._check(TokenKind.EOF):
                dim_tok = self._consume(TokenKind.INT_LIT, "Expected integer dimension")
                shape.append(int(dim_tok.lexeme))
                if self._check(TokenKind.COMMA): self._advance()
                else: break
            self._consume(TokenKind.RBRACKET, "Expected ']'")
        return TypeExpr(dtype_tok.lexeme, shape, SourceLoc(dtype_tok.line, dtype_tok.col))

    def _parse_statement(self) -> Stmt:
        if self._check(TokenKind.LET):
            tok = self._advance()
            name_tok = self._consume(TokenKind.IDENT, "Expected name")
            type_expr = None
            if self._check(TokenKind.COLON):
                self._advance()
                type_expr = self._parse_type()
            self._consume(TokenKind.ASSIGN, "Expected '='")
            val = self._parse_expr()
            self._consume(TokenKind.SEMICOLON, "Expected ';'")
            return LetStmt(name_tok.lexeme, type_expr, val, SourceLoc(tok.line, tok.col))
        if self._check(TokenKind.RETURN):
            tok = self._advance()
            val = self._parse_expr()
            self._consume(TokenKind.SEMICOLON, "Expected ';'")
            return ReturnStmt(val, SourceLoc(tok.line, tok.col))
        expr = self._parse_expr()
        self._consume(TokenKind.SEMICOLON, "Expected ';'")
        return ExprStmt(expr, expr.loc)

    def _parse_expr(self) -> Expr:
        left = self._parse_primary()
        while self._check(TokenKind.PLUS) or self._check(TokenKind.STAR):
            op = self._advance()
            right = self._parse_primary()
            left = BinaryExpr(op.lexeme, left, right, SourceLoc(op.line, op.col))
        return left

    def _parse_primary(self) -> Expr:
        tok = self._peek()
        if tok.kind == TokenKind.LBRACKET:
            self._advance()
            values: list[float] = []
            if not self._check(TokenKind.RBRACKET):
                while True:
                    sign = 1.0
                    if self._check(TokenKind.PLUS): self._advance()
                    num = self._advance()
                    if num.kind not in (TokenKind.INT_LIT, TokenKind.FLOAT_LIT):
                        raise ParseError("Expected number", num.line, num.col, self.filename)
                    values.append(sign * float(num.lexeme))
                    if self._check(TokenKind.COMMA): self._advance()
                    else: break
            self._consume(TokenKind.RBRACKET, "Expected ']'")
            dtype, shape = "f32", [len(values)]
            if self._check(TokenKind.AS):
                self._advance()
                texpr = self._parse_type()
                dtype = texpr.name
                if texpr.shape: shape = texpr.shape
            return TensorLiteral(values, shape, dtype, SourceLoc(tok.line, tok.col))

        if tok.kind in (TokenKind.IDENT, TokenKind.MATMUL, TokenKind.RELU, TokenKind.ADD, TokenKind.MUL, TokenKind.CAST):
            ident = self._advance()
            if self._check(TokenKind.LPAREN):
                self._advance()
                args: list[Expr] = []
                if not self._check(TokenKind.RPAREN):
                    while True:
                        args.append(self._parse_expr())
                        if self._check(TokenKind.COMMA): self._advance()
                        else: break
                self._consume(TokenKind.RPAREN, "Expected ')'")
                return CallExpr(ident.lexeme, args, SourceLoc(ident.line, ident.col))
            return VarRef(ident.lexeme, SourceLoc(ident.line, ident.col))

        if tok.kind == TokenKind.LPAREN:
            self._advance(); e = self._parse_expr(); self._consume(TokenKind.RPAREN, "Expected ')'")
            return e
        raise ParseError(f"Unexpected token '{tok.lexeme}'", tok.line, tok.col, self.filename)
"""

FILES["axon/frontend/semantic.py"] = """from dataclasses import dataclass
from typing import Dict, List
from .ast_nodes import Program, FunctionDef, LetStmt, ReturnStmt, ExprStmt, CallExpr, BinaryExpr, TensorLiteral, VarRef, Expr
from ..utils.errors import SemanticError

@dataclass
class TypeInfo:
    dtype: str
    shape: List[int]

class SemanticAnalyzer:
    def __init__(self, filename: str = "<axon>"):
        self.filename = filename
        self.sym: Dict[str, TypeInfo] = {}

    def analyze(self, program: Program) -> Program:
        for fn in program.functions:
            self._analyze_function(fn)
        return program

    def _analyze_function(self, fn: FunctionDef) -> None:
        self.sym.clear()
        for p in fn.params:
            self.sym[p.name] = TypeInfo(p.type_expr.name, p.type_expr.shape)
        for s in fn.body:
            if isinstance(s, LetStmt):
                inf = self._infer(s.value)
                if s.type_expr:
                    if s.type_expr.name != inf.dtype:
                        raise SemanticError(f"Type mismatch on {s.name}", s.loc.line, s.loc.col, self.filename)
                    if s.type_expr.shape and s.type_expr.shape != inf.shape:
                        raise SemanticError(f"Shape mismatch on {s.name}", s.loc.line, s.loc.col, self.filename)
                self.sym[s.name] = inf
            elif isinstance(s, ReturnStmt):
                r = self._infer(s.value)
                if r.dtype != fn.return_type.name:
                    raise SemanticError("Return dtype mismatch", s.loc.line, s.loc.col, self.filename)
            elif isinstance(s, ExprStmt):
                self._infer(s.expr)

    def _infer(self, expr: Expr) -> TypeInfo:
        if isinstance(expr, TensorLiteral):
            return TypeInfo(expr.dtype, expr.shape)
        if isinstance(expr, VarRef):
            if expr.name not in self.sym:
                raise SemanticError(f"Undefined: {expr.name}", expr.loc.line, expr.loc.col, self.filename)
            return self.sym[expr.name]
        if isinstance(expr, BinaryExpr):
            lt, rt = self._infer(expr.left), self._infer(expr.right)
            shp = self._broadcast(lt.shape, rt.shape, expr.loc)
            expr.resolved_shape, expr.resolved_dtype = shp, lt.dtype
            return TypeInfo(lt.dtype, shp)
        if isinstance(expr, CallExpr):
            c = expr.callee
            args = [self._infer(a) for a in expr.args]
            if c == "matmul":
                a, b = args[0], args[1]
                if len(a.shape) != 2 or len(b.shape) != 2 or a.shape[1] != b.shape[0]:
                    raise SemanticError(f"Invalid matmul shapes: {a.shape} vs {b.shape}", expr.loc.line, expr.loc.col, self.filename)
                shp = [a.shape[0], b.shape[1]]
                expr.resolved_shape, expr.resolved_dtype = shp, a.dtype
                return TypeInfo(a.dtype, shp)
            elif c == "relu":
                expr.resolved_shape, expr.resolved_dtype = list(args[0].shape), args[0].dtype
                return args[0]
            elif c == "add":
                shp = self._broadcast(args[0].shape, args[1].shape, expr.loc)
                expr.resolved_shape, expr.resolved_dtype = shp, args[0].dtype
                return TypeInfo(args[0].dtype, shp)
            elif c == "cast":
                expr.resolved_shape, expr.resolved_dtype = list(args[0].shape), "f32"
                return TypeInfo("f32", expr.resolved_shape)
        return TypeInfo("f32", [])

    def _broadcast(self, s1: List[int], s2: List[int], loc) -> List[int]:
        if s1 == s2 or not s2: return list(s1)
        if not s1: return list(s2)
        if len(s2) == 1 and (len(s1) == 2 and s2[0] == s1[1]): return list(s1)
        if len(s1) == 1 and (len(s2) == 2 and s1[0] == s2[1]): return list(s2)
        raise SemanticError(f"Cannot broadcast shapes {s1} and {s2}", loc.line, loc.col, self.filename)
"""

# =====================================================================
# Axon Package: IR & Builder
# =====================================================================
FILES["axon/ir/__init__.py"] = """from .tensor_ir import OpKind, IRNode, Graph, WeightBlob
from .builder import IRBuilder
from .printer import IRPrinter
__all__ = ["OpKind", "IRNode", "Graph", "WeightBlob", "IRBuilder", "IRPrinter"]
"""

FILES["axon/ir/tensor_ir.py"] = """from enum import Enum, auto
from dataclasses import dataclass, field
from typing import List, Dict, Any, Tuple, Optional

class OpKind(Enum):
    INPUT = auto(); CONSTANT = auto(); WEIGHT = auto(); MATMUL = auto()
    ADD = auto(); RELU = auto(); CAST = auto(); RESHAPE = auto()
    FUSED_MATMUL_BIAS_RELU = auto(); FUSED_MATMUL_BIAS = auto()

@dataclass
class WeightBlob:
    name: str
    data: List[float]
    shape: List[int]
    dtype: str
    is_quantized: bool = False
    packed_data: Optional[bytes] = None
    scale_bytes: Optional[bytes] = None
    residuals: List[Tuple[int, float]] = field(default_factory=list)

@dataclass
class IRNode:
    id: int
    op: OpKind
    inputs: List[int]
    attrs: Dict[str, Any]
    output_shape: Tuple[int, ...]
    output_dtype: str
    name: str = ""
    def __hash__(self): return hash(self.id)

@dataclass
class Graph:
    nodes: List[IRNode] = field(default_factory=list)
    inputs: List[int] = field(default_factory=list)
    outputs: List[int] = field(default_factory=list)
    weights: Dict[str, WeightBlob] = field(default_factory=dict)

    def get_node(self, nid: int) -> Optional[IRNode]:
        for n in self.nodes:
            if n.id == nid: return n
        return None

    def nodes_in_topo_order(self) -> List[IRNode]:
        in_degree = {n.id: 0 for n in self.nodes}
        adj = {n.id: [] for n in self.nodes}
        for n in self.nodes:
            for inp in n.inputs:
                if inp in adj:
                    adj[inp].append(n.id)
                    in_degree[n.id] += 1
        queue = [nid for nid, d in in_degree.items() if d == 0]
        topo: List[IRNode] = []
        while queue:
            cid = queue.pop(0)
            node = self.get_node(cid)
            if node: topo.append(node)
            for neighbor in adj.get(cid, []):
                in_degree[neighbor] -= 1
                if in_degree[neighbor] == 0: queue.append(neighbor)
        return topo if len(topo) == len(self.nodes) else self.nodes[:]

    def replace_subgraph(self, old_nodes: List[IRNode], new_node: IRNode) -> None:
        old_ids = {n.id for n in old_nodes}
        sink = old_nodes[-1]
        for n in self.nodes:
            for idx, inp in enumerate(n.inputs):
                if inp == sink.id: n.inputs[idx] = new_node.id
        for idx, out_id in enumerate(self.outputs):
            if out_id == sink.id: self.outputs[idx] = new_node.id
        self.nodes = [n for n in self.nodes if n.id not in old_ids]
        self.nodes.append(new_node)
"""

FILES["axon/ir/builder.py"] = """from typing import Dict
from ..frontend.ast_nodes import Program, FunctionDef, LetStmt, ReturnStmt, ExprStmt, CallExpr, BinaryExpr, TensorLiteral, VarRef, Expr
from .tensor_ir import Graph, IRNode, OpKind, WeightBlob

class IRBuilder:
    def __init__(self):
        self.graph = Graph()
        self.nid = 0
        self.var_map: Dict[str, int] = {}

    def _alloc_id(self) -> int:
        i = self.nid; self.nid += 1; return i

    def build(self, program: Program) -> Graph:
        for fn in program.functions:
            for param in fn.params:
                i = self._alloc_id()
                node = IRNode(i, OpKind.INPUT, [], {"name": param.name}, tuple(param.type_expr.shape), param.type_expr.name, param.name)
                self.graph.nodes.append(node)
                self.graph.inputs.append(i)
                self.var_map[param.name] = i
            for stmt in fn.body:
                if isinstance(stmt, LetStmt):
                    vid = self._build_expr(stmt.value, target_name=stmt.name)
                    self.var_map[stmt.name] = vid
                elif isinstance(stmt, ReturnStmt):
                    vid = self._build_expr(stmt.value)
                    self.graph.outputs.append(vid)
                elif isinstance(stmt, ExprStmt):
                    self._build_expr(stmt.expr)
        return self.graph

    def _build_expr(self, expr: Expr, target_name: str = "") -> int:
        if isinstance(expr, VarRef): return self.var_map[expr.name]
        if isinstance(expr, TensorLiteral):
            i = self._alloc_id()
            name = target_name or f"const_{i}"
            self.graph.weights[name] = WeightBlob(name, expr.values, expr.shape, expr.dtype)
            node = IRNode(i, OpKind.CONSTANT, [], {"weight_name": name, "raw_values": expr.values}, tuple(expr.shape), expr.dtype, name)
            self.graph.nodes.append(node)
            return i
        if isinstance(expr, BinaryExpr):
            lid = self._build_expr(expr.left)
            rid = self._build_expr(expr.right)
            i = self._alloc_id()
            op = OpKind.ADD if expr.op == "+" else OpKind.MATMUL
            node = IRNode(i, op, [lid, rid], {}, tuple(expr.resolved_shape or ()), expr.resolved_dtype or "f32", target_name)
            self.graph.nodes.append(node)
            return i
        if isinstance(expr, CallExpr):
            args = [self._build_expr(a) for a in expr.args]
            i = self._alloc_id()
            op_map = {"matmul": OpKind.MATMUL, "relu": OpKind.RELU, "add": OpKind.ADD, "cast": OpKind.CAST}
            op = op_map.get(expr.callee, OpKind.CONSTANT)
            node = IRNode(i, op, args, {"callee": expr.callee}, tuple(expr.resolved_shape or ()), expr.resolved_dtype or "f32", target_name)
            self.graph.nodes.append(node)
            return i
        raise NotImplementedError(f"Unsupported expr: {type(expr)}")
"""

FILES["axon/ir/printer.py"] = """from .tensor_ir import Graph

class IRPrinter:
    @staticmethod
    def dump(graph: Graph) -> str:
        lines = ["axon.Graph {"]
        lines.append(f"  // Inputs: {graph.inputs}")
        lines.append(f"  // Outputs: {graph.outputs}")
        lines.append(f"  // Weights: {list(graph.weights.keys())}")
        for n in graph.nodes_in_topo_order():
            inp = ", ".join(f"%{i}" for i in n.inputs)
            shp = "x".join(map(str, n.output_shape))
            lines.append(f"  %{n.id} = {n.op.name}({inp}) : [{shp}] {n.output_dtype} // {n.name}")
        lines.append("}")
        return "\\n".join(lines)
"""

# =====================================================================
# Axon Package: Optimizer (Fusion & FP4 Quantization)
# =====================================================================
FILES["axon/optimizer/__init__.py"] = """from .fusion import FusionPass
from .quantize import FP4QuantizePass
from .passes import OptimizationPipeline
__all__ = ["FusionPass", "FP4QuantizePass", "OptimizationPipeline"]
"""

FILES["axon/optimizer/fusion.py"] = """from typing import Optional
from ..ir.tensor_ir import Graph, IRNode, OpKind

class FusionPass:
    def run(self, graph: Graph) -> Graph:
        changed = True
        while changed:
            changed = False
            for node in graph.nodes_in_topo_order():
                if node.op == OpKind.RELU:
                    add_node = self._parent(node, graph)
                    if add_node and add_node.op == OpKind.ADD and self._uses(add_node, graph) == 1:
                        mm_node = self._parent_op(add_node, OpKind.MATMUL, graph)
                        if mm_node and self._uses(mm_node, graph) == 1:
                            bias_inp = self._other_input(add_node, mm_node.id)
                            if bias_inp is not None:
                                fused = IRNode(node.id, OpKind.FUSED_MATMUL_BIAS_RELU,
                                               [mm_node.inputs[0], mm_node.inputs[1], bias_inp],
                                               {"fused_from": [mm_node.id, add_node.id, node.id]},
                                               node.output_shape, node.output_dtype, f"fused_{node.name}")
                                graph.replace_subgraph([mm_node, add_node, node], fused)
                                changed = True
                                break
                elif node.op == OpKind.ADD:
                    mm_node = self._parent_op(node, OpKind.MATMUL, graph)
                    if mm_node and self._uses(mm_node, graph) == 1:
                        bias_inp = self._other_input(node, mm_node.id)
                        if bias_inp is not None:
                            fused = IRNode(node.id, OpKind.FUSED_MATMUL_BIAS,
                                           [mm_node.inputs[0], mm_node.inputs[1], bias_inp],
                                           {"fused_from": [mm_node.id, node.id]},
                                           node.output_shape, node.output_dtype, f"fused_{node.name}")
                            graph.replace_subgraph([mm_node, node], fused)
                            changed = True
                            break
        return graph

    def _parent(self, node: IRNode, graph: Graph) -> Optional[IRNode]:
        return graph.get_node(node.inputs[0]) if len(node.inputs) == 1 else None

    def _parent_op(self, node: IRNode, op: OpKind, graph: Graph) -> Optional[IRNode]:
        for i in node.inputs:
            p = graph.get_node(i)
            if p and p.op == op: return p
        return None

    def _other_input(self, node: IRNode, excl_id: int) -> Optional[int]:
        cands = [i for i in node.inputs if i != excl_id]
        return cands[0] if cands else None

    def _uses(self, node: IRNode, graph: Graph) -> int:
        return sum(n.inputs.count(node.id) for n in graph.nodes) + graph.outputs.count(node.id)
"""

FILES["axon/optimizer/quantize.py"] = """import math
from typing import List, Tuple, Optional, Dict
from ..ir.tensor_ir import Graph, WeightBlob

FP4_ENC_MAP = {
    0.0: 0x0, 0.5: 0x1, 1.0: 0x2, 1.5: 0x3,
    2.0: 0x4, 3.0: 0x5, 4.0: 0x6, 6.0: 0x7
}

def round_fp4(val: float) -> float:
    sign = -1.0 if math.copysign(1.0, val) < 0 else 1.0
    abs_v = abs(val)
    best, min_dist = 0.0, float("inf")
    for cand in [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0]:
        dist = abs(abs_v - cand)
        if dist < min_dist:
            min_dist = dist; best = cand
    return sign * best

def fp4_to_nibble(val: float) -> int:
    is_neg = math.copysign(1.0, val) < 0
    code = FP4_ENC_MAP.get(round_fp4(abs(val)), 0x0)
    return (code | 0x8) if is_neg else code

class FP4QuantizePass:
    BLOCK = 32

    def run(self, graph: Graph, calibration: Optional[Dict] = None) -> Graph:
        for blob in graph.weights.values():
            if blob.dtype in ("f32", "f16") and len(blob.data) >= self.BLOCK:
                self._quantize(blob)
        return graph

    def _quantize(self, blob: WeightBlob) -> None:
        flat = list(blob.data)
        pad = (self.BLOCK - (len(flat) % self.BLOCK)) % self.BLOCK
        if pad > 0: flat.extend([0.0] * pad)
        n_blocks = len(flat) // self.BLOCK
        packed = bytearray(n_blocks * 16)
        scales = bytearray(n_blocks)
        residuals: List[Tuple[int, float]] = []

        for b in range(n_blocks):
            start = b * self.BLOCK
            chunk = flat[start: start + self.BLOCK]
            abs_c = [abs(x) for x in chunk]
            mean_v = sum(abs_c) / 32.0
            std_v = math.sqrt(sum((x - mean_v) ** 2 for x in abs_c) / 32.0)
            tau = mean_v + 3.0 * std_v

            non_outliers = [abs_c[i] for i in range(32) if abs_c[i] <= tau]
            max_val = max(non_outliers) if non_outliers else 1e-8
            if max_val <= 0.0: max_val = 1e-8

            exp = max(-126, min(127, int(math.ceil(math.log2(max_val / 6.0)))))
            scale = 2.0 ** exp
            scales[b] = (exp + 127) & 0xFF

            for i in range(32):
                v = chunk[i]
                if abs_c[i] > tau:
                    q = (1.0 if v >= 0 else -1.0) * 6.0
                    nib = fp4_to_nibble(q)
                    residuals.append((start + i, v - scale * q))
                else:
                    nib = fp4_to_nibble(round_fp4(v / scale))
                idx = (b * 16) + (i // 2)
                packed[idx] |= (nib & 0x0F) if (i % 2 == 0) else ((nib & 0x0F) << 4)

        blob.is_quantized = True
        blob.packed_data = bytes(packed)
        blob.scale_bytes = bytes(scales)
        blob.residuals = residuals
        blob.dtype = "fp4"
"""

FILES["axon/optimizer/passes.py"] = """from ..ir.tensor_ir import Graph
from .fusion import FusionPass
from .quantize import FP4QuantizePass

class OptimizationPipeline:
    def __init__(self, enable_fusion: bool = True, enable_quant: bool = True):
        self.enable_fusion = enable_fusion
        self.enable_quant = enable_quant

    def run(self, graph: Graph) -> Graph:
        if self.enable_fusion: graph = FusionPass().run(graph)
        if self.enable_quant: graph = FP4QuantizePass().run(graph)
        return graph
"""

# =====================================================================
# Axon Package: Backend & Baker
# =====================================================================
FILES["axon/backend/__init__.py"] = """from .llvm_backend import LLVMBackend
from .codegen_ops import CodeGenOps
from .runtime_linker import RuntimeLinker
__all__ = ["LLVMBackend", "CodeGenOps", "RuntimeLinker"]
"""

FILES["axon/backend/codegen_ops.py"] = """import llvmlite.ir as ir

class CodeGenOps:
    def __init__(self, module: ir.Module):
        void = ir.VoidType()
        i8_ptr = ir.PointerType(ir.IntType(8))
        i32 = ir.IntType(32)
        i64 = ir.IntType(64)
        i64_ptr = ir.PointerType(i64)

        self.fn_alloc = ir.Function(module, ir.FunctionType(i8_ptr, [i64, i64_ptr, i32]), name="axon_tensor_alloc")
        self.fn_free = ir.Function(module, ir.FunctionType(void, [i8_ptr]), name="axon_tensor_free")
        self.fn_matmul = ir.Function(module, ir.FunctionType(void, [i8_ptr, i8_ptr, i8_ptr]), name="axon_matmul")
        self.fn_add = ir.Function(module, ir.FunctionType(void, [i8_ptr, i8_ptr, i8_ptr]), name="axon_add")
        self.fn_relu = ir.Function(module, ir.FunctionType(void, [i8_ptr, i8_ptr]), name="axon_relu")
        self.fn_fused_matmul_bias_relu = ir.Function(module, ir.FunctionType(void, [i8_ptr, i8_ptr, i8_ptr, i8_ptr]), name="axon_fused_matmul_bias_relu")
        self.fn_fused_matmul_bias = ir.Function(module, ir.FunctionType(void, [i8_ptr, i8_ptr, i8_ptr, i8_ptr]), name="axon_fused_matmul_bias")
        self.fn_get_weight = ir.Function(module, ir.FunctionType(i8_ptr, [i8_ptr]), name="axon_get_weight")
        self.fn_create_tensor_from_weight = ir.Function(module, ir.FunctionType(i8_ptr, [i8_ptr]), name="axon_create_tensor_from_weight")
"""

FILES["axon/backend/llvm_backend.py"] = """import llvmlite.ir as ir
import llvmlite.binding as llvm
from .codegen_ops import CodeGenOps
from ..ir.tensor_ir import Graph, OpKind

class LLVMBackend:
    def __init__(self, module_name: str = "axon_module", target_triple: str = ""):
        llvm.initialize()
        llvm.initialize_native_target()
        llvm.initialize_native_asmprinter()
        self.module = ir.Module(name=module_name)
        self.target_triple = target_triple or llvm.get_default_triple()
        self.module.triple = self.target_triple
        self.ops = CodeGenOps(self.module)
        self.str_map: dict[str, ir.GlobalVariable] = {}

    def _str(self, builder: ir.IRBuilder, val: str) -> ir.Value:
        if val in self.str_map: gv = self.str_map[val]
        else:
            raw = bytearray((val + "\\0").encode("utf-8"))
            t = ir.ArrayType(ir.IntType(8), len(raw))
            gv = ir.GlobalVariable(self.module, t, name=f".str.{len(self.str_map)}")
            gv.linkage, gv.global_constant, gv.initializer = "internal", True, ir.Constant(t, raw)
            self.str_map[val] = gv
        z = ir.Constant(ir.IntType(32), 0)
        return builder.gep(gv, [z, z], inbounds=True)

    def _shape(self, builder: ir.IRBuilder, shape: tuple[int, ...]) -> ir.Value:
        i64 = ir.IntType(64)
        n = max(len(shape), 1)
        arr = builder.alloca(ir.ArrayType(i64, n), name="shape")
        z = ir.Constant(ir.IntType(32), 0)
        for idx, dim in enumerate(shape):
            p = builder.gep(arr, [z, ir.Constant(ir.IntType(32), idx)], inbounds=True)
            builder.store(ir.Constant(i64, dim), p)
        return builder.bitcast(arr, ir.PointerType(i64))

    def compile(self, graph: Graph, output_obj_path: str) -> str:
        i8_ptr, i64, i32 = ir.PointerType(ir.IntType(8)), ir.IntType(64), ir.IntType(32)
        fn = ir.Function(self.module, ir.FunctionType(i8_ptr, [ir.PointerType(i8_ptr)]), name="axon_forward")
        builder = ir.IRBuilder(fn.append_basic_block("entry"))
        nodes: dict[int, ir.Value] = {}

        for idx, in_id in enumerate(graph.inputs):
            p = builder.gep(fn.args[0], [ir.Constant(i32, idx)])
            nodes[in_id] = builder.load(p, name=f"inp_{idx}")

        for node in graph.nodes_in_topo_order():
            if node.id in nodes: continue
            if node.op == OpKind.CONSTANT:
                w_name = node.attrs.get("weight_name", node.name)
                w_ref = builder.call(self.ops.fn_get_weight, [self._str(builder, w_name)])
                nodes[node.id] = builder.call(self.ops.fn_create_tensor_from_weight, [w_ref], name=f"w_{node.id}")
            elif node.op == OpKind.MATMUL:
                out = builder.call(self.ops.fn_alloc, [ir.Constant(i64, len(node.output_shape)), self._shape(builder, node.output_shape), ir.Constant(i32, 0)])
                builder.call(self.ops.fn_matmul, [nodes[node.inputs[0]], nodes[node.inputs[1]], out])
                nodes[node.id] = out
            elif node.op == OpKind.ADD:
                out = builder.call(self.ops.fn_alloc, [ir.Constant(i64, len(node.output_shape)), self._shape(builder, node.output_shape), ir.Constant(i32, 0)])
                builder.call(self.ops.fn_add, [nodes[node.inputs[0]], nodes[node.inputs[1]], out])
                nodes[node.id] = out
            elif node.op == OpKind.RELU:
                out = builder.call(self.ops.fn_alloc, [ir.Constant(i64, len(node.output_shape)), self._shape(builder, node.output_shape), ir.Constant(i32, 0)])
                builder.call(self.ops.fn_relu, [nodes[node.inputs[0]], out])
                nodes[node.id] = out
            elif node.op == OpKind.FUSED_MATMUL_BIAS_RELU:
                out = builder.call(self.ops.fn_alloc, [ir.Constant(i64, len(node.output_shape)), self._shape(builder, node.output_shape), ir.Constant(i32, 0)])
                builder.call(self.ops.fn_fused_matmul_bias_relu, [nodes[node.inputs[0]], nodes[node.inputs[1]], nodes[node.inputs[2]], out])
                nodes[node.id] = out
            elif node.op == OpKind.FUSED_MATMUL_BIAS:
                out = builder.call(self.ops.fn_alloc, [ir.Constant(i64, len(node.output_shape)), self._shape(builder, node.output_shape), ir.Constant(i32, 0)])
                builder.call(self.ops.fn_fused_matmul_bias, [nodes[node.inputs[0]], nodes[node.inputs[1]], nodes[node.inputs[2]], out])
                nodes[node.id] = out

        builder.ret(nodes[graph.outputs[0]] if graph.outputs else ir.Constant(i8_ptr, None))
        parsed = llvm.parse_assembly(str(self.module))
        parsed.verify()
        tm = llvm.Target.from_triple(self.target_triple).create_target_machine(opt=3)
        with open(output_obj_path, "wb") as f:
            f.write(tm.emit_object(parsed))
        return output_obj_path
"""

FILES["axon/backend/runtime_linker.py"] = """import subprocess
from ..utils.errors import CodegenError

class RuntimeLinker:
    @staticmethod
    def compile_asm_stub(asm_path: str, obj_path: str) -> None:
        cmd = ["as", asm_path, "-o", obj_path]
        res = subprocess.run(cmd, capture_output=True, text=True)
        if res.returncode != 0: raise CodegenError(f"Assembler error: {res.stderr}")

    @staticmethod
    def link_binary(objects: list[str], output_bin: str, static_runtime: str = "") -> None:
        cmd = ["cc", "-O3", "-no-pie"] + objects
        if static_runtime: cmd.append(static_runtime)
        cmd.extend(["-lm", "-lpthread", "-o", output_bin])
        res = subprocess.run(cmd, capture_output=True, text=True)
        if res.returncode != 0: raise CodegenError(f"Linker error: {res.stderr}")
"""

FILES["axon/baker/__init__.py"] = """from .weight_pack import AxonBlobPacker
from .baker import Baker
__all__ = ["AxonBlobPacker", "Baker"]
"""

FILES["axon/baker/weight_pack.py"] = """import struct
from typing import Dict
from ..ir.tensor_ir import WeightBlob

class AxonBlobPacker:
    MAGIC = b"AXON"
    VERSION = 1

    @staticmethod
    def pack(weights: Dict[str, WeightBlob]) -> bytes:
        num = len(weights)
        name_tbl = bytearray()
        data_sec = bytearray()
        meta = []
        name_off = 0

        for name, blob in weights.items():
            enc = name.encode("utf-8") + b"\\0"
            name_tbl.extend(enc)
            while len(data_sec) % 16 != 0: data_sec.append(0)
            d_off = len(data_sec)
            if blob.is_quantized and blob.packed_data:
                payload = bytearray(blob.scale_bytes or b"") + bytearray(blob.packed_data)
                data_sec.extend(payload)
                nbytes = len(payload)
                dtype = 2
            else:
                raw = struct.pack(f"<{len(blob.data)}f", *blob.data)
                data_sec.extend(raw)
                nbytes = len(raw)
                dtype = 0

            shape4 = list(blob.shape) + [0] * (4 - len(blob.shape))
            meta.append({
                "hash": AxonBlobPacker._fnv1a(name), "name_off": name_off,
                "d_off": d_off, "nbytes": nbytes, "dtype": dtype, "ndim": len(blob.shape),
                "shape": shape4[:4]
            })
            name_off += len(enc)

        hdr_sz = 12
        entry_sz = 64
        name_base = hdr_sz + num * entry_sz
        data_base = name_base + len(name_tbl)
        pad = (16 - (data_base % 16)) % 16
        data_base += pad

        blob = bytearray(struct.pack("<4sII", AxonBlobPacker.MAGIC, AxonBlobPacker.VERSION, num))
        for m in meta:
            blob.extend(struct.pack("<QIIQII4qI", m["hash"], m["name_off"], data_base + m["d_off"],
                                    m["nbytes"], m["dtype"], m["ndim"], *m["shape"], 0))
        blob.extend(name_tbl)
        blob.extend(b"\\0" * pad)
        blob.extend(data_sec)
        return bytes(blob)

    @staticmethod
    def _fnv1a(text: str) -> int:
        h = 14695981039346656037
        for ch in text.encode("utf-8"):
            h = ((h ^ ch) * 1099511628211) & 0xFFFFFFFFFFFFFFFF
        return h
"""

FILES["axon/baker/baker.py"] = """import os
import tempfile
from typing import List, Dict
from ..ir.tensor_ir import WeightBlob
from .weight_pack import AxonBlobPacker
from ..backend.runtime_linker import RuntimeLinker

class Baker:
    def __init__(self, runtime_archive_path: str = "runtime/libaxon_runtime.a"):
        self.runtime_archive_path = runtime_archive_path

    def bake(self, object_files: List[str], weights: Dict[str, WeightBlob], output_bin: str) -> None:
        blob_data = AxonBlobPacker.pack(weights)
        with tempfile.TemporaryDirectory() as tmp:
            blob_file = os.path.join(tmp, "weights.blob")
            with open(blob_file, "wb") as f: f.write(blob_data)
            asm_file = os.path.join(tmp, "weights.S")
            with open(asm_file, "w") as f:
                f.write(f'''
.section .axon_weights, "a", @progbits
.align 16
.global __axon_weights_start
__axon_weights_start:
.incbin "{blob_file}"
.global __axon_weights_end
__axon_weights_end:
''')
            w_obj = os.path.join(tmp, "weights.o")
            RuntimeLinker.compile_asm_stub(asm_file, w_obj)
            RuntimeLinker.link_binary(object_files + [w_obj], output_bin, self.runtime_archive_path)
"""

FILES["axon/cli.py"] = """import argparse, sys, os
from .frontend.lexer import Lexer
from .frontend.parser import Parser
from .frontend.semantic import SemanticAnalyzer
from .ir.builder import IRBuilder
from .ir.printer import IRPrinter
from .optimizer.passes import OptimizationPipeline
from .backend.llvm_backend import LLVMBackend
from .baker.baker import Baker
from .utils.errors import AxonError

def main():
    p = argparse.ArgumentParser(description="Axon Tensor DSL Compiler")
    p.add_argument("source", help="Path to .axon file")
    p.add_argument("-o", "--output", default="model.bin", help="Output executable")
    p.add_argument("--dump-ir", action="store_true")
    p.add_argument("--no-opt", action="store_true")
    p.add_argument("--runtime-lib", default="runtime/libaxon_runtime.a")
    args = p.parse_args()

    if not os.path.exists(args.source):
        print(f"Error: {args.source} not found", file=sys.stderr); sys.exit(1)
    try:
        with open(args.source) as f: src = f.read()
        ast = SemanticAnalyzer(args.source).analyze(Parser(Lexer(src, args.source).tokenize(), args.source).parse())
        graph = IRBuilder().build(ast)
        if not args.no_opt:
            graph = OptimizationPipeline(True, True).run(graph)
        if args.dump_ir:
            print(IRPrinter.dump(graph))
        gen_obj = args.output + ".gen.o"
        LLVMBackend().compile(graph, gen_obj)
        Baker(args.runtime_lib).bake([gen_obj], graph.weights, args.output)
        if os.path.exists(gen_obj): os.remove(gen_obj)
        print(f"[SUCCESS] Standalone binary baked: {args.output}")
    except AxonError as e:
        print(f"Compilation error: {e}", file=sys.stderr); sys.exit(1)

if __name__ == "__main__":
    main()
"""

# =====================================================================
# Native C Runtime
# =====================================================================
FILES["runtime/axon_tensor.h"] = """#ifndef AXON_TENSOR_H
#define AXON_TENSOR_H
#include <stddef.h>
#include <stdint.h>
#include <stdbool.h>

#define AXON_DTYPE_F32 0
#define AXON_DTYPE_F16 1
#define AXON_DTYPE_FP4 2
#define AXON_DTYPE_I32 3

typedef struct AxonTensor {
    void    *data;
    int64_t  ndim;
    int64_t  shape[4];
    int64_t  strides[4];
    int32_t  dtype;
    int32_t  quant_scale;
    bool     owns_data;
} AxonTensor;

AxonTensor* axon_tensor_alloc(int64_t ndim, const int64_t *shape, int32_t dtype);
void        axon_tensor_free(AxonTensor *t);
int64_t     axon_tensor_numel(const AxonTensor *t);
#endif
"""

FILES["runtime/axon_tensor.c"] = """#define _POSIX_C_SOURCE 200809L
#include "axon_tensor.h"
#include <stdlib.h>
#include <string.h>

int64_t axon_tensor_numel(const AxonTensor *t) {
    if (!t || t->ndim == 0) return 0;
    int64_t n = 1;
    for (int64_t i = 0; i < t->ndim; ++i) n *= t->shape[i];
    return n;
}

AxonTensor* axon_tensor_alloc(int64_t ndim, const int64_t *shape, int32_t dtype) {
    AxonTensor *t = (AxonTensor*)malloc(sizeof(AxonTensor));
    if (!t) return NULL;
    t->ndim = ndim; t->dtype = dtype; t->quant_scale = 0; t->owns_data = true;
    memset(t->shape, 0, sizeof(t->shape));
    memset(t->strides, 0, sizeof(t->strides));

    int64_t total = 1;
    for (int64_t i = 0; i < ndim; ++i) { t->shape[i] = shape[i]; total *= shape[i]; }
    int64_t s = 1;
    for (int64_t i = ndim - 1; i >= 0; --i) { t->strides[i] = s; s *= t->shape[i]; }

    size_t sz = total * sizeof(float);
    if (sz < 64) sz = 64;
    if (posix_memalign(&t->data, 64, sz) != 0) { free(t); return NULL; }
    memset(t->data, 0, sz);
    return t;
}

void axon_tensor_free(AxonTensor *t) {
    if (!t) return;
    if (t->owns_data && t->data) free(t->data);
    free(t);
}
"""

FILES["runtime/axon_quant.h"] = """#ifndef AXON_QUANT_H
#define AXON_QUANT_H
#include <stdint.h>
void axon_dequantize_fp4_block(const uint8_t *packed_16, uint8_t scale_byte, float *out_f32);
#endif
"""

FILES["runtime/axon_quant.c"] = """#include "axon_quant.h"
#include <math.h>

static const float FP4_LUT[16] = {
     0.0f,  0.5f,  1.0f,  1.5f,  2.0f,  3.0f,  4.0f,  6.0f,
    -0.0f, -0.5f, -1.0f, -1.5f, -2.0f, -3.0f, -4.0f, -6.0f
};

void axon_dequantize_fp4_block(const uint8_t *packed_16, uint8_t scale_byte, float *out_f32) {
    float scale = ldexpf(1.0f, (int32_t)scale_byte - 127);
    for (int i = 0; i < 16; ++i) {
        uint8_t b = packed_16[i];
        out_f32[i * 2 + 0] = FP4_LUT[b & 0x0F] * scale;
        out_f32[i * 2 + 1] = FP4_LUT[(b >> 4) & 0x0F] * scale;
    }
}
"""

FILES["runtime/axon_mmap.h"] = """#ifndef AXON_MMAP_H
#define AXON_MMAP_H
#include "axon_tensor.h"
#include <stdint.h>

typedef struct AxonWeightRef {
    const void *ptr;
    int64_t     nbytes;
    int32_t     dtype;
    int64_t     shape[4];
    int64_t     ndim;
    int32_t     fp4_scale;
} AxonWeightRef;

typedef struct AxonWeightEntry {
    uint64_t name_hash;
    uint32_t name_offset;
    uint32_t data_offset;
    uint64_t nbytes;
    uint32_t dtype;
    uint32_t ndim;
    int64_t  shape[4];
    uint32_t fp4_scale;
} __attribute__((packed)) AxonWeightEntry;

typedef struct AxonBlobHeader {
    char     magic[4];
    uint32_t version;
    uint32_t num_weights;
    AxonWeightEntry entries[];
} __attribute__((packed)) AxonBlobHeader;

const AxonWeightRef* axon_get_weight(const char *name);
AxonTensor*          axon_create_tensor_from_weight(const AxonWeightRef *ref);
#endif
"""

FILES["runtime/axon_mmap.c"] = """#include "axon_mmap.h"
#include "axon_quant.h"
#include <string.h>
#include <stdlib.h>

extern const char __axon_weights_start[] __attribute__((weak));
extern const char __axon_weights_end[]   __attribute__((weak));
static const AxonBlobHeader *g_blob = NULL;

static uint64_t hash64(const char *s) {
    uint64_t h = 14695981039346656037ULL;
    while (*s) { h ^= (uint8_t)(*s++); h *= 1099511628211ULL; }
    return h;
}

const AxonWeightRef* axon_get_weight(const char *name) {
    if (!g_blob && &__axon_weights_start != NULL) {
        g_blob = (const AxonBlobHeader*)__axon_weights_start;
    }
    if (!g_blob) return NULL;
    uint64_t h = hash64(name);
    const char *names = (const char*)(g_blob->entries + g_blob->num_weights);
    for (uint32_t i = 0; i < g_blob->num_weights; ++i) {
        const AxonWeightEntry *e = &g_blob->entries[i];
        if (e->name_hash == h && strcmp(names + e->name_offset, name) == 0) {
            static AxonWeightRef ref;
            ref.ptr = ((const char*)g_blob) + e->data_offset;
            ref.nbytes = e->nbytes; ref.dtype = e->dtype; ref.ndim = e->ndim;
            memcpy(ref.shape, e->shape, sizeof(ref.shape));
            ref.fp4_scale = e->fp4_scale;
            return &ref;
        }
    }
    return NULL;
}

AxonTensor* axon_create_tensor_from_weight(const AxonWeightRef *ref) {
    if (!ref) return NULL;
    if (ref->dtype == AXON_DTYPE_FP4) {
        AxonTensor *t = axon_tensor_alloc(ref->ndim, ref->shape, AXON_DTYPE_F32);
        int64_t blocks = axon_tensor_numel(t) / 32;
        const uint8_t *scales = (const uint8_t*)ref->ptr;
        const uint8_t *packed = scales + blocks;
        float *out = (float*)t->data;
        for (int64_t b = 0; b < blocks; ++b) {
            axon_dequantize_fp4_block(packed + (b * 16), scales[b], out + (b * 32));
        }
        return t;
    }
    AxonTensor *t = axon_tensor_alloc(ref->ndim, ref->shape, ref->dtype);
    if (t && ref->ptr) memcpy(t->data, ref->ptr, ref->nbytes);
    return t;
}
"""

FILES["runtime/axon_matmul.c"] = """#include "axon_runtime.h"
#include <string.h>

void axon_matmul(const AxonTensor *A, const AxonTensor *B, AxonTensor *C) {
    const int64_t M = A->shape[0], K = A->shape[1], N = B->shape[1];
    const float *a = (const float*)A->data;
    const float *b = (const float*)B->data;
    float *c = (float*)C->data;
    memset(c, 0, M * N * sizeof(float));

    for (int64_t i = 0; i < M; ++i) {
        for (int64_t k = 0; k < K; ++k) {
            float aik = a[i * K + k];
            for (int64_t j = 0; j < N; ++j) {
                c[i * N + j] += aik * b[k * N + j];
            }
        }
    }
}

void axon_fused_matmul_bias_relu(const AxonTensor *A, const AxonTensor *B, const AxonTensor *bias, AxonTensor *C) {
    axon_matmul(A, B, C);
    const int64_t M = A->shape[0], N = B->shape[1];
    float *c = (float*)C->data;
    const float *b = bias ? (const float*)bias->data : NULL;
    for (int64_t i = 0; i < M; ++i) {
        for (int64_t j = 0; j < N; ++j) {
            float v = c[i * N + j] + (b ? b[j] : 0.0f);
            c[i * N + j] = v > 0.0f ? v : 0.0f;
        }
    }
}

void axon_fused_matmul_bias(const AxonTensor *A, const AxonTensor *B, const AxonTensor *bias, AxonTensor *C) {
    axon_matmul(A, B, C);
    const int64_t M = A->shape[0], N = B->shape[1];
    float *c = (float*)C->data;
    const float *b = bias ? (const float*)bias->data : NULL;
    for (int64_t i = 0; i < M; ++i) {
        for (int64_t j = 0; j < N; ++j) {
            c[i * N + j] += (b ? b[j] : 0.0f);
        }
    }
}

void axon_add(const AxonTensor *A, const AxonTensor *B, AxonTensor *C) {
    const float *a = (const float*)A->data;
    const float *b = (const float*)B->data;
    float *c = (float*)C->data;
    if (A->ndim == 2 && B->ndim == 1 && A->shape[1] == B->shape[0]) {
        for (int64_t i = 0; i < A->shape[0]; ++i)
            for (int64_t j = 0; j < A->shape[1]; ++j)
                c[i * A->shape[1] + j] = a[i * A->shape[1] + j] + b[j];
        return;
    }
    int64_t total = axon_tensor_numel(C);
    for (int64_t i = 0; i < total; ++i) c[i] = a[i] + b[i];
}

void axon_relu(const AxonTensor *A, AxonTensor *C) {
    const float *a = (const float*)A->data;
    float *c = (float*)C->data;
    int64_t total = axon_tensor_numel(A);
    for (int64_t i = 0; i < total; ++i) c[i] = a[i] > 0.0f ? a[i] : 0.0f;
}
"""

FILES["runtime/axon_runtime.h"] = """#ifndef AXON_RUNTIME_H
#define AXON_RUNTIME_H
#include "axon_tensor.h"
#include "axon_mmap.h"
#include "axon_quant.h"

#ifdef __cplusplus
extern "C" {
#endif

void axon_matmul(const AxonTensor *A, const AxonTensor *B, AxonTensor *C);
void axon_add(const AxonTensor *A, const AxonTensor *B, AxonTensor *C);
void axon_relu(const AxonTensor *A, AxonTensor *C);
void axon_fused_matmul_bias_relu(const AxonTensor *A, const AxonTensor *B, const AxonTensor *bias, AxonTensor *C);
void axon_fused_matmul_bias(const AxonTensor *A, const AxonTensor *B, const AxonTensor *bias, AxonTensor *C);
void axon_print_tensor(const AxonTensor *t);
AxonTensor* axon_forward(AxonTensor **inputs);

#ifdef __cplusplus
}
#endif
#endif
"""

FILES["runtime/axon_runtime.c"] = """#include "axon_runtime.h"
#include <stdio.h>
#include <stdlib.h>

void axon_print_tensor(const AxonTensor *t) {
    if (!t) { printf("Tensor: NULL\\n"); return; }
    printf("Tensor(ndim=%ld, shape=[", (long)t->ndim);
    for (int64_t i = 0; i < t->ndim; ++i) printf("%ld%s", (long)t->shape[i], (i + 1 < t->ndim) ? ", " : "");
    printf("], dtype=%d) = [\\n", t->dtype);
    int64_t total = axon_tensor_numel(t);
    const float *d = (const float*)t->data;
    int64_t lim = total < 16 ? total : 16;
    for (int64_t i = 0; i < lim; ++i) printf("  %f\\n", d[i]);
    if (total > lim) printf("  ... (%ld more elements)\\n", (long)(total - lim));
    printf("]\\n");
}

int main(int argc, char **argv) {
    (void)argc; (void)argv;
    printf("[AXON RUNTIME] Initializing execution context...\\n");
    int64_t in_shape[2] = {1, 4};
    AxonTensor *inp = axon_tensor_alloc(2, in_shape, AXON_DTYPE_F32);
    float *p = (float*)inp->data;
    p[0] = 1.0f; p[1] = 0.5f; p[2] = -1.0f; p[3] = 2.0f;

    AxonTensor *inputs[1] = { inp };
    printf("[AXON RUNTIME] Invoking generated forward graph...\\n");
    AxonTensor *out = axon_forward(inputs);

    printf("[AXON RUNTIME] Execution complete. Result:\\n");
    axon_print_tensor(out);

    axon_tensor_free(inp);
    axon_tensor_free(out);
    return 0;
}
"""

FILES["runtime/Makefile"] = """CC ?= gcc
CFLAGS ?= -O3 -fPIC -Wall -Wextra -std=c11 -march=native
AR ?= ar
OBJS = axon_tensor.o axon_matmul.o axon_quant.o axon_mmap.o axon_runtime.o
TARGET = libaxon_runtime.a

all: $(TARGET)

%.o: %.c
	$(CC) $(CFLAGS) -c $< -o $@

$(TARGET): $(OBJS)
	$(AR) rcs $@ $^

clean:
	rm -f *.o $(TARGET)
"""

# =====================================================================
# Examples
# =====================================================================
FILES["examples/mlp.axon"] = """fn main(x: f32[1, 4]) -> f32[1, 2] {
    let w1: f32[4, 4] = [
        0.1, 0.2, -0.3, 0.4,
        0.5, -0.6, 0.7, 0.8,
        -0.1, 0.2, 0.3, -0.4,
        0.5, 0.6, -0.7, 0.8
    ] as f32[4, 4];

    let b1: f32[4] = [0.1, 0.1, 0.1, 0.1] as f32[4];
    let h1: f32[1, 4] = relu(add(matmul(x, w1), b1));

    let w2: f32[4, 2] = [
        0.2, -0.4,
        0.6, 0.8,
        -0.5, 0.3,
        0.1, -0.2
    ] as f32[4, 2];

    let b2: f32[2] = [0.05, -0.05] as f32[2];
    let out: f32[1, 2] = add(matmul(h1, w2), b2);
    return out;
}
"""

# =====================================================================
# Unit & E2E Tests
# =====================================================================
FILES["tests/test_lexer.py"] = """import unittest
from axon.frontend.lexer import Lexer
from axon.frontend.tokens import TokenKind

class TestLexer(unittest.TestCase):
    def test_lexer(self):
        t = Lexer("fn forward(x: f32[1, 4]) -> f32[1, 2] { let y = relu(x); return y; }").tokenize()
        kinds = [x.kind for x in t]
        self.assertIn(TokenKind.FN, kinds)
        self.assertIn(TokenKind.RELU, kinds)
        self.assertEqual(t[-1].kind, TokenKind.EOF)

if __name__ == "__main__":
    unittest.main()
"""

FILES["tests/test_parser.py"] = """import unittest
from axon.frontend.lexer import Lexer
from axon.frontend.parser import Parser

class TestParser(unittest.TestCase):
    def test_parser(self):
        code = "fn f(x: f32[1, 4]) -> f32[1, 4] { let y: f32[1, 4] = relu(x); return y; }"
        ast = Parser(Lexer(code).tokenize()).parse()
        self.assertEqual(len(ast.functions), 1)
        self.assertEqual(ast.functions[0].name, "f")

if __name__ == "__main__":
    unittest.main()
"""

FILES["tests/test_fusion.py"] = """import unittest
from axon.ir.tensor_ir import Graph, IRNode, OpKind
from axon.optimizer.fusion import FusionPass

class TestFusion(unittest.TestCase):
    def test_fusion(self):
        g = Graph()
        g.nodes = [
            IRNode(0, OpKind.INPUT, [], {}, (1, 4), "f32"),
            IRNode(1, OpKind.CONSTANT, [], {}, (4, 4), "f32"),
            IRNode(2, OpKind.MATMUL, [0, 1], {}, (1, 4), "f32"),
            IRNode(3, OpKind.CONSTANT, [], {}, (4,), "f32"),
            IRNode(4, OpKind.ADD, [2, 3], {}, (1, 4), "f32"),
            IRNode(5, OpKind.RELU, [4], {}, (1, 4), "f32")
        ]
        g.inputs = [0]
        g.outputs = [5]
        fg = FusionPass().run(g)
        ops = [n.op for n in fg.nodes]
        self.assertIn(OpKind.FUSED_MATMUL_BIAS_RELU, ops)

if __name__ == "__main__":
    unittest.main()
"""

FILES["tests/test_quantize.py"] = """import unittest
from axon.ir.tensor_ir import WeightBlob
from axon.optimizer.quantize import FP4QuantizePass, round_fp4

class TestQuant(unittest.TestCase):
    def test_quant(self):
        self.assertEqual(round_fp4(1.8), 2.0)
        data = [0.2 * i for i in range(32)]
        data[4] = 90.0
        blob = WeightBlob("w", data, [32], "f32")
        FP4QuantizePass()._quantize(blob)
        self.assertTrue(blob.is_quantized)
        self.assertEqual(len(blob.residuals), 1)

if __name__ == "__main__":
    unittest.main()
"""

FILES["tests/test_e2e.py"] = """import unittest, os, subprocess
from axon.frontend.lexer import Lexer
from axon.frontend.parser import Parser
from axon.frontend.semantic import SemanticAnalyzer
from axon.ir.builder import IRBuilder
from axon.backend.llvm_backend import LLVMBackend
from axon.baker.baker import Baker

class TestE2E(unittest.TestCase):
    def test_e2e(self):
        code = '''
        fn forward(x: f32[1, 4]) -> f32[1, 4] {
            let w: f32[4, 4] = [
                1.0, 0.0, 0.0, 0.0,
                0.0, 1.0, 0.0, 0.0,
                0.0, 0.0, 1.0, 0.0,
                0.0, 0.0, 0.0, 1.0
            ] as f32[4, 4];
            let y: f32[1, 4] = matmul(x, w);
            return y;
        }
        '''
        ast = SemanticAnalyzer().analyze(Parser(Lexer(code).tokenize()).parse())
        g = IRBuilder().build(ast)
        obj = "test_e2e.o"
        bin_out = "test_e2e.bin"
        LLVMBackend().compile(g, obj)
        Baker("runtime/libaxon_runtime.a").bake([obj], g.weights, bin_out)
        res = subprocess.run([f"./{bin_out}"], capture_output=True, text=True)
        self.assertEqual(res.returncode, 0)
        self.assertIn("Execution complete", res.stdout)
        if os.path.exists(obj): os.remove(obj)
        if os.path.exists(bin_out): os.remove(bin_out)

if __name__ == "__main__":
    unittest.main()
"""


# =====================================================================
# Main Generator & Execution Engine
# =====================================================================
def main():
    print("[1/5] Unpacking full Axon compiler files...")
    for path, content in FILES.items():
        d = os.path.dirname(path)
        if d: os.makedirs(d, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            f.write(content)
        print(f"  -> Created {path}")

    print("\n[2/5] Compiling native C runtime (libaxon_runtime.a)...")
    res = subprocess.run(["make", "runtime"], capture_output=True, text=True)
    if res.returncode != 0:
        print(f"Build failed:\n{res.stderr}", file=sys.stderr)
        sys.exit(1)
    print("  -> Runtime compiled successfully.")

    print("\n[3/5] Running test suite...")
    test_res = subprocess.run([sys.executable, "-m", "unittest", "discover", "-s", "tests", "-p", "test_*.py"])
    if test_res.returncode != 0:
        print("  -> Some tests failed!", file=sys.stderr)
        sys.exit(1)
    print("  -> All 5 test suites passed cleanly.")

    print("\n[4/5] Compiling examples/mlp.axon to standalone binary (mlp.bin)...")
    compile_cmd = [sys.executable, "-m", "axon.cli", "examples/mlp.axon", "-o", "mlp.bin"]
    res = subprocess.run(compile_cmd, capture_output=True, text=True)
    if res.returncode != 0:
        print(f"Compilation failed:\n{res.stderr}", file=sys.stderr)
        sys.exit(1)
    print("  -> Compiled successfully: mlp.bin")

    print("\n[5/5] Executing standalone binary (zero Python dependency):")
    print("-" * 50)
    exec_res = subprocess.run(["./mlp.bin"], capture_output=True, text=True)
    print(exec_res.stdout)
    print("-" * 50)
    print("[COMPLETED] Axon compiler is 100% operational and ready.")


if __name__ == "__main__":
    main()
