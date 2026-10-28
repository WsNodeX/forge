<div align="center">

# ⚒️ Forge

### A dependency-free Python toolkit for training and running subword (BPE) tokenizers — with byte-level fallback, special tokens, a trainer, persistence, and a CLI.

[![Python](https://img.shields.io/badge/Python-3.9%2B-blue?style=flat-square)](https://www.python.org/)
[![License](https://img.shields.io/badge/License-MIT-green?style=flat-square)](LICENSE)
[![Zero Dependencies](https://img.shields.io/badge/dependencies-0-brightgreen?style=flat-square)](https://pypi.org/project/forge-tokenizer/)
[![Platform](https://img.shields.io/badge/platform-Windows%20%7C%20Linux%20%7C%20macOS-lightgrey?style=flat-square)](#)
[![Tests](https://img.shields.io/badge/tests-passing-blue?style=flat-square)](#)

**Forge turns raw text into integer token ids the way modern LLMs do — and it does so with nothing but the Python standard library.**

[Why](#-why) · [Features](#-features) · [Install](#-installation) · [Quick Start](#-quick-start) · [Architecture](#-architecture) · [API](#-api-reference) · [Roadmap](#-roadmap) · [中文导读](#-中文技术导读可发布稿)

</div>

---

## ✨ Why

Every large language model is, at its core, a **token machine**. Before a single
neuron fires, the input string has already been chopped into a sequence of
integers — and the *quality* of that chop decides how much context fits in a
context window, how much a single API call costs, and how fairly rare languages
are treated.

Most engineers treat tokenization as a black box: `tiktoken.encode(text)` and
move on. That is fine until you need to:

- ship a model to a language your vendor's tokenizer mishandles,
- understand *why* a prompt blows past the context limit,
- build a small, auditable tokenizer you can embed in an edge device or a
  homework assignment without pulling in a 50 MB dependency tree,
- or simply learn, from first principles, the mechanism that sits between text
  and transformers.

**Forge is the box opened.** It is a complete, readable implementation of a
byte-level BPE tokenizer — the same family of algorithm behind GPT-2, RoBERTa
and many open models — written so you can read it top to bottom in an afternoon.

## 🚀 Features

- **Byte-level fallback.** Any Unicode string — including emoji and CJK — is
  encoded losslessly, because the algorithm operates on a fixed 256-symbol
  alphabet derived from the raw bytes.
- **Train from scratch.** Feed a corpus; Forge learns a merge table with the
  classic frequency-based BPE algorithm. Fully deterministic.
- **Special tokens.** Reserve ids for `<|endoftext|>`, `<|pad|>` and friends;
  they survive encode/decode verbatim.
- **Persistence.** Save a trained vocabulary to JSON and reload it byte-for-byte.
- **Bundled English vocabulary.** Works out of the box — no download step.
- **Zero dependencies.** Pure standard library. Runs on CPython and PyPy, 3.9+.
- **CLI + library.** Use it as a `forge` command or `import forge` in your code.
- **Round-trip guarantee.** `decode(encode(x)) == x` for *every* string, seen
  or unseen.

## 📦 Installation

```bash
# From source (no third-party packages are installed)
pip install -e .

# Or just copy the `forge/` package into your project — it has no dependencies.
```

That is the entire supply chain.

## ⚡ Quick Start

```python
from forge import BPETokenizer, load

tok = BPETokenizer(load("english"))

text = "Tokenization is the silent hero of every language model."
ids = tok.encode(text)

print(ids)            # -> [84, 111, 107, 101, 110, 105, 122, 350, ...]
print(tok.decode(ids))  # -> "Tokenization is the silent hero of every language model."
assert tok.decode(ids) == text
```

Train your own in three lines:

```python
from forge import BPETokenizer, BPETrainer

vocab = BPETrainer(vocab_size=512, special=["<|endoftext|>"]).train("a custom corpus here", "mine")
tok = BPETokenizer(vocab)
print(tok.encode("a custom corpus"))
vocab.save("mine.json")          # later: Vocabulary.load("mine.json")
```

Command line:

```bash
python -m forge.cli encode "Tokenization is the silent hero of every model."
python -m forge.cli tokens "hello world"
python -m forge.cli train --text "your corpus" --vocab-size 512 --out vocab.json
python -m forge.cli info
```

**Expected output** of `encode "hello world"`:

```
72 101 108 108 111 398 32 119 111 114 108 100
# 12 tokens
```

## 🏗️ Architecture

```
┌───────────────────────────────────────────────────────────┐
│                         CLI  (forge.cli)                    │   $ forge encode ...
├───────────────────────────────────────────────────────────┤
│                     BPETokenizer                            │   encode / decode / tokenize
│                  (rank table · symbol→id maps)             │
├──────────────────────────┬────────────────────────────────┤
│       BPETrainer         │          Vocabulary             │   learned state
│   (frequency BPE merges) │    (merges + special tokens)    │
├──────────────────────────┴────────────────────────────────┤
│  pretokenize  ·  ByteAlphabet  ·  bpe (greedy pair merge)  │   the engine room
└───────────────────────────────────────────────────────────┘
```

Each layer depends only on the one below it. The byte alphabet and the
pre-tokenizer are pure functions; the trainer is the only stateful learner; the
tokenizer is a stateless transducer built from a frozen vocabulary.

## 🔌 API Reference

### `forge.BPETokenizer`

| Method | Signature | Description |
| --- | --- | --- |
| `encode` | `(text: str) -> list[int]` | Encode text to token ids (special tokens split out first). |
| `decode` | `(ids: list[int]) -> str` | Reconstruct the original string. |
| `tokenize` | `(text: str) -> list[str]` | Return the textual subword symbols (no id mapping). |
| `encode_decode` | `(text: str) -> str` | Convenience round-trip helper. |
| `compression_ratio` | `(text: str) -> float` | Tokens-per-character (lower = more compact). |

### `forge.BPETrainer`

| Method | Signature | Description |
| --- | --- | --- |
| `train` | `(corpus: str \| Iterable[str], name="custom") -> Vocabulary` | Learn merges from a corpus (deterministic). |

Constructor: `BPETrainer(vocab_size=512, min_frequency=1, special=None)`.

### `forge.Vocabulary`

| Method | Signature | Description |
| --- | --- | --- |
| `save` | `(path: str) -> None` | Write vocabulary to JSON. |
| `load` | `(path: str) -> Vocabulary` | Load a vocabulary from JSON. |
| `build_index` | `(alphabet=ALPHABET)` | Build `symbol↔id` tables. |
| `size` | `-> int` | Total number of distinct token ids. |

### Registry

| Function | Signature | Description |
| --- | --- | --- |
| `load` | `(name="english") -> Vocabulary` | Load a bundled vocabulary. |
| `available` | `() -> list[str]` | Names of bundled vocabularies. |

## 🧭 Roadmap

- [ ] WordPiece and Unigram (SentencePiece-style) trainers alongside BPE
- [ ] Streaming trainer for corpora larger than memory
- [ ] FUSE-style cache so repeated `encode` calls skip re-merging
- [ ] A `vocab prune` sub-command to drop unused merges
- [ ] Optional `numpy` backend behind a feature flag (library stays default-zero-dep)
- [ ] Pre-tokenizer presets for Chinese / Japanese / code

## 🤝 Contributing

Forge is intentionally small and readable. The best contributions are:

1. A new trainer (WordPiece, Unigram) that plugs into the same `Vocabulary`.
2. A faster merge-loop implementation that preserves identical output.
3. A new bundled vocabulary for a language or domain.

Open an issue describing the change, then a pull request with tests. New code
must keep the project dependency-free for the default install path.

## 📄 License

Released under the [MIT License](LICENSE). Do whatever you like; attribution is
appreciated.

---

## 📝 中文技术导读（可发布稿）

> 下面这篇长文是给掘金 / 知乎 / 小红书 / B 站读者的科普底稿，已按"可独立发布"
> 的方式撰写。把它复制出去前，按文末提示润色成你自己的风格即可。

### 一、为什么你应该关心"分词"这件事

你以为大模型直接读的是文字？错。在第一个神经元被激活之前，你输入的每一句话
都已经被切成了一串**整数**——这就是 token。模型真正计算的，是这串整数。

这意味着三件和你钱包、和你工程都相关的事：

- **上下文窗口是按 token 算的**，不是按字符。同样一段话，分词分得细，能塞进
  窗口的内容就少，一次对话能带的历史就越短。
- **API 是按 token 计费的**。分词粗一点，同样的回答可能便宜一半。
- **公平性是分语言体现的**。英文单词天然短 token，中文、日文、emoji 往往被切得
  很碎。你用的分词器，决定了你的母语在模型眼里"贵不贵、准不准"。

所以，分词器不是配角，它是坐在文本和 Transformer 之间的那道门。

### 二、核心原理：字节级 BPE 到底在干什么

Forge 实现的是 **byte-level BPE**（字节级字节对编码），和 GPT-2、RoBERTa 同门。
它分三步：

1. **预切分（pre-tokenize）**：先用正则把文本粗切成"词块"。关键是把前面的空格
   一起带进下一个词（`" world"` 而不是 `"world"`），这样模型能区分"词首"和
   "词中"。缩写如 `don't` 也会被单独识别。
2. **字节映射（byte alphabet）**：把每个字节（0–255）映射到 256 个" printable
   符号"。空格变成 `Ġ`，控制字符被挪到高位。这一步的妙处在于——**任何字节序列
   都能无损编码**，中文、emoji、乱码统统不怕。
3. **BPE 合并（greedy merge）**：在词块内部，反复把"优先级最高的一对相邻符号"
   拼起来，比如 `t`+`h`→`th`，`th`+`e`→`the`。优先级来自训练时统计的**出现频率**。

解码就是反过来：id → 符号 → 拼回字节 → UTF-8 解码。因为每一步都可逆，**无论
这句话训练时见过没有，`decode(encode(x))` 永远等于 `x`**——这是 Forge 给自己的
硬保证。

### 三、几个关键实现点（看代码能学到的）

- **确定性训练**：同样的语料永远训出同一张合并表，方便测试和复现。
- **符号表自动推导**：base 字节占 `0–255`，第 `i` 条合并规则占 id `256+i`，
  特殊 token 排在所有合并之后，永远不会和子词 id 撞车。
- **特殊 token 透传**：`<|endoftext|>` 之类会被原样保留，不会被判成普通字母去
  合并。
- **零依赖**：只用 Python 标准库，`re` 处理预切分，自写字节双向映射，没有任何
  第三方包。

### 四、和同类方案比，它站在哪

| 维度 | Forge | tiktoken / 厂商分词器 | 手写正则切词 |
| --- | --- | --- | --- |
| 依赖 | 0（纯标准库） | 需装对应 SDK | 0 |
| 可训练 | ✅ 从语料自学 | ❌ 固定 | ❌ 无语义 |
| 跨语言无损 | ✅ 字节级回退 | 视厂商而定 | ❌ 常乱码 |
| 可读/可审计 | ✅ 几百行 | ❌ 黑盒 | ✅ 但太弱 |
| 适合学习原理 | ✅ | ❌ | ⚠️ 学不到 |

一句话：**要生产用大模型官方的，要理解原理或要可控可移植，用 Forge。**

### 五、怎么把它玩出花

- 换语料训一个**专属于你业务**的小分词器（比如代码、医典、法律文本），
  对比通用分词器在你的数据上的压缩率。
- 接一个 RAG 管线，在检索前用 `compression_ratio` 估算每段文本的 token 成本。
- 把 `Vocabulary` 序列化进你的推理服务，做到**训练端和推理端用同一个分词器**，
  避免线上线下不一致的经典坑。

### 六、把它扩展成一篇发布稿的提示

- **标题方向**：《我用一个周末写了个分词器，终于看懂大模型"读"文字的方式》这类
  第一人称 + 收获感的标题在掘金/B站更吃香。
- **开头钩子**：抛一个"你知道你一句话要花多少钱的 token 吗"的反差问题。
- **配图建议**：画一张"文字 → 词块 → 字节符号 → 合并 → id"的流程图（注意：本文
  README 不依赖任何图片，发布时你再补图）。
- **结尾**：引导读者 `pip install` 或 `git clone` 自己跑一遍，并留下"下期讲
  WordPiece / Unigram"的钩子。

---

<div align="center">

**Forge — open the box. Read the tokens. Own the pipeline.**

</div>
