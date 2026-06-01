# CrackMt19937

***

# Libbitcoin MT19937 Vulnerability (CVE-2023-39910) GPU Reproducer

## ⚠️ Disclaimer
**This project is strictly for educational and research purposes.** The code provided here is designed to demonstrate the catastrophic security implications of using non-cryptographically secure pseudo-random number generators (PRNGs) for cryptographic key generation. Do not use this tool to attack or steal funds from unowned wallets. The authors assume no liability for any malicious use of this software.

## 📖 Background
In 2023, a critical vulnerability registered as **CVE-2023-39910** (also known as the "MilkSad" vulnerability) was disclosed affecting the Libbitcoin Explorer (bx) 3.x versions. The software utilized the Mersenne Twister-32 (MT19937-32) pseudo-random number generator (PRNG) to generate wallet entropy, seeding it solely with the 32-bit system time.

Because the seed space is restricted to $2^{32}$ (approximately 4.29 billion) possible values, the randomness is entirely predictable. This fundamental flaw allowed attackers to brute-force the seed space, reconstruct private keys, and steal over 120,000 BTC—most notably involving the massive $15 billion theft from the LuBian mining pool.

## ⚙️ Technical Details
This repository reproduces the exact deterministic process used by the attackers to recover the private keys:

1. **Weak Seed Initialization:** The MT19937 PRNG is initialized with a low-entropy 32-bit seed ranging from `0` to `2^32 - 1`.
2. **Byte Extraction:** In each round, MT19937 outputs a 32-bit integer. However, only the **highest 8 bits** (1 byte) are selected.
3. **The OFFSET Feature:** To generate a 32-byte (256-bit) private key seed, 32 consecutive rounds are required. The process introduces a sliding window mechanism called `OFFSET`. The 32 bytes are extracted from round `(32 * OFFSET)` to `(32 * OFFSET + 31)`. The offset value can reach up to 3232.
4. **BIP39 & BIP32 Derivation:** The extracted 32-byte entropy is converted into a 24-word mnemonic and subsequently a 64-byte binary seed. Using the BIP32 derivation algorithm and following the specific path `m/49'/0'/0'/0/0`, child public-private key pairs are generated.
5. **Address Matching:** The derived public key is formatted into a `P2WPKH-nested-in-P2SH` wallet address (starting with the prefix `3`). If the generated address matches a target address on the blockchain, the private key is successfully recovered.

## 🚀 Why GPU?
A standard CPU testing each seed sequentially would take a considerable amount of time to exhaust the 4.29 billion possibilities along with thousands of `OFFSET` variations. However, since the MT19937 state transitions and ECDSA derivations are purely deterministic mathematical operations, this task is highly parallelizable.

By leveraging the massive parallel compute cores of a GPU (CUDA/OpenCL), this tool can simultaneously calculate thousands of PRNG states and perform elliptic curve derivations, reducing the brute-force time from days to mere minutes or hours.

## 🛠️ Build and Usage

### Prerequisites
* CUDA Toolkit (e.g., v11.0+) / OpenCL SDK
* C++ Compiler (GCC/Clang or MSVC)
* [Any other specific libraries you used, e.g., secp256k1, OpenSSL]

### Compilation
```bash
# Example compilation commands (Update according to your actual build system)
git clone https://github.com/brainzhang-bitcoin/CrackMt19937
cd CrackMt19937
make
```

``

## 📚 References
* [美国政府是如何没收大量比特币的](https://brainz.fun/blog/2026/06/01/mei-guo-zheng-fu-shi-ru-he-mei-shou-da-liang-bi-te-bi-de/)
* [CVE-2023-39910: Libbitcoin Explorer PRNG Vulnerability](https://nvd.nist.gov/vuln/detail/CVE-2023-39910)
* [MilkSad Vulnerability Disclosure](https://milksad.info/)
* Elliptic Report: $15 billion seized by US originates from Iran/China bitcoin miner "theft"
* Safeheron Lab: Retrospective on the Largest PRNG Vulnerability in Crypto History

***
*Developed for educational security research.*

