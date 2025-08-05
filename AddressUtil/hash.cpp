#include "AddressUtil.h"
#include "CryptoUtil.h"

#include <stdio.h>
#include <string.h>

static unsigned int endian(unsigned int x)
{
	return (x << 24) | ((x << 8) & 0x00ff0000) | ((x >> 8) & 0x0000ff00) | (x >> 24);
}

bool Address::verifyAddress(std::string address)
{
	// Check length
	if(address.length() > 34) {
		false;
	}

	// Check encoding
	if(!Base58::isBase58(address)) {
		return false;
	}

	std::string noPrefix = address.substr(1);

	secp256k1::uint256 value = Base58::toBigInt(noPrefix);
	unsigned int words[6];
	unsigned int hash[5];
	unsigned int checksum;

	value.exportWords(words, 6, secp256k1::uint256::BigEndian);
	memcpy(hash, words, sizeof(unsigned int) * 5);
	checksum = words[5];

	return crypto::checksum(hash) == checksum;
}

std::string Address::fromPublicKey(const secp256k1::ecpoint &p, bool compressed)
{
	unsigned int xWords[8] = { 0 };
	unsigned int yWords[8] = { 0 };

	p.x.exportWords(xWords, 8, secp256k1::uint256::BigEndian);
	p.y.exportWords(yWords, 8, secp256k1::uint256::BigEndian);

	unsigned int digest[5];

	if(compressed) {
		Hash::hashPublicKeyCompressed(xWords, yWords, digest);
	} else {
		Hash::hashPublicKey(xWords, yWords, digest);
	}

	unsigned int checksum = crypto::checksum(digest);

	unsigned int addressWords[8] = { 0 };
	for(int i = 0; i < 5; i++) {
		addressWords[2 + i] = digest[i];
	}
	addressWords[7] = checksum;

	secp256k1::uint256 addressBigInt(addressWords, secp256k1::uint256::BigEndian);

	return "1" + Base58::toBase58(addressBigInt);
}

void Hash::hashPublicKey(const secp256k1::ecpoint &p, unsigned int *digest)
{
	unsigned int xWords[8];
	unsigned int yWords[8];

	p.x.exportWords(xWords, 8, secp256k1::uint256::BigEndian);
	p.y.exportWords(yWords, 8, secp256k1::uint256::BigEndian);

	hashPublicKey(xWords, yWords, digest);
}


void Hash::hashPublicKeyCompressed(const secp256k1::ecpoint &p, unsigned int *digest)
{
	unsigned int xWords[8];
	unsigned int yWords[8];

	p.x.exportWords(xWords, 8, secp256k1::uint256::BigEndian);
	p.y.exportWords(yWords, 8, secp256k1::uint256::BigEndian);

	hashPublicKeyCompressed(xWords, yWords, digest);
}

void Hash::hashPublicKey(const unsigned int *x, const unsigned int *y, unsigned int *digest)
{
	unsigned char pub[65];

	// Build uncompressed public key bytes: 0x04 || x || y
	pub[0] = 0x04;
	for(int i = 0; i < 8; ++i) {
		pub[1 + i * 4] = (unsigned char)(x[i] >> 24);
		pub[2 + i * 4] = (unsigned char)(x[i] >> 16);
		pub[3 + i * 4] = (unsigned char)(x[i] >> 8);
		pub[4 + i * 4] = (unsigned char)(x[i]);
	}
	for(int i = 0; i < 8; ++i) {
		pub[33 + i * 4] = (unsigned char)(y[i] >> 24);
		pub[34 + i * 4] = (unsigned char)(y[i] >> 16);
		pub[35 + i * 4] = (unsigned char)(y[i] >> 8);
		pub[36 + i * 4] = (unsigned char)(y[i]);
	}

	unsigned int sha256Digest[8];
	unsigned int msg[16];
	unsigned char block[64];

	// First 64 bytes
	memcpy(block, pub, 64);
	for(int i = 0; i < 16; ++i) {
		msg[i] = ((unsigned int)block[i * 4] << 24) |
			 ((unsigned int)block[i * 4 + 1] << 16) |
			 ((unsigned int)block[i * 4 + 2] << 8) |
			 (unsigned int)block[i * 4 + 3];
	}

	crypto::sha256Init(sha256Digest);
	crypto::sha256(msg, sha256Digest);

	// Second block: remaining byte, padding and length
	memset(block, 0, sizeof(block));
	block[0] = pub[64];
	block[1] = 0x80;
	unsigned long long bitLen = 65ULL * 8ULL;
	for(int i = 0; i < 8; ++i) {
		block[56 + i] = (unsigned char)(bitLen >> (56 - 8 * i));
	}
	for(int i = 0; i < 16; ++i) {
		msg[i] = ((unsigned int)block[i * 4] << 24) |
			 ((unsigned int)block[i * 4 + 1] << 16) |
			 ((unsigned int)block[i * 4 + 2] << 8) |
			 (unsigned int)block[i * 4 + 3];
	}
	crypto::sha256(msg, sha256Digest);

	// Convert SHA-256 digest to bytes
	unsigned char shaBytes[32];
	for(int i = 0; i < 8; ++i) {
		shaBytes[i * 4] = (unsigned char)(sha256Digest[i] >> 24);
		shaBytes[i * 4 + 1] = (unsigned char)(sha256Digest[i] >> 16);
		shaBytes[i * 4 + 2] = (unsigned char)(sha256Digest[i] >> 8);
		shaBytes[i * 4 + 3] = (unsigned char)(sha256Digest[i]);
	}

	// Prepare RIPEMD160 block
	memset(block, 0, sizeof(block));
	memcpy(block, shaBytes, 32);
	block[32] = 0x80;
	unsigned long long rmdLen = 32ULL * 8ULL;
	block[56] = (unsigned char)(rmdLen & 0xff);
	block[57] = (unsigned char)((rmdLen >> 8) & 0xff);
	block[58] = (unsigned char)((rmdLen >> 16) & 0xff);
	block[59] = (unsigned char)((rmdLen >> 24) & 0xff);
	for(int i = 0; i < 16; ++i) {
		msg[i] = ((unsigned int)block[i * 4]) |
			 ((unsigned int)block[i * 4 + 1] << 8) |
			 ((unsigned int)block[i * 4 + 2] << 16) |
			 ((unsigned int)block[i * 4 + 3] << 24);
	}

	unsigned int rmdDigest[5];
	crypto::ripemd160(msg, rmdDigest);

	for(int i = 0; i < 5; ++i) {
		digest[i] = rmdDigest[i];
	}
}



void Hash::hashPublicKeyCompressed(const unsigned int *x, const unsigned int *y, unsigned int *digest)
{
	unsigned char pub[33];

	pub[0] = (y[7] & 0x01) ? 0x03 : 0x02;
	for(int i = 0; i < 8; ++i) {
		pub[1 + i * 4] = (unsigned char)(x[i] >> 24);
		pub[2 + i * 4] = (unsigned char)(x[i] >> 16);
		pub[3 + i * 4] = (unsigned char)(x[i] >> 8);
		pub[4 + i * 4] = (unsigned char)(x[i]);
	}

	unsigned int sha256Digest[8];
	unsigned int msg[16];
	unsigned char block[64] = {0};

	memcpy(block, pub, 33);
	block[33] = 0x80;
	unsigned long long bitLen = 33ULL * 8ULL;
	for(int i = 0; i < 8; ++i) {
		block[56 + i] = (unsigned char)(bitLen >> (56 - 8 * i));
	}
	for(int i = 0; i < 16; ++i) {
		msg[i] = ((unsigned int)block[i * 4] << 24) |
			 ((unsigned int)block[i * 4 + 1] << 16) |
			 ((unsigned int)block[i * 4 + 2] << 8) |
			 (unsigned int)block[i * 4 + 3];
	}

	crypto::sha256Init(sha256Digest);
	crypto::sha256(msg, sha256Digest);

	unsigned char shaBytes[32];
	for(int i = 0; i < 8; ++i) {
		shaBytes[i * 4] = (unsigned char)(sha256Digest[i] >> 24);
		shaBytes[i * 4 + 1] = (unsigned char)(sha256Digest[i] >> 16);
		shaBytes[i * 4 + 2] = (unsigned char)(sha256Digest[i] >> 8);
		shaBytes[i * 4 + 3] = (unsigned char)(sha256Digest[i]);
	}

	memset(block, 0, sizeof(block));
	memcpy(block, shaBytes, 32);
	block[32] = 0x80;
	unsigned long long rmdLen = 32ULL * 8ULL;
	block[56] = (unsigned char)(rmdLen & 0xff);
	block[57] = (unsigned char)((rmdLen >> 8) & 0xff);
	block[58] = (unsigned char)((rmdLen >> 16) & 0xff);
	block[59] = (unsigned char)((rmdLen >> 24) & 0xff);
	for(int i = 0; i < 16; ++i) {
		msg[i] = ((unsigned int)block[i * 4]) |
			 ((unsigned int)block[i * 4 + 1] << 8) |
			 ((unsigned int)block[i * 4 + 2] << 16) |
			 ((unsigned int)block[i * 4 + 3] << 24);
	}

	unsigned int rmdDigest[5];
	crypto::ripemd160(msg, rmdDigest);

	for(int i = 0; i < 5; ++i) {
		digest[i] = rmdDigest[i];
	}
}

void Hash::hashPublicKeyCompressed(const unsigned char *key, unsigned int *digest)
{
	unsigned int msg[16] = {0};
	unsigned int sha256Digest[8];

	// Copy 33-byte compressed key into message buffer for SHA-256
	for(int i = 0; i < 8; ++i) {
		msg[i] = ((unsigned int)key[i * 4] << 24) |
			 ((unsigned int)key[i * 4 + 1] << 16) |
			 ((unsigned int)key[i * 4 + 2] << 8) |
			 (unsigned int)key[i * 4 + 3];
	}

	// Last byte followed by padding and message length
	msg[8] = ((unsigned int)key[32] << 24) | 0x00800000;
	msg[15] = 33 * 8;

	crypto::sha256Init(sha256Digest);
	crypto::sha256(msg, sha256Digest);

	// Prepare RIPEMD160 input
	for(int i = 0; i < 16; ++i) {
		msg[i] = 0;
	}

	for(int i = 0; i < 8; ++i) {
		msg[i] = endian(sha256Digest[i]);
	}

	msg[8] = 0x00000080;
	msg[14] = 256;
	msg[15] = 0;

	crypto::ripemd160(msg, digest);
}
