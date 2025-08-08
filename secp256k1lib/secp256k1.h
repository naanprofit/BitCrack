#ifndef _HOST_SECP256K1_H
#define _HOST_SECP256K1_H

#include<stdio.h>
#include<stdint.h>
#include<string.h>
#include<string>
#include<vector>

namespace secp256k1 {

	class uint256 {

	public:
		static const int BigEndian = 1;
		static const int LittleEndian = 2;

        uint32_t v[8] = { 0 };

		uint256()
		{
			memset(this->v, 0, sizeof(this->v));
		}

		uint256(const std::string &s)
		{
			std::string t = s;

			// 0x prefix
			if(t.length() >= 2 && (t[0] == '0' && t[1] == 'x' || t[1] == 'X')) {
				t = t.substr(2);
			}

			// 'h' suffix
			if(t.length() >= 1 && t[t.length() - 1] == 'h') {
				t = t.substr(0, t.length() - 1);
			}
			
			if(t.length() == 0) {
				throw std::string("Incorrect hex formatting");
			}

			// Verify only valid hex characters
			for(int i = 0; i < (int)t.length(); i++) {
				if(!((t[i] >= 'a' && t[i] <= 'f') || (t[i] >= 'A' && t[i] <= 'F') || (t[i] >= '0' && t[i] <= '9'))) {
					throw std::string("Incorrect hex formatting");
				}
			}

			// Ensure the value is 64 hex digits. If it is longer, take the least-significant 64 hex digits.
			// If shorter, pad with 0's.
			if(t.length() > 64) {
				t = t.substr(t.length() - 64);
			} else if(t.length() < 64) {
				t = std::string(64 - t.length(), '0') + t;
			}

			int len = (int)t.length();

			memset(this->v, 0, sizeof(uint32_t) * 8);

			int j = 0;
			for(int i = len - 8; i >= 0; i-= 8) {
				std::string sub = t.substr(i, 8);
				uint32_t val;
				if(sscanf(sub.c_str(), "%x", &val) != 1) {
					throw std::string("Incorrect hex formatting");
				}
				this->v[j] = val;
				j++;
			}
		}

		uint256(unsigned int x)
		{
			memset(this->v, 0, sizeof(this->v));
			this->v[0] = x;
		}

		uint256(uint64_t x)
		{
			memset(this->v, 0, sizeof(this->v));
			this->v[0] = (unsigned int)x;
			this->v[1] = (unsigned int)(x >> 32);
		}

		uint256(int x)
		{
			memset(this->v, 0, sizeof(this->v));
			this->v[0] = (unsigned int)x;
		}

		uint256(const unsigned int x[8], int endian = LittleEndian)
		{
			if(endian == LittleEndian) {
				for(int i = 0; i < 8; i++) {
					this->v[i] = x[i];
				}
			} else {
				for(int i = 0; i < 8; i++) {
					this->v[i] = x[7 - i];
				}
			}
		}

		bool operator==(const uint256 &x) const
		{
			for(int i = 0; i < 8; i++) {
				if(this->v[i] != x.v[i]) {
					return false;
				}
			}

			return true;
		}

        uint256 operator+(const uint256 &x) const
        {
            return add(x);
        }

        uint256 operator+(uint32_t x) const
        {
            return add(x);
        }

        uint256 operator*(uint32_t x) const
        {
            return mul(x);
        }

        uint256 operator*(const uint256 &x) const
        {
            return mul(x);
        }

        uint256 operator*(uint64_t x) const
        {
            return mul(x);
        }

        uint256 operator-(const uint256 &x) const
        {
            return sub(x);
        }

                void exportWords(unsigned int *buf, int len, int endian = LittleEndian) const
                {
                        if(endian == LittleEndian) {
                                for(int i = 0; i < len; i++) {
                                        buf[i] = v[i];
                                }
                        } else {
                                for(int i = 0; i < len; i++) {
                                        buf[len - i - 1] = v[i];
                                }
                        }
                }

                static uint256 importBigEndian(const unsigned int *src, int words)
                {
                        uint256 val;
                        for(int i = 0; i < 8; i++) {
                                val.v[i] = 0;
                        }
                        for(int i = 0; i < words && i < 8; i++) {
                                unsigned int w = src[words - 1 - i];
                                w = (w << 24) | ((w << 8) & 0x00ff0000U) |
                                    ((w >> 8) & 0x0000ff00U) | (w >> 24);
                                val.v[i] = w;
                        }
                        return val;
                }

		uint256 mul(const uint256 &val) const;

		uint256 mul(int val) const;

        uint256 mul(uint32_t val) const;

        uint256 mul(uint64_t val) const;

		uint256 add(int val) const;

		uint256 add(unsigned int val) const;

		uint256 add(uint64_t val) const;

		uint256 sub(int val) const;

        uint256 sub(const uint256 &val) const;

		uint256 add(const uint256 &val) const;

		uint256 div(uint32_t val) const;

		uint256 mod(uint32_t val) const;

		unsigned int toInt32() const
		{
			return this->v[0];
		}

		bool isZero() const
		{
			for(int i = 0; i < 8; i++) {
				if(this->v[i] != 0) {
					return false;
				}
			}

			return true;
		}

		int cmp(const uint256 &val) const
		{
			for(int i = 7; i >= 0; i--) {

				if(this->v[i] < val.v[i]) {
					// less than
					return -1;
				} else if(this->v[i] > val.v[i]) {
					// greater than
					return 1;
				}
			}

			// equal
			return 0;
		}

		int cmp(unsigned int &val) const
		{
			// If any higher bits are set then it is greater
			for(int i = 7; i >= 1; i--) {
				if(this->v[i]) {
					return 1;
				}
			}

			if(this->v[0] > val) {
				return 1;
			} else if(this->v[0] < val) {
				return -1;
			}

			return 0;
		}

		uint256 pow(int n)
		{
			uint256 product(1);
			uint256 square = *this;

			while(n) {
				if(n & 1) {
					product = product.mul(square);
				}
				square = square.mul(square);

				n >>= 1;
			}

			return product;
		}

		bool bit(int n)
		{
			n = n % 256;

			return (this->v[n / 32] & (0x1 << (n % 32))) != 0;
		}

                bool isEven() const
                {
                        return (this->v[0] & 1) == 0;
                }

                std::string toString(int base = 16) const;

        uint64_t toUint64()
        {
            return ((uint64_t)this->v[1] << 32) | v[0];
        }
	};

	const unsigned int _POINT_AT_INFINITY_WORDS[8] = { 0xFFFFFFFF, 0xFFFFFFFF, 0xFFFFFFFF, 0xFFFFFFFF, 0xFFFFFFFF, 0xFFFFFFFF, 0xFFFFFFFF, 0xFFFFFFFF };
	const unsigned int _P_WORDS[8] = { 0xFFFFFC2F, 0xFFFFFFFE, 0xFFFFFFFF, 0xFFFFFFFF, 0xFFFFFFFF, 0xFFFFFFFF, 0xFFFFFFFF, 0xFFFFFFFF };
	const unsigned int _N_WORDS[8] = { 0xD0364141, 0xBFD25E8C, 0xAF48A03B, 0xBAAEDCE6, 0xFFFFFFFE, 0xFFFFFFFF, 0xFFFFFFFF, 0xFFFFFFFF };
	const unsigned int _GX_WORDS[8] = { 0x16F81798, 0x59F2815B, 0x2DCE28D9, 0x029BFCDB, 0xCE870B07, 0x55A06295, 0xF9DCBBAC, 0x79BE667E };
	const unsigned int _GY_WORDS[8] = { 0xFB10D4B8, 0x9C47D08F, 0xA6855419, 0xFD17B448, 0x0E1108A8, 0x5DA4FBFC, 0x26A3C465, 0x483ADA77 };
	const unsigned int _BETA_WORDS[8] = {0x719501EE, 0xC1396C28, 0x12F58995, 0x9CF04975, 0xAC3434E9, 0x6E64479E, 0x657C0710, 0x7AE96A2B };
	const unsigned int _LAMBDA_WORDS[8] = {0x1B23BD72, 0xDF02967C, 0x20816678, 0x122E22EA, 0x8812645A, 0xA5261C02, 0xC05C30E0, 0x5363AD4C };


	class ecpoint {

	public:
		uint256 x;
		uint256 y;

		ecpoint()
		{
			this->x = uint256(_POINT_AT_INFINITY_WORDS);
			this->y = uint256(_POINT_AT_INFINITY_WORDS);
		}

		ecpoint(const uint256 &x, const uint256 &y)
		{
			this->x = x;
			this->y = y;
		}

		ecpoint(const ecpoint &p)
		{
			this->x = p.x;
			this->y = p.y;
		}

		ecpoint operator=(const ecpoint &p)
		{
			this->x = p.x;
			this->y = p.y;

			return *this;
		}

		bool operator==(const ecpoint &p) const
		{
			return this->x == p.x && this->y == p.y;
		}

                std::string toString(bool compressed = false) const
                {
                        if(!compressed) {
                                return "04" + this->x.toString() + this->y.toString();
                        } else {
                                if(this->y.isEven()) {
                                        return "02" + this->x.toString();
                                } else {
                                        return "03" + this->x.toString();
                                }
                        }
                }
	};

	const uint256 P(_P_WORDS);
	const uint256 N(_N_WORDS);

        const uint256 BETA(_BETA_WORDS);
        const uint256 LAMBDA(_LAMBDA_WORDS);

        inline const uint256& glvLambda() { return LAMBDA; }
        inline const uint256& glvBeta() { return BETA; }

        ecpoint pointAtInfinity();
        ecpoint G();
        uint256 multiplyModP(const uint256 &a, const uint256&b);

        struct GLVSplit {
            uint256 k1;
            uint256 k2;
            bool k1Neg;
            bool k2Neg;
            GLVSplit() : k1(0), k2(0), k1Neg(false), k2Neg(false) {}
        };

        inline ecpoint glvEndomorphismBasePoint()
        {
            uint256 x = multiplyModP(G().x, glvBeta());
            return ecpoint(x, G().y);
        }

        inline GLVSplit splitScalar(const uint256 &k)
        {
            static const unsigned int A1_WORDS[8] = {
                2458184469u, 3899429092u, 2815716301u, 814141985u,
                0u, 0u, 0u, 0u
            };
            static const unsigned int B1_WORDS[8] = {
                180348099u, 1867808681u, 17729576u, 3829628630u,
                0u, 0u, 0u, 0u
            };
            static const unsigned int A2_WORDS[8] = {
                2638532568u, 1472270477u, 2833445878u, 348803319u,
                1u, 0u, 0u, 0u
            };
            static const unsigned int B2_WORDS[8] = {
                2458184469u, 3899429092u, 2815716301u, 814141985u,
                0u, 0u, 0u, 0u
            };
            static const unsigned int G1_WORDS[5] = {
                3944037802u, 2430898820u, 1808656492u, 3525421012u, 12422u
            };
            static const unsigned int G2_WORDS[5] = {
                3838059026u, 2141784767u, 2284351316u, 2127954190u, 58435u
            };

            auto mulWords = [](const unsigned int *x, int xLen,
                                const unsigned int *y, int yLen,
                                unsigned int *z) {
                for(int i = 0; i < xLen + yLen; ++i) z[i] = 0u;
                for(int i = 0; i < xLen; ++i) {
                    unsigned int carry = 0u;
                    for(int j = 0; j < yLen; ++j) {
                        uint64_t p = (uint64_t)x[i] * (uint64_t)y[j] + z[i+j] + carry;
                        z[i+j] = (unsigned int)p;
                        carry = (unsigned int)(p >> 32);
                    }
                    z[i + yLen] = carry;
                }
            };

            unsigned int kWords[8];
            k.exportWords(kWords, 8, uint256::LittleEndian);

            unsigned int prod1[13];
            mulWords(kWords, 8, G1_WORDS, 5, prod1);
            unsigned int c1Words[8] = {0};
            for(int i = 0; i < 5; ++i) {
                unsigned int lo = (i + 8 < 13) ? prod1[i + 8] : 0u;
                unsigned int hi = (i + 9 < 13) ? prod1[i + 9] : 0u;
                c1Words[i] = (lo >> 16) | (hi << 16);
            }
            uint256 c1(c1Words);

            unsigned int prod2[13];
            mulWords(kWords, 8, G2_WORDS, 5, prod2);
            unsigned int c2Words[8] = {0};
            for(int i = 0; i < 5; ++i) {
                unsigned int lo = (i + 8 < 13) ? prod2[i + 8] : 0u;
                unsigned int hi = (i + 9 < 13) ? prod2[i + 9] : 0u;
                c2Words[i] = (lo >> 16) | (hi << 16);
            }
            uint256 c2(c2Words);

            uint256 t1 = c1.mul(uint256(A1_WORDS));
            uint256 t2 = c2.mul(uint256(A2_WORDS));
            uint256 sum = t1.add(t2);

            GLVSplit r;
            if(k.cmp(sum) >= 0) {
                r.k1 = k.sub(sum);
                r.k1Neg = false;
            } else {
                r.k1 = sum.sub(k);
                r.k1Neg = true;
            }

            uint256 u1 = c1.mul(uint256(B1_WORDS));
            uint256 u2 = c2.mul(uint256(B2_WORDS));
            if(u1.cmp(u2) >= 0) {
                r.k2 = u1.sub(u2);
                r.k2Neg = false;
            } else {
                r.k2 = u2.sub(u1);
                r.k2Neg = true;
            }

            return r;
        }


        uint256 negModP(const uint256 &x);
	uint256 negModN(const uint256 &x);

	uint256 addModP(const uint256 &a, const uint256 &b);
	uint256 subModP(const uint256 &a, const uint256 &b);
	uint256 multiplyModP(const uint256 &a, const uint256&b);
	uint256 multiplyModN(const uint256 &a, const uint256 &b);

	ecpoint addPoints(const ecpoint &p, const ecpoint &q);
	ecpoint doublePoint(const ecpoint &p);

	uint256 invModP(const uint256 &x);

        bool isPointAtInfinity(const ecpoint &p);
        ecpoint multiplyPoint(const uint256 &k, const ecpoint &p);
        ecpoint multiplyPointSmall(const uint256 &k, const ecpoint &p);

        inline ecpoint glvEndomorphism(const ecpoint &p)
        {
            return ecpoint(multiplyModP(p.x, glvBeta()), p.y);
        }

        inline ecpoint glvRecombine(const GLVSplit &s, const ecpoint &p1, const ecpoint &p2)
        {
            ecpoint r1 = p1;
            ecpoint r2 = p2;
            if(s.k1Neg) {
                r1.y = negModP(r1.y);
            }
            if(s.k2Neg) {
                r2.y = negModP(r2.y);
            }
            return addPoints(r1, r2);
        }

	uint256 addModN(const uint256 &a, const uint256 &b);
	uint256 subModN(const uint256 &a, const uint256 &b);

	uint256 generatePrivateKey();

	bool pointExists(const ecpoint &p);

	void generateKeyPairsBulk(unsigned int count, const ecpoint &basePoint, std::vector<uint256> &privKeysOut, std::vector<ecpoint> &pubKeysOut);
	void generateKeyPairsBulk(const ecpoint &basePoint, std::vector<uint256> &privKeys, std::vector<ecpoint> &pubKeysOut);

	ecpoint parsePublicKey(const std::string &pubKeyString);
}

#endif