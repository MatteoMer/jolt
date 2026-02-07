use super::transcript::Transcript;
use crate::field::JoltField;
use ark_ec::{AffineRepr, CurveGroup};
use ark_serialize::CanonicalSerialize;
use blake2::digest::consts::U32;
use blake2::{Blake2b, Digest};

type Blake2b256 = Blake2b<U32>;
use std::borrow::Borrow;

/// Represents the current state of the protocol's Fiat-Shamir transcript using Blake2b.
#[derive(Default, Clone)]
pub struct Blake2bTranscript {
    /// 256-bit running state
    pub state: [u8; 32],
    /// We append an ordinal to each invocation of the hash
    n_rounds: u32,
    #[cfg(test)]
    /// A complete history of the transcript's `state`; used for testing.
    state_history: Vec<[u8; 32]>,
    #[cfg(test)]
    /// For a proof to be valid, the verifier's `state_history` should always match
    /// the prover's. In testing, the Jolt verifier may be provided the prover's
    /// `state_history` so that we can detect any deviations and the backtrace can
    /// tell us where it happened.
    expected_state_history: Option<Vec<[u8; 32]>>,
}

impl Blake2bTranscript {
    /// Gives the hasher object with the running seed and index added
    /// To load hash you must call finalize, after appending u8 vectors
    fn hasher(&self) -> Blake2b256 {
        let mut packed = [0_u8; 28].to_vec();
        packed.append(&mut self.n_rounds.to_be_bytes().to_vec());
        Blake2b256::new()
            .chain_update(self.state)
            .chain_update(&packed)
    }

    // Loads arbitrary byte lengths using ceil(out/32) invocations of 32 byte randoms
    // Discards top bits when the size is less than 32 bytes
    fn challenge_bytes(&mut self, out: &mut [u8]) {
        let mut remaining_len = out.len();
        let mut start = 0;
        while remaining_len > 32 {
            self.challenge_bytes32(&mut out[start..start + 32]);
            start += 32;
            remaining_len -= 32;
        }
        // We load a full 32 byte random region
        let mut full_rand = vec![0_u8; 32];
        self.challenge_bytes32(&mut full_rand);
        // Then only clone the first bits of this random region to perfectly fill out
        out[start..start + remaining_len].clone_from_slice(&full_rand[0..remaining_len]);
    }

    // Loads exactly 32 bytes from the transcript by hashing the seed with the round constant
    fn challenge_bytes32(&mut self, out: &mut [u8]) {
        assert_eq!(32, out.len());
        let rand: [u8; 32] = self.hasher().finalize().into();
        out.clone_from_slice(rand.as_slice());
        self.update_state(rand);
    }

    fn update_state(&mut self, new_state: [u8; 32]) {
        self.state = new_state;
        self.n_rounds += 1;
        #[cfg(test)]
        {
            if let Some(expected_state_history) = &self.expected_state_history {
                assert!(
                    new_state == expected_state_history[self.n_rounds as usize],
                    "Fiat-Shamir transcript mismatch"
                );
            }
            self.state_history.push(new_state);
        }
    }
}

impl Transcript for Blake2bTranscript {
    fn new(label: &'static [u8]) -> Self {
        // Hash in the label
        assert!(label.len() < 33);
        let hasher = if label.len() == 32 {
            Blake2b256::new().chain_update(label)
        } else {
            let zeros = vec![0_u8; 32 - label.len()];
            Blake2b256::new().chain_update(label).chain_update(zeros)
        };
        let out = hasher.finalize();

        Self {
            state: out.into(),
            n_rounds: 0,
            #[cfg(test)]
            state_history: vec![out.into()],
            #[cfg(test)]
            expected_state_history: None,
        }
    }

    #[cfg(test)]
    /// Compare this transcript to `other` and panic if/when they deviate.
    /// Typically used to compare the verifier's transcript to the prover's.
    fn compare_to(&mut self, other: Self) {
        self.expected_state_history = Some(other.state_history);
    }

    fn append_message(&mut self, msg: &'static [u8]) {
        // We require all messages to fit into one evm word and then right pad them
        // right padding matches the format of the strings when cast to bytes 32 in solidity
        assert!(msg.len() < 33);
        #[cfg(feature = "zolt-debug")]
        {
            // Only print for UniPoly markers
            if msg == b"UniPoly_begin" || msg == b"UniPoly_end"
               || msg == b"UncompressedUniPoly_begin" || msg == b"UncompressedUniPoly_end" {
                let msg_str = std::str::from_utf8(msg).unwrap_or("(invalid utf8)");
                eprintln!("[JOLT TRANSCRIPT MSG] append_message({:?}), state BEFORE: {:02x?}", msg_str, &self.state[0..8]);
            }
        }
        let hasher = if msg.len() == 32 {
            self.hasher().chain_update(msg)
        } else {
            let mut packed = msg.to_vec();
            packed.append(&mut vec![0_u8; 32 - msg.len()]);
            self.hasher().chain_update(packed)
        };
        // Instantiate hasher add our seed, position and msg
        self.update_state(hasher.finalize().into());
        #[cfg(feature = "zolt-debug")]
        {
            if msg == b"UniPoly_begin" || msg == b"UniPoly_end"
               || msg == b"UncompressedUniPoly_begin" || msg == b"UncompressedUniPoly_end" {
                let msg_str = std::str::from_utf8(msg).unwrap_or("(invalid utf8)");
                eprintln!("[JOLT TRANSCRIPT MSG] append_message({:?}), state AFTER: {:02x?}", msg_str, &self.state[0..8]);
            }
        }
    }

    fn append_bytes(&mut self, bytes: &[u8]) {
        // Add the message and label
        let hasher = self.hasher().chain_update(bytes);
        self.update_state(hasher.finalize().into());
    }

    fn append_u64(&mut self, x: u64) {
        // Allocate into a 32 byte region
        let mut packed = [0_u8; 24].to_vec();
        packed.append(&mut x.to_be_bytes().to_vec());
        let hasher = self.hasher().chain_update(packed.clone());
        self.update_state(hasher.finalize().into());
    }

    fn append_scalar<F: JoltField>(&mut self, scalar: &F) {
        let mut buf = vec![];
        scalar.serialize_uncompressed(&mut buf).unwrap();
        // Serialize uncompressed gives the scalar in LE byte order which is not
        // a natural representation in the EVM for scalar math so we reverse
        // to get an EVM compatible version.
        buf = buf.into_iter().rev().collect();
        self.append_bytes(&buf);
    }

    fn append_serializable<F: CanonicalSerialize>(&mut self, scalar: &F) {
        let mut buf = vec![];
        scalar.serialize_uncompressed(&mut buf).unwrap();
        // Serialize uncompressed gives the scalar in LE byte order which is not
        // a natural representation in the EVM for scalar math so we reverse
        // to get an EVM compatible version.
        buf = buf.into_iter().rev().collect();
        self.append_bytes(&buf);
    }

    fn append_scalars<F: JoltField>(&mut self, scalars: &[impl Borrow<F>]) {
        self.append_message(b"begin_append_vector");
        for item in scalars.iter() {
            self.append_scalar(item.borrow());
        }
        self.append_message(b"end_append_vector");
    }

    fn append_point<G: CurveGroup>(&mut self, point: &G) {
        // If we add the point at infinity then we hash over a region of zeros
        if point.is_zero() {
            self.append_bytes(&[0_u8; 64]);
            return;
        }

        let aff = point.into_affine();
        let mut x_bytes = vec![];
        let mut y_bytes = vec![];
        // The native serialize for the points are le encoded in x,y format and simply reversing
        // can lead to errors so we extract the affine coordinates and the encode them be before writing
        let x = aff.x().unwrap();
        x.serialize_compressed(&mut x_bytes).unwrap();
        x_bytes = x_bytes.into_iter().rev().collect();
        let y = aff.y().unwrap();
        y.serialize_compressed(&mut y_bytes).unwrap();
        y_bytes = y_bytes.into_iter().rev().collect();

        let hasher = self.hasher().chain_update(x_bytes).chain_update(y_bytes);
        self.update_state(hasher.finalize().into());
    }

    fn append_points<G: CurveGroup>(&mut self, points: &[G]) {
        self.append_message(b"begin_append_vector");
        for item in points.iter() {
            self.append_point(item);
        }
        self.append_message(b"end_append_vector");
    }

    fn challenge_u128(&mut self) -> u128 {
        let mut buf = vec![0u8; 16];
        self.challenge_bytes(&mut buf);
        buf = buf.into_iter().rev().collect();
        u128::from_be_bytes(buf.try_into().unwrap())
    }

    fn challenge_scalar<F: JoltField>(&mut self) -> F {
        // Under the hood all Fr are 128 bits for performance
        self.challenge_scalar_128_bits()
    }

    fn challenge_scalar_128_bits<F: JoltField>(&mut self) -> F {
        #[cfg(feature = "zolt-debug")]
        {
            eprintln!("[JOLT TRANSCRIPT] challenge_scalar state BEFORE: {{ {:02x} {:02x} {:02x} {:02x} {:02x} {:02x} {:02x} {:02x} }}",
                self.state[0], self.state[1], self.state[2], self.state[3],
                self.state[4], self.state[5], self.state[6], self.state[7]);
            eprintln!("[JOLT TRANSCRIPT] n_rounds BEFORE: {}", self.n_rounds);
        }
        let mut buf = vec![0u8; 16];
        self.challenge_bytes(&mut buf);

        buf = buf.into_iter().rev().collect();
        let result = F::from_bytes(&buf);
        #[cfg(feature = "zolt-debug")]
        {
            use ark_serialize::CanonicalSerialize;
            let mut result_bytes = [0u8; 32];
            result.serialize_compressed(&mut result_bytes[..]).ok();
            eprintln!("[JOLT TRANSCRIPT] challenge_scalar result bytes (LE): {:02x?}", &result_bytes);
            eprintln!("[JOLT TRANSCRIPT] challenge_scalar state AFTER: {{ {:02x} {:02x} {:02x} {:02x} {:02x} {:02x} {:02x} {:02x} }}",
                self.state[0], self.state[1], self.state[2], self.state[3],
                self.state[4], self.state[5], self.state[6], self.state[7]);
        }
        result
    }

    fn challenge_vector<F: JoltField>(&mut self, len: usize) -> Vec<F> {
        (0..len)
            .map(|_i| self.challenge_scalar())
            .collect::<Vec<F>>()
    }

    // Compute powers of scalar q : (1, q, q^2, ..., q^(len-1))
    fn challenge_scalar_powers<F: JoltField>(&mut self, len: usize) -> Vec<F> {
        #[cfg(feature = "zolt-debug")]
        {
            eprintln!("[JOLT TRANSCRIPT] challenge_scalar_powers({}) state BEFORE: {:02x?}", len, &self.state[0..16]);
            eprintln!("[JOLT TRANSCRIPT] n_rounds BEFORE: {}", self.n_rounds);
        }
        let q: F = self.challenge_scalar();
        let mut q_powers = vec![F::one(); len];
        for i in 1..len {
            q_powers[i] = q_powers[i - 1] * q;
        }
        #[cfg(feature = "zolt-debug")]
        {
            use ark_serialize::CanonicalSerialize;
            let mut q_bytes = [0u8; 32];
            q.serialize_compressed(&mut q_bytes[..]).ok();
            eprintln!("[JOLT TRANSCRIPT] challenge_scalar_powers({}) gamma (LE full): {:02x?}", len, &q_bytes);
            eprintln!("[JOLT TRANSCRIPT]   gamma low bytes (0..16): {:02x?}", &q_bytes[0..16]);
        }
        q_powers
    }

    fn challenge_scalar_optimized<F: JoltField>(&mut self) -> F::Challenge {
        // The smaller challenge which is then converted into a
        // MontU128Challenge
        let challenge_scalar: u128 = self.challenge_u128();
        F::Challenge::from(challenge_scalar)
    }

    fn challenge_vector_optimized<F: JoltField>(&mut self, len: usize) -> Vec<F::Challenge> {
        (0..len)
            .map(|_i| self.challenge_scalar_optimized::<F>())
            .collect::<Vec<F::Challenge>>()
    }

    fn challenge_scalar_powers_optimized<F: JoltField>(&mut self, len: usize) -> Vec<F> {
        // This is still different from challenge_scalar_powers as inside the for loop
        // we use an optimised multiplication every time we compute the powers.
        let q: F::Challenge = self.challenge_scalar_optimized::<F>();
        let mut q_powers = vec![<F as ark_std::One>::one(); len];
        for i in 1..len {
            q_powers[i] = q * q_powers[i - 1]; // this is optimised
        }
        q_powers
    }

    #[cfg(feature = "zolt-debug")]
    fn debug_state(&self) -> [u8; 32] {
        self.state
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ark_bn254::Fr;
    use std::collections::HashSet;

    #[test]
    fn test_challenge_scalar_128_bits() {
        let mut transcript = Blake2bTranscript::new(b"test_128_bit_scalar");
        let mut scalars = HashSet::new();

        for i in 0..10000 {
            let scalar: Fr = transcript.challenge_scalar_128_bits();

            let num_bits = scalar.num_bits();
            assert!(
                num_bits <= 128,
                "Scalar at iteration {i} has {num_bits} bits, expected <= 128",
            );

            assert!(
                scalars.insert(scalar),
                "Duplicate scalar found at iteration {i}",
            );
        }
    }

    #[test]
    fn test_challenge_special_trivial() {
        use ark_std::UniformRand;
        let mut rng = ark_std::test_rng();
        let mut transcript1 = Blake2bTranscript::new(b"test_trivial_challenge");

        let challenge = transcript1.challenge_scalar_optimized::<Fr>();
        // The same challenge as a full fat Fr element
        let challenge_regular: Fr = challenge.into();

        let field_elements: Vec<Fr> = (0..10).map(|_| Fr::rand(&mut rng)).collect();

        for (i, &field_elem) in field_elements.iter().enumerate() {
            let result_challenge = field_elem * challenge;
            let result_regular = field_elem * challenge_regular;

            assert_eq!(
                result_challenge, result_regular,
                "Multiplication mismatch at index {i}"
            );
        }

        let field_elem = Fr::rand(&mut rng);
        #[allow(clippy::op_ref)]
        let result_ref = field_elem * &challenge;
        let result_regular = field_elem * challenge;
        assert_eq!(
            result_ref, result_regular,
            "Reference multiplication mismatch"
        );
    }

    #[test]
    fn test_challenge_limbs_for_zolt_compat() {
        use ark_ff::PrimeField;
        use ark_serialize::CanonicalSerialize;

        let mut transcript = Blake2bTranscript::new(b"zolt_compat_test");
        transcript.append_message(b"test_data");

        // Get the raw u128 challenge value
        let mut transcript2 = Blake2bTranscript::new(b"zolt_compat_test");
        transcript2.append_message(b"test_data");
        let raw_u128 = transcript2.challenge_u128();

        // Get the MontU128Challenge
        let challenge = transcript.challenge_scalar_optimized::<Fr>();

        // Convert to Fr to get the internal limbs
        let fr: Fr = challenge.into();

        // Get the Montgomery representation as bytes
        let mut fr_bytes = [0u8; 32];
        fr.serialize_compressed(&mut fr_bytes[..]).unwrap();

        println!("\n=== CHALLENGE LIMBS TEST FOR ZOLT COMPATIBILITY ===");
        println!("Raw u128 value: 0x{:032x}", raw_u128);
        println!("  low:  0x{:016x}", raw_u128 as u64);
        println!("  high: 0x{:016x}", (raw_u128 >> 64) as u64);
        println!("MontU128Challenge limbs: [0, 0, 0x{:016x}, 0x{:016x}]", challenge.low, challenge.high);
        println!("Fr bytes (serialized): {:02x?}", &fr_bytes);

        // Get the BigInt representation
        let bigint = fr.into_bigint();
        println!("Fr BigInt limbs: [0x{:016x}, 0x{:016x}, 0x{:016x}, 0x{:016x}]",
            bigint.0[0], bigint.0[1], bigint.0[2], bigint.0[3]);

        // Verify the MontU128Challenge matches the raw values
        assert_eq!(challenge.low, raw_u128 as u64);
        assert_eq!(challenge.high, ((raw_u128 >> 64) as u64) & (u64::MAX >> 3)); // 125-bit mask

        println!("=== END TEST ===\n");
    }

    #[test]
    fn test_eq_poly_operations_for_zolt_compat() {
        use ark_ff::PrimeField;
        use ark_ff::One;
        use ark_serialize::CanonicalSerialize;

        let mut transcript = Blake2bTranscript::new(b"zolt_compat_test");
        transcript.append_message(b"test_data");

        let challenge = transcript.challenge_scalar_optimized::<Fr>();
        let fr_challenge: Fr = challenge.into();

        println!("\n=== EQ POLY OPERATIONS TEST FOR ZOLT COMPATIBILITY ===");
        println!("Challenge as Fr BigInt: [{:016x}, {:016x}, {:016x}, {:016x}]",
            fr_challenge.into_bigint().0[0], fr_challenge.into_bigint().0[1],
            fr_challenge.into_bigint().0[2], fr_challenge.into_bigint().0[3]);

        // F::one() - challenge
        let one = Fr::one();
        let one_minus_challenge = one - fr_challenge;

        let mut one_bytes = [0u8; 32];
        let mut challenge_bytes = [0u8; 32];
        let mut result_bytes = [0u8; 32];

        one.serialize_compressed(&mut one_bytes[..]).unwrap();
        fr_challenge.serialize_compressed(&mut challenge_bytes[..]).unwrap();
        one_minus_challenge.serialize_compressed(&mut result_bytes[..]).unwrap();

        // Reverse for BE display
        one_bytes.reverse();
        challenge_bytes.reverse();
        result_bytes.reverse();

        println!("F::one() BE bytes: {:02x?}", &one_bytes);
        println!("challenge BE bytes: {:02x?}", &challenge_bytes);
        println!("1 - challenge BE bytes: {:02x?}", &result_bytes);

        // challenge * challenge
        let challenge_squared = fr_challenge * fr_challenge;
        let mut sq_bytes = [0u8; 32];
        challenge_squared.serialize_compressed(&mut sq_bytes[..]).unwrap();
        sq_bytes.reverse();
        println!("challenge^2 BE bytes: {:02x?}", &sq_bytes);

        println!("=== END TEST ===\n");
    }
}
