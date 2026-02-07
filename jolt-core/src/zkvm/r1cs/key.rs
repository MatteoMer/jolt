use ark_serialize::{CanonicalDeserialize, CanonicalSerialize};
use sha3::{Digest, Sha3_256};

use crate::{
    field::JoltField, poly::lagrange_poly::LagrangePolynomial, zkvm::r1cs::inputs::NUM_R1CS_INPUTS,
};

use super::constraints::{
    OUTER_UNIVARIATE_SKIP_DOMAIN_SIZE, R1CS_CONSTRAINTS, R1CS_CONSTRAINTS_FIRST_GROUP,
    R1CS_CONSTRAINTS_SECOND_GROUP,
};
use crate::utils::math::Math;
use crate::zkvm::r1cs::inputs::JoltR1CSInputs;

#[derive(Clone, Copy, CanonicalSerialize, CanonicalDeserialize)]
pub struct UniformSpartanKey<F: JoltField> {
    /// Number of constraints across all steps padded to nearest power of 2
    pub num_cons_total: usize,

    /// Number of steps padded to the nearest power of 2
    pub num_steps: usize,

    /// Digest of verifier key
    pub(crate) vk_digest: F,
}

impl<F: JoltField> UniformSpartanKey<F> {
    pub fn new(num_steps: usize) -> Self {
        assert!(num_steps.is_power_of_two());
        let rows_per_step_padded = Self::num_rows_per_step().next_power_of_two();
        let total_rows = (num_steps * rows_per_step_padded).next_power_of_two();
        let vk_digest = Self::digest(num_steps);
        Self {
            num_cons_total: total_rows,
            num_steps,
            vk_digest,
        }
    }

    #[inline]
    fn num_vars() -> usize {
        JoltR1CSInputs::num_inputs()
    }

    #[inline]
    fn num_rows_per_step() -> usize {
        R1CS_CONSTRAINTS.len()
    }

    pub fn num_vars_uniform_padded(&self) -> usize {
        Self::num_vars().next_power_of_two()
    }

    /// Number of cycle variables, e.g. number of bits needed to represent all cycles in the trace
    pub fn num_cycle_vars(&self) -> usize {
        self.num_steps.next_power_of_two().log_2()
    }

    /// Number of bits needed for all rows.
    /// With univariate skip, this is the number of cycle variables plus two (one for univariate skip of degree ~13-15, and one for the streaming round)
    pub fn num_rows_bits(&self) -> usize {
        self.num_cycle_vars() + 2
    }

    /// Evaluates `sum_y A(rx_constr, y)*z(y) * sum_y B(rx_constr, y)*z(y)`.
    ///
    /// Note `rx_constr` is the randomness used to bind the rows of `A` and `B`.
    /// `r1cs_input_evals` should ordered as per [`ALL_R1CS_INPUTS`].
    ///
    /// [`ALL_R1CS_INPUTS`]: crate::zkvm::r1cs::inputs::ALL_R1CS_INPUTS
    pub fn evaluate_inner_sum_product_at_point(
        &self,
        rx_constr: &[F::Challenge],
        r1cs_input_evals: [F; NUM_R1CS_INPUTS],
    ) -> F {
        // Row axis: r_constr = [r_stream, r0]; use Lagrange basis for first-round
        // (half the number of R1CS constraints)
        // and linear blend for the two groups using r_stream
        debug_assert!(rx_constr.len() >= 2);
        let r_stream = rx_constr[0];
        let r0 = rx_constr[1];

        // Lagrange weights over the univariate-skip base domain at r0
        let w =
            LagrangePolynomial::<F>::evals::<F::Challenge, OUTER_UNIVARIATE_SKIP_DOMAIN_SIZE>(&r0);

        // Build z(r_cycle) vector with a trailing 1 for the constant column
        let z_const_col = JoltR1CSInputs::num_inputs();
        let mut z = r1cs_input_evals.to_vec();
        z.push(F::one());

        // Group 0 fused Az,Bz via dot product of LC with z(r_cycle)
        let mut az_g0 = F::zero();
        let mut bz_g0 = F::zero();
        for i in 0..R1CS_CONSTRAINTS_FIRST_GROUP.len() {
            let lc_a = &R1CS_CONSTRAINTS_FIRST_GROUP[i].cons.a;
            let lc_b = &R1CS_CONSTRAINTS_FIRST_GROUP[i].cons.b;
            az_g0 += w[i] * lc_a.dot_product::<F>(&z, z_const_col);
            bz_g0 += w[i] * lc_b.dot_product::<F>(&z, z_const_col);
        }

        // Group 1 fused Az,Bz (use same Lagrange weights order as construction)
        let mut az_g1 = F::zero();
        let mut bz_g1 = F::zero();
        let g2_len = core::cmp::min(
            R1CS_CONSTRAINTS_SECOND_GROUP.len(),
            OUTER_UNIVARIATE_SKIP_DOMAIN_SIZE,
        );
        for i in 0..g2_len {
            let lc_a = &R1CS_CONSTRAINTS_SECOND_GROUP[i].cons.a;
            let lc_b = &R1CS_CONSTRAINTS_SECOND_GROUP[i].cons.b;
            az_g1 += w[i] * lc_a.dot_product::<F>(&z, z_const_col);
            bz_g1 += w[i] * lc_b.dot_product::<F>(&z, z_const_col);
        }

        // Bind by r_stream to match the outer streaming combination used for final Az,Bz
        let az_final = az_g0 + r_stream * (az_g1 - az_g0);
        let bz_final = bz_g0 + r_stream * (bz_g1 - bz_g0);

        #[cfg(feature = "zolt-debug")]
        {
            use ark_serialize::CanonicalSerialize;
            let mut az_g0_bytes = [0u8; 32];
            let mut bz_g0_bytes = [0u8; 32];
            let mut az_g1_bytes = [0u8; 32];
            let mut bz_g1_bytes = [0u8; 32];
            let mut az_final_bytes = [0u8; 32];
            let mut bz_final_bytes = [0u8; 32];
            let mut r_stream_bytes = [0u8; 32];
            let mut r0_bytes = [0u8; 32];
            az_g0.serialize_compressed(&mut az_g0_bytes[..]).ok();
            bz_g0.serialize_compressed(&mut bz_g0_bytes[..]).ok();
            az_g1.serialize_compressed(&mut az_g1_bytes[..]).ok();
            bz_g1.serialize_compressed(&mut bz_g1_bytes[..]).ok();
            az_final.serialize_compressed(&mut az_final_bytes[..]).ok();
            bz_final.serialize_compressed(&mut bz_final_bytes[..]).ok();
            let r_stream_f: F = r_stream.into();
            let r0_f: F = r0.into();
            r_stream_f.serialize_compressed(&mut r_stream_bytes[..]).ok();
            r0_f.serialize_compressed(&mut r0_bytes[..]).ok();
            eprintln!("[inner_sum_prod] rx_constr: r_stream={:02x?}, r0={:02x?}", &r_stream_bytes[0..8], &r0_bytes[0..8]);
            eprintln!("[inner_sum_prod] az_g0={:02x?}", &az_g0_bytes);
            eprintln!("[inner_sum_prod] bz_g0={:02x?}", &bz_g0_bytes);
            eprintln!("[inner_sum_prod] az_g1={:02x?}", &az_g1_bytes);
            eprintln!("[inner_sum_prod] bz_g1={:02x?}", &bz_g1_bytes);
            eprintln!("[inner_sum_prod] az_final={:02x?}", &az_final_bytes);
            eprintln!("[inner_sum_prod] bz_final={:02x?}", &bz_final_bytes);
        }

        az_final * bz_final
    }

    /// Returns the digest of the R1CS "shape" derived from compile-time constants
    /// Canonical serialization of constants:
    /// - domain tag
    /// - num_steps (u64 BE)
    /// - num_vars (u32 BE)
    /// - for each row in R1CS_CONSTRAINTS:
    ///   - tag 'A' | row.a terms (sorted by input_index asc) + const term
    ///   - tag 'B' | row.b terms (sorted by input_index asc) + const term
    fn digest(num_steps: usize) -> F {
        let mut bytes: Vec<u8> = Vec::new();
        bytes.extend_from_slice(b"JOLT_R1CS_CONSTRAINTS");
        bytes.extend_from_slice(&num_steps.to_be_bytes());

        let num_vars: u32 = JoltR1CSInputs::num_inputs() as u32;
        bytes.extend_from_slice(&num_vars.to_be_bytes());

        for row_named in R1CS_CONSTRAINTS.iter() {
            let row = &row_named.cons;
            row.a.serialize_canonical(b'A', &mut bytes);
            row.b.serialize_canonical(b'B', &mut bytes);
        }

        let mut hasher = Sha3_256::new();
        hasher.update(&bytes);

        let map_to_field = |digest: &[u8]| -> F {
            let bv = (0..250).map(|i| {
                let (byte_pos, bit_pos) = (i / 8, i % 8);
                let bit = (digest[byte_pos] >> bit_pos) & 1;
                bit == 1
            });

            // turn the bit vector into a scalar
            let mut digest = F::zero();
            let mut coeff = F::one();
            for bit in bv {
                if bit {
                    digest += coeff;
                }
                coeff += coeff;
            }
            digest
        };
        map_to_field(&hasher.finalize())
    }
}
