use crate::field::{ChallengeFieldOps, FieldChallengeOps};
use crate::zkvm::instruction_lookups::LOG_K;
use crate::{field::JoltField, utils::lookup_bits::LookupBits};

use super::{PrefixCheckpoint, Prefixes, SparseDensePrefix};

pub enum LowerWordPrefix<const XLEN: usize> {}

impl<const XLEN: usize, F: JoltField> SparseDensePrefix<F> for LowerWordPrefix<XLEN> {
    fn prefix_mle<C>(
        checkpoints: &[PrefixCheckpoint<F>],
        r_x: Option<C>,
        c: u32,
        mut b: LookupBits,
        j: usize,
    ) -> F
    where
        C: ChallengeFieldOps<F>,
        F: FieldChallengeOps<C>,
    {
        let suffix_len = LOG_K - j - b.len() - 1;
        // Ignore high-order variables
        if j < XLEN {
            return F::zero();
        }
        let mut result = checkpoints[Prefixes::LowerWord].unwrap_or(F::zero());

        if let Some(r_x) = r_x {
            let y = F::from_u8(c as u8);
            let x_shift = 2 * XLEN - j;
            let y_shift = 2 * XLEN - j - 1;
            result += F::from_u128(1u128 << x_shift) * r_x;
            result += F::from_u128(1u128 << y_shift) * y;
        } else {
            let x = F::from_u8(c as u8);
            let y_msb = b.pop_msb();
            let x_shift = 2 * XLEN - j - 1;
            let y_shift = 2 * XLEN - j - 2;
            result += F::from_u128(1 << x_shift) * x;
            result += F::from_u128(1 << y_shift) * F::from_u8(y_msb);
        }

        // Add in low-order bits from `b`
        result += F::from_u128(u128::from(b) << suffix_len);

        result
    }

    fn update_prefix_checkpoint<C>(
        checkpoints: &[PrefixCheckpoint<F>],
        r_x: C,
        r_y: C,
        j: usize,
        _suffix_len: usize,
    ) -> PrefixCheckpoint<F>
    where
        C: ChallengeFieldOps<F>,
        F: FieldChallengeOps<C>,
    {
        if j < XLEN {
            return None.into();
        }
        let x_shift = 2 * XLEN - j;
        let y_shift = 2 * XLEN - j - 1;
        let mut updated = checkpoints[Prefixes::LowerWord].unwrap_or(F::zero());

        let coeff_x = F::from_u128(1 << x_shift);
        let coeff_y = F::from_u128(1 << y_shift);
        let contrib_x: F = coeff_x * r_x;
        let contrib_y: F = coeff_y * r_y;

        #[cfg(feature = "zolt-debug")]
        if j == 65 || j == 127 {
            use ark_serialize::CanonicalSerialize;
            let r_x_f: F = r_x.into();
            let r_y_f: F = r_y.into();
            let mut r_x_bytes = [0u8; 32];
            let mut r_y_bytes = [0u8; 32];
            let mut coeff_x_bytes = [0u8; 32];
            let mut contrib_x_bytes = [0u8; 32];
            let mut contrib_y_bytes = [0u8; 32];
            let mut updated_bytes = [0u8; 32];
            r_x_f.serialize_compressed(&mut r_x_bytes[..]).ok();
            r_y_f.serialize_compressed(&mut r_y_bytes[..]).ok();
            coeff_x.serialize_compressed(&mut coeff_x_bytes[..]).ok();
            contrib_x.serialize_compressed(&mut contrib_x_bytes[..]).ok();
            contrib_y.serialize_compressed(&mut contrib_y_bytes[..]).ok();
            updated.serialize_compressed(&mut updated_bytes[..]).ok();
            eprintln!("[JOLT LOWERWORD UPDATE] j={}, x_shift={}, y_shift={}", j, x_shift, y_shift);
            eprintln!("  r_x (as F)       = {:02x?}", r_x_bytes);
            eprintln!("  r_y (as F)       = {:02x?}", r_y_bytes);
            eprintln!("  coeff_x          = {:02x?}", coeff_x_bytes);
            eprintln!("  contrib_x        = {:02x?}", contrib_x_bytes);
            eprintln!("  contrib_y        = {:02x?}", contrib_y_bytes);
            eprintln!("  prev             = {:02x?}", updated_bytes);
        }

        updated += contrib_x;
        updated += contrib_y;

        #[cfg(feature = "zolt-debug")]
        if j == 65 || j == 127 {
            use ark_serialize::CanonicalSerialize;
            let mut updated_bytes = [0u8; 32];
            updated.serialize_compressed(&mut updated_bytes[..]).ok();
            eprintln!("  new              = {:02x?}", updated_bytes);
        }

        Some(updated).into()
    }
}
