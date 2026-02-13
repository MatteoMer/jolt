use std::iter::zip;
use std::sync::Arc;

use allocative::Allocative;
#[cfg(feature = "allocative")]
use allocative::FlameGraphBuilder;
use common::constants::XLEN;
use num_traits::Zero;
use rayon::prelude::*;
use strum::{EnumCount, IntoEnumIterator};
use tracer::instruction::Cycle;

use super::LOG_K;

use crate::{
    field::{JoltField, MulTrunc},
    poly::{
        dense_mlpoly::DensePolynomial,
        eq_poly::EqPolynomial,
        identity_poly::{IdentityPolynomial, OperandPolynomial, OperandSide},
        multilinear_polynomial::{
            BindingOrder, MultilinearPolynomial, PolynomialBinding, PolynomialEvaluation,
        },
        opening_proof::{
            OpeningAccumulator, OpeningPoint, ProverOpeningAccumulator, SumcheckId,
            VerifierOpeningAccumulator, BIG_ENDIAN,
        },
        prefix_suffix::{Prefix, PrefixRegistry, PrefixSuffixDecomposition},
        split_eq_poly::GruenSplitEqPolynomial,
        unipoly::UniPoly,
    },
    subprotocols::{
        mles_product_sum::{eval_linear_prod_accumulate, finish_mles_product_sum_from_evals},
        sumcheck_prover::SumcheckInstanceProver,
        sumcheck_verifier::{SumcheckInstanceParams, SumcheckInstanceVerifier},
    },
    transcripts::Transcript,
    utils::{
        expanding_table::ExpandingTable,
        lookup_bits::LookupBits,
        math::Math,
        thread::{drop_in_background_thread, unsafe_allocate_zero_vec},
    },
    zkvm::{
        config::{self, OneHotParams},
        instruction::{Flags, InstructionLookup, InterleavedBitsMarker, LookupQuery},
        lookup_table::{
            prefixes::{PrefixCheckpoint, PrefixEval, Prefixes},
            suffixes::Suffixes,
            LookupTables,
        },
        witness::VirtualPolynomial,
    },
};

use rayon::iter::{IndexedParallelIterator, ParallelIterator};

// Instruction lookups: Read + RAF batched sumcheck
//
// Notation:
// - Field F. Let K = 2^{LOG_K}, T = 2^{log_T}.
// - Address index k ∈ {0..K-1}, cycle index j ∈ {0..T-1}.
// - eq(k; r_addr) := multilinear equality polynomial over LOG_K vars.
// - eq(j; r_reduction) := equality polynomials over LOG_T vars.
// - ra(k, j) is the selector arising from prefix/suffix condensation.
//   It is decomposed as the product of virtual sub selectors:
//   ra((k_0, k_1, ..., k_{n-1}), j) := ra_0(k_0, j) * ra_1(k_1, j) * ... * ra_{n-1}(k_{n-1}, j).
//   n is typically 1, 2, 4 or 8.
//   logically ra(k, j) = 1 when the j-th cycle's lookup key equals k, and 0 otherwise.// - Val_j(k) ∈ F is the lookup-table value selected by (j, k); concretely Val_j(k) = table_j(k)
//   if cycle j uses a table and 0 otherwise (materialized via prefix/suffix decomposition).
// - raf_flag(j) ∈ {0,1} is 1 iff the instruction at cycle j is NOT interleaved operands.
// - Let LeftPrefix_j, RightPrefix_j, IdentityPrefix_j ∈ F be the address-only (prefix) factors for
//   the left/right operand and identity polynomials at cycle j (from `PrefixSuffixDecomposition`).
//
// We introduce a batching challenge γ ∈ F. Define
//   RafVal_j(k) := (1 - raf_flag(j)) · (LeftPrefix_j + γ · RightPrefix_j)
//                  + raf_flag(j) · γ · IdentityPrefix_j.
// The overall γ-weights are arranged so that γ multiplies RafVal_j(k) in the final identity.
//
// Claims supplied by the accumulator (LHS), all claimed at `SumcheckId::InstructionClaimReduction`
// and `SumcheckId::SpartanProductVirtualization`:
// - rv         := ⟦LookupOutput⟧
// - left_op    := ⟦LeftLookupOperand⟧
// - right_op   := ⟦RightLookupOperand⟧
//   Combined as: rv + γ·left_op + γ^2·right_op
//
// Statement proved by this sumcheck (RHS), for random challenges
// r_addr ∈ F^{LOG_K}, r_reduction ∈ F^{log_T}:
//
//   rv(r_reduction) + γ·left_op(r_reduction) + γ^2·right_op(r_reduction)
//   = Σ_{j=0}^{T-1} Σ_{k=0}^{K-1} [ eq(j; r_reduction) · ra(k, j) · (Val_j(k) + γ · RafVal_j(k)) ].
//
// Prover structure:
// - First log(K) rounds bind address vars using prefix/suffix decomposition, accumulating:
//   Σ_k ra(k, j)·Val_j(k)  and  Σ_k ra(k, j)·RafVal_j(k)
//   for each j (via u_evals vectors and suffix polynomials).
// - Last log(T) rounds bind cycle vars producing a degree-3 univariate with the required previous-round claim.
// - The published univariate matches the RHS above; the verifier checks it against the LHS claims.

#[derive(Allocative, Clone)]
pub struct InstructionReadRafSumcheckParams<F: JoltField> {
    /// γ and its square (γ^2) used for batching rv/branch/raf components.
    pub gamma: F,
    pub gamma_sqr: F,
    /// log2(T): number of cycle variables (last rounds bind cycles).
    pub log_T: usize,
    /// How many address variables each virtual ra polynomial has.
    pub ra_virtual_log_k_chunk: usize,
    /// Number of phases for instruction lookups.
    pub phases: usize,
    pub r_reduction: OpeningPoint<BIG_ENDIAN, F>,
}

impl<F: JoltField> InstructionReadRafSumcheckParams<F> {
    pub fn new(
        n_cycle_vars: usize,
        one_hot_params: &OneHotParams,
        opening_accumulator: &dyn OpeningAccumulator<F>,
        transcript: &mut impl Transcript,
    ) -> Self {
        let gamma = transcript.challenge_scalar::<F>();
        let gamma_sqr = gamma.square();
        let phases = config::get_instruction_sumcheck_phases(n_cycle_vars);

        #[cfg(feature = "zolt-debug")]
        {
            use ark_serialize::CanonicalSerialize;
            let mut gamma_bytes = [0u8; 32];
            gamma.serialize_compressed(&mut gamma_bytes[..]).ok();
            eprintln!("[STAGE5] gamma_lookups_raf = {:02x?}", &gamma_bytes);
        }

        let (r_reduction, _) = opening_accumulator.get_virtual_polynomial_opening(
            VirtualPolynomial::LookupOutput,
            SumcheckId::InstructionClaimReduction,
        );

        Self {
            gamma,
            gamma_sqr,
            log_T: n_cycle_vars,
            ra_virtual_log_k_chunk: one_hot_params.lookups_ra_virtual_log_k_chunk,
            phases,
            r_reduction,
        }
    }
}

impl<F: JoltField> SumcheckInstanceParams<F> for InstructionReadRafSumcheckParams<F> {
    fn num_rounds(&self) -> usize {
        LOG_K + self.log_T
    }

    fn input_claim(&self, accumulator: &dyn OpeningAccumulator<F>) -> F {
        let (_, rv_claim) = accumulator.get_virtual_polynomial_opening(
            VirtualPolynomial::LookupOutput,
            SumcheckId::InstructionClaimReduction,
        );
        let (_, rv_claim_branch) = accumulator.get_virtual_polynomial_opening(
            VirtualPolynomial::LookupOutput,
            SumcheckId::SpartanProductVirtualization,
        );
        // TODO: Make error and move to more appropriate place.
        assert_eq!(rv_claim, rv_claim_branch);
        let (_, left_operand_claim) = accumulator.get_virtual_polynomial_opening(
            VirtualPolynomial::LeftLookupOperand,
            SumcheckId::InstructionClaimReduction,
        );
        let (_, right_operand_claim) = accumulator.get_virtual_polynomial_opening(
            VirtualPolynomial::RightLookupOperand,
            SumcheckId::InstructionClaimReduction,
        );
        rv_claim + self.gamma * left_operand_claim + self.gamma_sqr * right_operand_claim
    }

    fn degree(&self) -> usize {
        let n_virtual_ra_polys = LOG_K / self.ra_virtual_log_k_chunk;
        n_virtual_ra_polys + 2
    }

    fn normalize_opening_point(
        &self,
        challenges: &[<F as JoltField>::Challenge],
    ) -> OpeningPoint<BIG_ENDIAN, F> {
        let (r_address_prime, r_cycle_prime) = challenges.split_at(LOG_K);
        let r_cycle_prime = r_cycle_prime.iter().copied().rev().collect::<Vec<_>>();

        OpeningPoint::new([r_address_prime.to_vec(), r_cycle_prime].concat())
    }
}

/// Binds address variables first using prefix/suffix decomposition to aggregate, per cycle j,
///   Σ_k ra(k, j)·Val_j(k) and Σ_k ra(k, j)·RafVal_j(k),
#[derive(Allocative)]
pub struct InstructionReadRafSumcheckProver<F: JoltField> {
    /// The execution trace, shared via Arc for efficient access in cache_openings.
    #[allocative(skip)]
    trace: Arc<Vec<Cycle>>,

    /// Running list of sumcheck challenges r_j (address then cycle) in binding order.
    r: Vec<F::Challenge>,

    /// Precomputed lookup keys k (bit-packed) per cycle j.
    lookup_indices: Vec<LookupBits>,
    /// Indices of cycles grouped by selected lookup table; used to form per-table flags.
    lookup_indices_by_table: Vec<Vec<usize>>,
    /// Per-cycle flag: instruction uses interleaved operands.
    is_interleaved_operands: Vec<bool>,

    /// Prefix checkpoints for each registered `Prefix` variant, updated every two rounds.
    prefix_checkpoints: Vec<PrefixCheckpoint<F>>,
    /// For each lookup table, dense polynomials holding suffix contributions in the current phase.
    suffix_polys: Vec<Vec<DensePolynomial<F>>>,
    /// Expanding tables accumulating address-prefix products per phase.
    v: Vec<ExpandingTable<F>>,
    /// u_evals for read-checking and RAF: eq(r_reduction,j).
    u_evals: Vec<F>,

    /// Registry holding prefix checkpoint values for `PrefixSuffixDecomposition` instances.
    prefix_registry: PrefixRegistry<F>,
    /// Prefix-suffix decomposition for right operand identity polynomial family.
    right_operand_ps: PrefixSuffixDecomposition<F, 2>,
    /// Prefix-suffix decomposition for left operand identity polynomial family.
    left_operand_ps: PrefixSuffixDecomposition<F, 2>,
    /// Prefix-suffix decomposition for the instruction-identity path (RAF flag path).
    identity_ps: PrefixSuffixDecomposition<F, 2>,

    /// Gruen-split equality polynomial over cycle vars. Present only in the last log(T) rounds.
    eq_r_reduction: GruenSplitEqPolynomial<F>,

    /// Materialized `ra_i(k_i, j)` polynomials. Present only in the last log(T) rounds.
    ra_polys: Option<Vec<MultilinearPolynomial<F>>>,

    /// Materialized Val_j(k) + γ · RafVal_j(k) over (address, cycle) for final log T rounds.
    /// Combines lookup table values with γ-weighted RAF operand contributions.
    combined_val_polynomial: Option<MultilinearPolynomial<F>>,

    #[allocative(skip)]
    params: InstructionReadRafSumcheckParams<F>,
}

impl<F: JoltField> InstructionReadRafSumcheckProver<F> {
    /// Creates a prover-side instance for the Read+RAF batched sumcheck.
    ///
    /// Builds prover-side working state:
    /// - Precomputes per-cycle lookup index, interleaving flags, and table choices
    /// - Buckets cycles by table and by path (interleaved vs identity)
    /// - Allocates per-table suffix accumulators and u-evals for rv/raf parts
    /// - Instantiates the three RAF decompositions and Gruen EQs over cycles
    #[tracing::instrument(skip_all, name = "InstructionReadRafSumcheckProver::initialize")]
    pub fn initialize(params: InstructionReadRafSumcheckParams<F>, trace: Arc<Vec<Cycle>>) -> Self {
        let log_T = trace.len().log_2();

        let log_m = LOG_K / params.phases;
        let right_operand_poly = OperandPolynomial::new(LOG_K, OperandSide::Right);
        let left_operand_poly = OperandPolynomial::new(LOG_K, OperandSide::Left);
        let identity_poly = IdentityPolynomial::new(LOG_K);
        let span = tracing::span!(tracing::Level::INFO, "Init PrefixSuffixDecomposition");
        let _guard = span.enter();
        let right_operand_ps =
            PrefixSuffixDecomposition::new(Box::new(right_operand_poly), log_m, LOG_K);
        let left_operand_ps =
            PrefixSuffixDecomposition::new(Box::new(left_operand_poly), log_m, LOG_K);
        let identity_ps = PrefixSuffixDecomposition::new(Box::new(identity_poly), log_m, LOG_K);
        drop(_guard);
        drop(span);

        let num_tables = LookupTables::<XLEN>::COUNT;

        let span = tracing::span!(tracing::Level::INFO, "Build cycle_data");
        let _guard = span.enter();
        struct CycleData<const XLEN: usize> {
            idx: usize,
            lookup_index: LookupBits,
            is_interleaved: bool,
            table: Option<LookupTables<XLEN>>,
        }

        let cycle_data: Vec<CycleData<XLEN>> = trace
            .par_iter()
            .enumerate()
            .map(|(idx, cycle)| {
                let bits = LookupBits::new(LookupQuery::<XLEN>::to_lookup_index(cycle), LOG_K);
                let is_interleaved = cycle
                    .instruction()
                    .circuit_flags()
                    .is_interleaved_operands();
                let table = cycle.lookup_table();

                CycleData {
                    idx,
                    lookup_index: bits,
                    is_interleaved,
                    table,
                }
            })
            .collect();
        drop(_guard);
        drop(span);

        let span = tracing::span!(tracing::Level::INFO, "Extract vectors");
        let _guard = span.enter();
        // Extract all vectors in parallel using par_extend
        let mut lookup_indices = Vec::with_capacity(cycle_data.len());
        let mut is_interleaved_operands = Vec::with_capacity(cycle_data.len());

        {
            let span = tracing::span!(tracing::Level::INFO, "par_extend basic vectors");
            let _guard = span.enter();
            lookup_indices.par_extend(cycle_data.par_iter().map(|data| data.lookup_index));
            is_interleaved_operands
                .par_extend(cycle_data.par_iter().map(|data| data.is_interleaved));
        }

        // Build lookup_indices_by_table fully in parallel
        // Create a vector for each table in parallel
        let lookup_indices_by_table: Vec<Vec<usize>> = (0..num_tables)
            .into_par_iter()
            .map(|t_idx| {
                // Each table gets its own parallel collection
                cycle_data
                    .par_iter()
                    .filter_map(|data| {
                        data.table.and_then(|t| {
                            if LookupTables::<XLEN>::enum_index(&t) == t_idx {
                                Some(data.idx)
                            } else {
                                None
                            }
                        })
                    })
                    .collect()
            })
            .collect();
        drop_in_background_thread(cycle_data);
        drop(_guard);
        drop(span);

        let suffix_polys: Vec<Vec<DensePolynomial<F>>> = LookupTables::<XLEN>::iter()
            .collect::<Vec<_>>()
            .par_iter()
            .map(|table| {
                table
                    .suffixes()
                    .par_iter()
                    .map(|_| DensePolynomial::default()) // Will be properly initialized in `init_phase`
                    .collect()
            })
            .collect();

        // Build split-eq polynomials and u_evals.
        let span = tracing::span!(tracing::Level::INFO, "Compute u_evals");
        let _guard = span.enter();
        let eq_poly_r_reduction =
            GruenSplitEqPolynomial::<F>::new(&params.r_reduction.r, BindingOrder::LowToHigh);
        let u_evals = EqPolynomial::evals(&params.r_reduction.r);
        drop(_guard);
        drop(span);

        let mut res = Self {
            trace,
            r: Vec::with_capacity(log_T + LOG_K),
            lookup_indices,

            // Prefix-suffix state (first log(K) rounds)
            lookup_indices_by_table,
            is_interleaved_operands,
            prefix_checkpoints: vec![None.into(); Prefixes::COUNT],
            suffix_polys,
            v: (0..params.phases)
                .map(|_| ExpandingTable::new(1 << log_m, BindingOrder::HighToLow))
                .collect(),
            u_evals,
            right_operand_ps,
            left_operand_ps,
            identity_ps,

            // State for last log(T) rounds
            ra_polys: None,
            eq_r_reduction: eq_poly_r_reduction,
            prefix_registry: PrefixRegistry::new(),
            combined_val_polynomial: None,
            params,
        };
        res.init_phase(0);
        res
    }

    /// To be called in the beginning of each phase, before any binding
    /// Phase initialization for address-binding:
    /// - Condenses prior-phase u-evals through the expanding-table v[phase-1]
    /// - Builds Q for RAF (Left/Right dual and Identity) from cycle buckets
    /// - Refreshes per-table read-checking suffix polynomials for this phase
    /// - Initializes/caches P via the shared `PrefixRegistry`
    /// - Resets the current expanding table accumulator for this phase
    #[tracing::instrument(skip_all, name = "InstructionReadRafProver::init_phase")]
    fn init_phase(&mut self, phase: usize) {
        let log_m = LOG_K / self.params.phases;
        let m = 1 << log_m;
        let m_mask = m - 1;
        // Condensation
        if phase != 0 {
            let span = tracing::span!(tracing::Level::INFO, "Update u_evals");
            let _guard = span.enter();
            self.lookup_indices
                .par_iter()
                .zip(&mut self.u_evals)
                .for_each(|(k, u_eval)| {
                    let (prefix, _) = k.split((self.params.phases - phase) * log_m);
                    let k_bound = prefix & m_mask;
                    *u_eval *= self.v[phase - 1][k_bound];
                });
        }

        PrefixSuffixDecomposition::init_Q_raf(
            &mut self.left_operand_ps,
            &mut self.right_operand_ps,
            &mut self.identity_ps,
            &self.u_evals,
            &self.lookup_indices,
            &self.is_interleaved_operands,
        );

        self.init_suffix_polys(phase);

        self.identity_ps.init_P(&mut self.prefix_registry);
        self.right_operand_ps.init_P(&mut self.prefix_registry);
        self.left_operand_ps.init_P(&mut self.prefix_registry);

        self.v[phase].reset(F::one());
    }

    /// Recomputes per-table suffix accumulators for the current phase of read-checking.
    ///
    /// For each lookup table's suffix family, this function:
    /// 1. Partitions cycles by their current chunk value (the `log_m`-bit segment
    ///    extracted from each cycle's lookup index for this phase).
    /// 2. Aggregates weighted contributions `u_evals[j] * suffix_mle(suffix_bits)`
    ///    into dense MLEs of size `M = 2^{log_m}`.
    ///
    /// # Suffix classification
    ///
    /// Suffixes are classified into three categories for efficient accumulation:
    /// - **`Suffixes::One`**: Always evaluates to 1; we simply accumulate `u_evals[j]`.
    /// - **{0,1}-valued suffixes**: Add `u_evals[j]` only when `suffix_mle == 1`.
    /// - **General suffixes**: Multiply `u_evals[j]` by `suffix_mle` value.
    #[tracing::instrument(skip_all, name = "InstructionReadRafProver::init_suffix_polys")]
    fn init_suffix_polys(&mut self, phase: usize) {
        /// Maximum number of suffixes any lookup table can have.
        /// (Currently `ValidSignedRemainderTable` has the most with 5.)
        const MAX_SUFFIXES: usize = 5;

        let log_m = LOG_K / self.params.phases;
        let m = 1 << log_m;
        let m_mask = m - 1;
        let num_threads = rayon::current_num_threads();
        let chunk_size = self.lookup_indices.len().div_ceil(num_threads).max(1);

        let new_suffix_polys: Vec<_> = {
            LookupTables::<XLEN>::iter()
                .collect::<Vec<_>>()
                .par_iter()
                .zip(self.lookup_indices_by_table.par_iter())
                .map(|(table, lookup_indices)| {
                    let suffixes = table.suffixes();
                    let num_suffixes = suffixes.len();
                    debug_assert!(num_suffixes <= MAX_SUFFIXES);

                    // Early exit: if no cycles use this table, return zero polynomials
                    if lookup_indices.is_empty() {
                        return vec![unsafe_allocate_zero_vec(m); num_suffixes];
                    }

                    // Pre-partition suffixes using fixed-size arrays to avoid heap allocation.
                    // Also track `Suffixes::One` separately to avoid per-cycle match check.
                    let mut suffix_one_idx: Option<usize> = None;
                    let mut suffix_01_indices = [0usize; MAX_SUFFIXES];
                    let mut suffix_01_count = 0usize;
                    let mut suffix_other_indices = [0usize; MAX_SUFFIXES];
                    let mut suffix_other_count = 0usize;

                    for (s_idx, suffix) in suffixes.iter().enumerate() {
                        if matches!(suffix, Suffixes::One) {
                            suffix_one_idx = Some(s_idx);
                        } else if suffix.is_01_valued() {
                            suffix_01_indices[suffix_01_count] = s_idx;
                            suffix_01_count += 1;
                        } else {
                            suffix_other_indices[suffix_other_count] = s_idx;
                            suffix_other_count += 1;
                        }
                    }

                    let unreduced_polys = lookup_indices
                        .par_chunks(chunk_size)
                        .map(|chunk| {
                            // Single allocation for all suffix accumulators:
                            // layout: [suffix_0 | suffix_1 | ... | suffix_{num_suffixes-1}],
                            // each suffix segment has length `m`.
                            let total_len = num_suffixes * m;
                            let mut chunk_result: Vec<F::Unreduced<6>> =
                                unsafe_allocate_zero_vec(total_len);

                            for j in chunk {
                                let k = self.lookup_indices[*j];
                                let (prefix_bits, suffix_bits) =
                                    k.split((self.params.phases - 1 - phase) * log_m);
                                let idx = prefix_bits & m_mask;
                                let u = self.u_evals[*j];

                                // Suffixes::One always evaluates to 1, so just add u directly.
                                if let Some(one_idx) = suffix_one_idx {
                                    chunk_result[one_idx * m + idx] += *u.as_unreduced_ref();
                                }

                                // Other {0,1}-valued suffixes: add u when suffix_mle == 1.
                                for i in 0..suffix_01_count {
                                    let s_idx = suffix_01_indices[i];
                                    let t = suffixes[s_idx].suffix_mle::<XLEN>(suffix_bits);
                                    debug_assert!(t == 0 || t == 1);
                                    if t == 1 {
                                        chunk_result[s_idx * m + idx] += *u.as_unreduced_ref();
                                    }
                                }

                                // General suffixes: multiply by t.
                                for i in 0..suffix_other_count {
                                    let s_idx = suffix_other_indices[i];
                                    let t = suffixes[s_idx].suffix_mle::<XLEN>(suffix_bits);
                                    if t != 0 {
                                        chunk_result[s_idx * m + idx] += u.mul_u64_unreduced(t);
                                    }
                                }
                            }

                            chunk_result
                        })
                        .reduce(
                            || unsafe_allocate_zero_vec(num_suffixes * m),
                            |mut acc, new| {
                                // Merge accumulator vectors (parallelize over the flat buffer)
                                acc.par_iter_mut()
                                    .zip(new.par_iter())
                                    .for_each(|(a, b)| *a += b);
                                acc
                            },
                        );

                    // Reduce the unreduced values to field elements (parallelized over suffixes)
                    (0..num_suffixes)
                        .into_par_iter()
                        .map(|s_idx| {
                            let start = s_idx * m;
                            let end = start + m;
                            unreduced_polys[start..end]
                                .iter()
                                .copied()
                                .map(F::from_barrett_reduce)
                                .collect::<Vec<F>>()
                        })
                        .collect::<Vec<_>>()
                })
                .collect()
        };

        // Replace existing suffix polynomials
        self.suffix_polys
            .iter_mut()
            .zip(new_suffix_polys.into_iter())
            .for_each(|(old, new)| {
                old.iter_mut()
                    .zip(new.into_iter())
                    .for_each(|(poly, mut coeffs)| {
                        *poly = DensePolynomial::new(std::mem::take(&mut coeffs));
                    });
            });
    }

    /// To be called before the last log(T) rounds
    /// Handoff between address and cycle rounds:
    /// - Materializes all virtual ra_i(k_i,j) from expanding tables across all phases
    /// - Commits prefix checkpoints into a fixed `PrefixEval` vector
    /// - Materializes Val_j(k) from table prefixes/suffixes
    /// - Materializes RafVal_j(k) from (Left,Right,Identity) prefixes with γ-weights
    /// - Converts ra/Val/RafVal into MultilinearPolynomial over (addr,cycle)
    #[tracing::instrument(skip_all, name = "InstructionReadRafProver::init_log_t_rounds")]
    fn init_log_t_rounds(&mut self, gamma: F, gamma_sqr: F) {
        let log_m = LOG_K / self.params.phases;
        let m = 1 << log_m;
        let m_mask = m - 1;
        let num_cycles = self.lookup_indices.len();
        // Drop stuff that's no longer needed
        drop_in_background_thread(std::mem::take(&mut self.u_evals));

        let ra_polys: Vec<MultilinearPolynomial<F>> = {
            let span = tracing::span!(tracing::Level::INFO, "Materialize ra polynomials");
            let _guard = span.enter();
            assert!(self.v.len().is_power_of_two());
            let n = LOG_K / self.params.ra_virtual_log_k_chunk;
            let chunk_size = self.v.len() / n;

            #[cfg(feature = "zolt-debug")]
            {
                eprintln!("[JOLT RA_DEBUG] phases={}, n={}, chunk_size={}, log_m={}",
                    self.params.phases, n, chunk_size, log_m);
                // Print lookup index for cycle 0
                if !self.lookup_indices.is_empty() {
                    let v: u128 = self.lookup_indices[0].into();
                    eprintln!("[JOLT RA_DEBUG] Cycle 0 lookup_index: 0x{:032x}", v);
                    eprintln!("[JOLT RA_DEBUG]   k_hi=0x{:016x}, k_lo=0x{:016x}", (v >> 64) as u64, v as u64);
                }
                // Print expanding table values for first few entries
                for (phase, table) in self.v.iter().enumerate() {
                    if phase < 4 {
                        use ark_serialize::CanonicalSerialize;
                        let mut bytes0 = [0u8; 32];
                        let mut bytes1 = [0u8; 32];
                        table[0].serialize_compressed(&mut bytes0[..]).ok();
                        table[1].serialize_compressed(&mut bytes1[..]).ok();
                        eprintln!("[JOLT RA_DEBUG] expanding_table[{}]: v[0]={:02x?}, v[1]={:02x?}",
                            phase, &bytes0[..16], &bytes1[..16]);
                    }
                }
            }

            self.v
                .chunks(chunk_size)
                .enumerate()
                .map(|(chunk_i, v_chunk)| {
                    let phase_offset = chunk_i * chunk_size;
                    let res = self
                        .lookup_indices
                        .par_iter()
                        .enumerate()
                        .with_min_len(1024)
                        .map(|(j, i)| {
                            // Hot path: compute ra_i(k_i, j) as a product of per-phase expanding-table
                            // values. This is performance sensitive, so we:
                            // - Convert `LookupBits` -> `u128` once per cycle
                            // - Use a decrementing shift instead of recomputing `(phases-1-phase)*log_m`
                            // - Avoid an initial multiply-by-one by seeding `acc` with the first term
                            let v: u128 = (*i).into();

                            if v_chunk.is_empty() {
                                return F::one();
                            }

                            // shift(phase) = (phases - 1 - phase) * log_m
                            // For consecutive phases, this decreases by `log_m` each step.
                            let mut shift = (self.params.phases - 1 - phase_offset) * log_m;

                            let mut iter = v_chunk.iter();
                            let first = iter.next().unwrap();
                            let first_idx = ((v >> shift) as usize) & m_mask;
                            let mut acc = first[first_idx];

                            #[cfg(feature = "zolt-debug")]
                            if j == 0 && chunk_i < 2 {
                                use ark_serialize::CanonicalSerialize;
                                let mut bytes = [0u8; 32];
                                acc.serialize_compressed(&mut bytes[..]).ok();
                                eprintln!("[JOLT RA_DEBUG] j=0, chunk={}, phase_offset={}: shift={}, first_idx=0x{:x}, table_val={:02x?}",
                                    chunk_i, phase_offset, shift, first_idx, &bytes[..16]);
                            }

                            for (table_idx, table) in iter.enumerate() {
                                shift -= log_m;
                                let idx = ((v >> shift) as usize) & m_mask;
                                acc *= table[idx];

                                #[cfg(feature = "zolt-debug")]
                                if j == 0 && chunk_i < 2 {
                                    use ark_serialize::CanonicalSerialize;
                                    let mut bytes = [0u8; 32];
                                    acc.serialize_compressed(&mut bytes[..]).ok();
                                    eprintln!("[JOLT RA_DEBUG] j=0, chunk={}, table_idx={}: shift={}, idx=0x{:x}, acc={:02x?}",
                                        chunk_i, table_idx + 1, shift, idx, &bytes[..16]);
                                }
                            }

                            #[cfg(feature = "zolt-debug")]
                            if j == 0 && chunk_i < 2 {
                                use ark_serialize::CanonicalSerialize;
                                let mut bytes = [0u8; 32];
                                acc.serialize_compressed(&mut bytes[..]).ok();
                                eprintln!("[JOLT RA_DEBUG] j=0, chunk={}: final ra_val={:02x?}", chunk_i, &bytes[..16]);
                            }

                            acc
                        })
                        .collect::<Vec<F>>();
                    res.into()
                })
                .collect()
        };

        drop_in_background_thread(std::mem::take(&mut self.v));

        let prefixes: Vec<PrefixEval<F>> = std::mem::take(&mut self.prefix_checkpoints)
            .into_iter()
            .map(|checkpoint| checkpoint.unwrap())
            .collect();
        // Materialize combined_val_poly = Val_j(k) + γ·RafVal_j(k)
        // combining lookup table values with RAF operand contributions in a single pass.
        let mut combined_val_poly: Vec<F> = unsafe_allocate_zero_vec(num_cycles);
        {
            let span = tracing::span!(tracing::Level::INFO, "Materialize combined_val_poly");
            let _guard = span.enter();
            let left_prefix = self.prefix_registry.checkpoints[Prefix::LeftOperand].unwrap();
            let right_prefix = self.prefix_registry.checkpoints[Prefix::RightOperand].unwrap();
            let identity_prefix = self.prefix_registry.checkpoints[Prefix::Identity].unwrap();
            let raf_interleaved = gamma * left_prefix + gamma_sqr * right_prefix;
            let raf_identity = gamma_sqr * identity_prefix;

            // At this point we've finished all LOG_K address rounds, so the lookup-table suffix
            // variable set is empty. That means every suffix MLE is evaluated on an empty bitstring,
            // and `table.combine(&prefixes, &suffixes)` becomes a per-table constant that can be
            // precomputed once (instead of allocating a suffix Vec per cycle).
            let empty_suffix_bits = LookupBits::new(0, 0);
            let table_values_at_r_addr: Vec<F> = LookupTables::<XLEN>::iter()
                .map(|table| {
                    let suffix_evals: Vec<F> = table
                        .suffixes()
                        .iter()
                        .map(|suffix| {
                            // Suffix MLEs are u64-valued; convert once here.
                            F::from_u64(suffix.suffix_mle::<XLEN>(empty_suffix_bits))
                        })
                        .collect();
                    table.combine(&prefixes, &suffix_evals)
                })
                .collect();

            combined_val_poly
                .par_iter_mut()
                .zip(self.trace.par_iter())
                .zip(std::mem::take(&mut self.is_interleaved_operands))
                .for_each(|((val, cycle), is_interleaved_operands)| {
                    // Add lookup table value (Val_j(k)) - derive table from trace
                    if let Some(table) = cycle.lookup_table() {
                        let t_idx = LookupTables::<XLEN>::enum_index(&table);
                        *val += table_values_at_r_addr[t_idx];
                    }
                    // Add RAF operand contribution (γ·RafVal_j(k))
                    if is_interleaved_operands {
                        *val += raf_interleaved;
                    } else {
                        *val += raf_identity;
                    }
                });

            #[cfg(feature = "zolt-debug")]
            {
                use ark_serialize::CanonicalSerialize;
                // Print RAF values used
                let mut raf_i_bytes = [0u8; 32];
                let mut raf_id_bytes = [0u8; 32];
                raf_interleaved.serialize_compressed(&mut raf_i_bytes[..]).ok();
                raf_identity.serialize_compressed(&mut raf_id_bytes[..]).ok();
                eprintln!("[JOLT INIT_LOG_T] raf_interleaved = {:02x?}", &raf_i_bytes[..16]);
                eprintln!("[JOLT INIT_LOG_T] raf_identity = {:02x?}", &raf_id_bytes[..16]);

                // Print prefix checkpoints
                let mut lp_bytes = [0u8; 32];
                let mut rp_bytes = [0u8; 32];
                let mut ip_bytes = [0u8; 32];
                left_prefix.serialize_compressed(&mut lp_bytes[..]).ok();
                right_prefix.serialize_compressed(&mut rp_bytes[..]).ok();
                identity_prefix.serialize_compressed(&mut ip_bytes[..]).ok();
                eprintln!("[JOLT INIT_LOG_T] left_prefix = {:02x?}", &lp_bytes[..16]);
                eprintln!("[JOLT INIT_LOG_T] right_prefix = {:02x?}", &rp_bytes[..16]);
                eprintln!("[JOLT INIT_LOG_T] identity_prefix = {:02x?}", &ip_bytes[..16]);

                // Print first 5 combined_val and table_values
                for j in 0..std::cmp::min(5, combined_val_poly.len()) {
                    let mut cv_bytes = [0u8; 32];
                    combined_val_poly[j].serialize_compressed(&mut cv_bytes[..]).ok();
                    eprintln!("[JOLT INIT_LOG_T] j={}: combined_val={:02x?}", j, &cv_bytes[..16]);
                }

                // Print table_values_at_r_addr
                for (t_idx, tv) in table_values_at_r_addr.iter().enumerate() {
                    let mut tv_bytes = [0u8; 32];
                    tv.serialize_compressed(&mut tv_bytes[..]).ok();
                    if *tv != F::zero() {
                        eprintln!("[JOLT INIT_LOG_T] table_values_at_r_addr[{}] = {:02x?}", t_idx, &tv_bytes[..16]);
                    }
                }
            }
        }

        self.combined_val_polynomial = Some(MultilinearPolynomial::from(combined_val_poly));
        self.ra_polys = Some(ra_polys);

        #[cfg(feature = "zolt-debug")]
        {
            use ark_serialize::CanonicalSerialize;
            // Compute materialized sum: Σ_j eq(j, r_reduction) * Π_c ra_c(j) * combined_val(j)
            let eq_evals = EqPolynomial::evals(&self.params.r_reduction.r);
            let ra_polys_ref = self.ra_polys.as_ref().unwrap();
            let combined_val_ref = self.combined_val_polynomial.as_ref().unwrap();
            let num_cycles = combined_val_ref.len();
            let mut materialized_sum = F::zero();
            let mut sum_no_ra = F::zero();
            for j in 0..num_cycles {
                let eq_j = eq_evals[j];
                let cv_j = combined_val_ref.get_bound_coeff(j);
                let mut ra_product = F::one();
                for chunk in ra_polys_ref.iter() {
                    ra_product *= chunk.get_bound_coeff(j);
                }
                materialized_sum += eq_j * ra_product * cv_j;
                sum_no_ra += eq_j * cv_j;
            }
            let mut ms_bytes = [0u8; 32];
            let mut snr_bytes = [0u8; 32];
            materialized_sum.serialize_compressed(&mut ms_bytes[..]).ok();
            sum_no_ra.serialize_compressed(&mut snr_bytes[..]).ok();
            eprintln!("[JOLT INIT_LOG_T] materialized_sum (WITH ra_weights) = {:02x?}", &ms_bytes[..16]);
            eprintln!("[JOLT INIT_LOG_T] sum_no_ra (WITHOUT ra_weights) = {:02x?}", &snr_bytes[..16]);
            eprintln!("[JOLT INIT_LOG_T] num_cycles={}", num_cycles);
        }

        // After the address rounds are complete and we have materialized `ra_polys` and the
        // `combined_val_polynomial`, the following buffers are no longer needed for the remaining
        // log(T) cycle rounds:
        // - `lookup_indices` (used only to build `ra_polys` and to size `combined_val_poly`)
        // - `suffix_polys` (used only during the first LOG_K address rounds)
        drop_in_background_thread((
            std::mem::take(&mut self.lookup_indices),
            std::mem::take(&mut self.suffix_polys),
        ));
    }
}

impl<F: JoltField, T: Transcript> SumcheckInstanceProver<F, T>
    for InstructionReadRafSumcheckProver<F>
{
    fn get_params(&self) -> &dyn SumcheckInstanceParams<F> {
        &self.params
    }

    #[tracing::instrument(skip_all, name = "InstructionReadRafSumcheckProver::compute_message")]
    /// Produces the prover's degree-≤3 univariate for the current round.
    ///
    /// - For the first LOG_K rounds: returns two evaluations combining
    ///   read-checking and RAF prefix–suffix messages (at X∈{0,2}).
    /// - For the last log(T) rounds: uses Gruen-split EQ.
    fn compute_message(&mut self, round: usize, previous_claim: F) -> UniPoly<F> {
        if round < LOG_K {
            // Phase 1: First log(K) rounds
            self.compute_prefix_suffix_prover_message(round, previous_claim)
        } else {
            #[cfg(feature = "zolt-debug")]
            if round == LOG_K {
                use ark_serialize::CanonicalSerialize;
                let mut claim_bytes = [0u8; 32];
                previous_claim.serialize_compressed(&mut claim_bytes[..]).ok();
                eprintln!("[JOLT CYCLE R0] previous_claim (at start of cycle rounds) = {:02x?}", &claim_bytes[..16]);

                // Compute materialized_sum here to compare with claim
                let eq_evals = EqPolynomial::evals(&self.params.r_reduction.r);
                let ra_polys_ref = self.ra_polys.as_ref().unwrap();
                let combined_val_ref = self.combined_val_polynomial.as_ref().unwrap();
                let num_cycles = combined_val_ref.len();
                let mut mat_sum = F::zero();
                for j in 0..num_cycles {
                    let eq_j = eq_evals[j];
                    let cv_j = combined_val_ref.get_bound_coeff(j);
                    let mut ra_prod = F::one();
                    for chunk in ra_polys_ref.iter() {
                        ra_prod *= chunk.get_bound_coeff(j);
                    }
                    mat_sum += eq_j * ra_prod * cv_j;
                }
                let match_claim = mat_sum == previous_claim;
                let mut ms_bytes2 = [0u8; 32];
                mat_sum.serialize_compressed(&mut ms_bytes2[..]).ok();
                eprintln!("[JOLT CYCLE R0] materialized_sum = {:02x?}", &ms_bytes2[..16]);
                eprintln!("[JOLT CYCLE R0] materialized_sum == previous_claim: {}", match_claim);
            }
            let ra_polys = self.ra_polys.as_ref().unwrap();
            let combined_val = self.combined_val_polynomial.as_ref().unwrap();
            let n_evals = ra_polys.len() + 1;

            let mut sum_evals = self
                .eq_r_reduction
                .E_out_current()
                .par_iter()
                .enumerate()
                .map(|(j_out, e_out)| {
                    // Each pair is a linear polynomial.
                    let mut pairs = vec![(F::zero(), F::zero()); n_evals];
                    let mut evals_acc = vec![F::Unreduced::<9>::zero(); n_evals];

                    for (j_in, e_in) in self.eq_r_reduction.E_in_current().iter().enumerate() {
                        let j = self.eq_r_reduction.group_index(j_out, j_in);

                        let Some((val_pair, ra_pairs)) = pairs.split_first_mut() else {
                            unreachable!()
                        };

                        let v_at_0 = combined_val.get_bound_coeff(2 * j);
                        let v_at_1 = combined_val.get_bound_coeff(2 * j + 1);
                        // Load linear poly: eq * combined_val.
                        *val_pair = (*e_in * v_at_0, *e_in * v_at_1);
                        // Load ra polys.
                        zip(ra_pairs, ra_polys).for_each(|(pair, ra_poly)| {
                            let eval_at_0 = ra_poly.get_bound_coeff(2 * j);
                            let eval_at_1 = ra_poly.get_bound_coeff(2 * j + 1);
                            *pair = (eval_at_0, eval_at_1);
                        });

                        // TODO: Use unreduced arithmetic in eval_linear_prod_assign.
                        eval_linear_prod_accumulate(&pairs, &mut evals_acc);
                    }

                    evals_acc
                        .into_iter()
                        .map(|v| F::from_montgomery_reduce(v) * e_out)
                        .collect::<Vec<F>>()
                })
                .reduce(
                    || vec![F::zero(); n_evals],
                    |a, b| zip(a, b).map(|(a, b)| a + b).collect(),
                );

            let current_scalar = self.eq_r_reduction.get_current_scalar();
            sum_evals.iter_mut().for_each(|v| *v *= current_scalar);

            #[cfg(feature = "zolt-debug")]
            {
                use ark_serialize::CanonicalSerialize;
                let cycle_round = round - LOG_K;
                eprintln!("[JOLT PROVER CYCLE] Round {} (cycle {}):", round, cycle_round);

                // Print sum_evals before finishing
                eprintln!("  sum_evals (after current_scalar mul):");
                for (i, eval) in sum_evals.iter().enumerate().take(4) {
                    let mut bytes = [0u8; 32];
                    eval.serialize_compressed(&mut bytes[..]).ok();
                    eprintln!("    [{}]: {:02x?}", i, &bytes[..16]);
                }

                // Print current_scalar
                let mut cs_bytes = [0u8; 32];
                current_scalar.serialize_compressed(&mut cs_bytes[..]).ok();
                eprintln!("  current_scalar: {:02x?}", &cs_bytes[..16]);

                // Print r_round (from eq_r_reduction)
                let r_round = self.eq_r_reduction.get_current_w();
                let mut r_bytes = [0u8; 32];
                <F::Challenge as CanonicalSerialize>::serialize_compressed(&r_round, &mut r_bytes[..]).ok();
                eprintln!("  r_round: {:02x?}", &r_bytes);

                // Print previous_claim
                let mut pc_bytes = [0u8; 32];
                previous_claim.serialize_compressed(&mut pc_bytes[..]).ok();
                eprintln!("  previous_claim: {:02x?}", &pc_bytes[..16]);
            }

            let poly = finish_mles_product_sum_from_evals(&sum_evals, previous_claim, &self.eq_r_reduction);

            #[cfg(feature = "zolt-debug")]
            {
                use ark_serialize::CanonicalSerialize;
                // Print resulting polynomial coefficients
                eprintln!("  poly coeffs (first 4):");
                for (i, coeff) in poly.coeffs.iter().enumerate().take(4) {
                    let mut bytes = [0u8; 32];
                    coeff.serialize_compressed(&mut bytes[..]).ok();
                    eprintln!("    c[{}]: {:02x?}", i, &bytes[..16]);
                }
            }

            poly
        }
    }

    #[tracing::instrument(skip_all, name = "InstructionReadRafSumcheckProver::ingest_challenge")]
    /// Binds the next variable (address or cycle) and advances state.
    ///
    /// Address rounds: bind all active prefix–suffix polynomials and the
    /// expanding-table accumulator; update checkpoints every two rounds;
    /// initialize next phase/handoff when needed. Cycle rounds: bind the ra/Val
    /// polynomials and Gruen EQ.
    fn ingest_challenge(&mut self, r_j: F::Challenge, round: usize) {
        let log_m = LOG_K / self.params.phases;
        self.r.push(r_j);
        if round < LOG_K {
            let phase = round / log_m;
            rayon::scope(|s| {
                s.spawn(|_| {
                    self.suffix_polys.par_iter_mut().for_each(|polys| {
                        polys
                            .par_iter_mut()
                            .for_each(|poly| poly.bind_parallel(r_j, BindingOrder::HighToLow))
                    });
                });
                s.spawn(|_| self.identity_ps.bind(r_j));
                s.spawn(|_| self.right_operand_ps.bind(r_j));
                s.spawn(|_| self.left_operand_ps.bind(r_j));
                s.spawn(|_| self.v[phase].update(r_j));
            });
            {
                if self.r.len().is_multiple_of(2) {
                    // Calculate suffix_len based on phases, using the same formula as original current_suffix_len
                    let suffix_len = LOG_K - (round / log_m + 1) * log_m;
                    Prefixes::update_checkpoints::<XLEN, F, F::Challenge>(
                        &mut self.prefix_checkpoints,
                        self.r[self.r.len() - 2],
                        self.r[self.r.len() - 1],
                        round,
                        suffix_len,
                    );
                }
            }

            // check if this is the last round in the phase
            if (round + 1).is_multiple_of(log_m) {
                self.prefix_registry.update_checkpoints();
                if phase != self.params.phases - 1 {
                    // if not last phase, init next phase
                    self.init_phase(phase + 1);
                }
            }

            if (round + 1) == LOG_K {
                self.init_log_t_rounds(self.params.gamma, self.params.gamma_sqr);
            }
        } else {
            // log(T) rounds

            self.eq_r_reduction.bind(r_j);
            self.combined_val_polynomial
                .as_mut()
                .unwrap()
                .bind_parallel(r_j, BindingOrder::LowToHigh);

            self.ra_polys
                .as_mut()
                .unwrap()
                .iter_mut()
                .for_each(|poly| poly.bind_parallel(r_j, BindingOrder::LowToHigh));
        }
    }

    #[tracing::instrument(skip_all, name = "ReadRafSumcheckProver::cache_openings")]
    fn cache_openings(
        &self,
        accumulator: &mut ProverOpeningAccumulator<F>,
        transcript: &mut T,
        sumcheck_challenges: &[F::Challenge],
    ) {
        let r_sumcheck = self.params.normalize_opening_point(sumcheck_challenges);
        // Prover publishes new virtual openings derived by this sumcheck:
        // - Per-table LookupTableFlag(i) at r_cycle
        // - InstructionRa at r_sumcheck (ra MLE's final claim)
        // - InstructionRafFlag at r_cycle
        let (r_address, r_cycle) = r_sumcheck.clone().split_at(LOG_K);

        // Compute flag claims using split-eq + unreduced accumulation for efficiency.
        // This avoids materializing the full eq table (size T) and instead uses
        // E_hi (size √T) and E_lo (size √T), iterating contiguously for cache locality.
        let (flag_claims, raf_flag_claim) = self.compute_flag_claims(&r_cycle);

        for (i, claim) in flag_claims.into_iter().enumerate() {
            accumulator.append_virtual(
                transcript,
                VirtualPolynomial::LookupTableFlag(i),
                SumcheckId::InstructionReadRaf,
                r_cycle.clone(),
                claim,
            );
        }

        let ra_polys = self.ra_polys.as_ref().unwrap();
        let mut r_address_chunks = r_address.r.chunks(LOG_K / ra_polys.len());
        for (i, ra_poly) in self.ra_polys.as_ref().unwrap().iter().enumerate() {
            let r_address = r_address_chunks.next().unwrap();
            let opening_point =
                OpeningPoint::<BIG_ENDIAN, F>::new([r_address, &*r_cycle.r].concat());
            let ra_claim = ra_poly.final_sumcheck_claim();
            #[cfg(feature = "zolt-debug")]
            {
                use ark_serialize::CanonicalSerialize;
                let mut bytes = [0u8; 32];
                ra_claim.serialize_compressed(&mut bytes[..]).ok();
                eprintln!("[JOLT PROVER] InstructionRa({}) final_sumcheck_claim (LE): {:02x?}", i, &bytes);
            }
            accumulator.append_virtual(
                transcript,
                VirtualPolynomial::InstructionRa(i),
                SumcheckId::InstructionReadRaf,
                opening_point,
                ra_claim,
            );
        }

        accumulator.append_virtual(
            transcript,
            VirtualPolynomial::InstructionRafFlag,
            SumcheckId::InstructionReadRaf,
            r_cycle.clone(),
            raf_flag_claim,
        );
    }

    #[cfg(feature = "allocative")]
    fn update_flamegraph(&self, flamegraph: &mut FlameGraphBuilder) {
        flamegraph.visit_root(self);
    }
}

impl<F: JoltField> InstructionReadRafSumcheckProver<F> {
    /// Address-round prover message: sum of read-checking and RAF components.
    ///
    /// Each component is a degree-2 univariate evaluated at X∈{0,2} using
    /// prefix–suffix decomposition, then added to form the batched message.
    fn compute_prefix_suffix_prover_message(&self, round: usize, previous_claim: F) -> UniPoly<F> {
        let mut read_checking = [F::zero(), F::zero()];
        let mut raf = [F::zero(), F::zero()];

        rayon::join(
            || {
                read_checking = self.prover_msg_read_checking(round);
            },
            || {
                raf = self.prover_msg_raf();
            },
        );

        let eval_at_0 = read_checking[0] + raf[0];
        let eval_at_2 = read_checking[1] + raf[1];

        #[cfg(feature = "zolt-debug")]
        {
            use ark_serialize::CanonicalSerialize;
            {
                let mut bytes_0 = [0u8; 32];
                let mut bytes_2 = [0u8; 32];
                let mut bytes_claim = [0u8; 32];
                let mut bytes_rc0 = [0u8; 32];
                let mut bytes_rc1 = [0u8; 32];
                let mut bytes_raf0 = [0u8; 32];
                let mut bytes_raf1 = [0u8; 32];
                eval_at_0.serialize_compressed(&mut bytes_0[..]).ok();
                eval_at_2.serialize_compressed(&mut bytes_2[..]).ok();
                previous_claim.serialize_compressed(&mut bytes_claim[..]).ok();
                read_checking[0].serialize_compressed(&mut bytes_rc0[..]).ok();
                read_checking[1].serialize_compressed(&mut bytes_rc1[..]).ok();
                raf[0].serialize_compressed(&mut bytes_raf0[..]).ok();
                raf[1].serialize_compressed(&mut bytes_raf1[..]).ok();
                let eval_at_1 = previous_claim - eval_at_0;
                let mut bytes_1 = [0u8; 32];
                eval_at_1.serialize_compressed(&mut bytes_1[..]).ok();
                eprintln!("[JOLT INST2 R{}] previous_claim = {:02x?}", round, &bytes_claim[..16]);
                eprintln!("[JOLT INST2 R{}] eval_at_0 = {:02x?}", round, &bytes_0[..16]);
                eprintln!("[JOLT INST2 R{}] eval_at_1 = {:02x?}", round, &bytes_1[..16]);
                eprintln!("[JOLT INST2 R{}] eval_at_2 = {:02x?}", round, &bytes_2[..16]);
                eprintln!("[JOLT INST2 R{}] read_checking = [{:02x?}, {:02x?}]", round, &bytes_rc0[..16], &bytes_rc1[..16]);
                eprintln!("[JOLT INST2 R{}] raf = [{:02x?}, {:02x?}]", round, &bytes_raf0[..16], &bytes_raf1[..16]);
                // Check multilinear: eval_2 should equal 2*eval_1 - eval_0 for degree-1 polynomial
                let expected_eval_2 = eval_at_1 + eval_at_1 - eval_at_0;
                let ml_match = eval_at_2 == expected_eval_2;
                if !ml_match {
                    let mut bytes_exp = [0u8; 32];
                    expected_eval_2.serialize_compressed(&mut bytes_exp[..]).ok();
                    eprintln!("[JOLT MULTILINEAR R{}] eval_2 != 2*eval_1 - eval_0! actual={:02x?} expected={:02x?}", round, &bytes_2[..16], &bytes_exp[..16]);
                }
            }
        }

        UniPoly::from_evals_and_hint(previous_claim, &[eval_at_0, eval_at_2])
    }

    /// RAF part for address rounds.
    ///
    /// Builds two evaluations at X∈{0,2} for the batched
    /// (Left + γ·Right) vs Identity path, folding γ-weights into the result.
    fn prover_msg_raf(&self) -> [F; 2] {
        let len = self.identity_ps.Q_len();
        let round = self.r.len();

        // Debug: print Q sums at round 0 and 1
        #[cfg(feature = "zolt-debug")]
        if round == 0 || round == 1 {
            use ark_serialize::CanonicalSerialize;
            fn to_bytes<F: CanonicalSerialize>(f: &F) -> Vec<u8> {
                let mut buf = vec![];
                f.serialize_compressed(&mut buf).unwrap();
                buf
            }

            // Compute Q sums for left, right, identity
            let mut left_q0_sum = F::zero();
            let mut left_q1_sum = F::zero();
            let mut right_q0_sum = F::zero();
            let mut right_q1_sum = F::zero();
            let mut identity_q0_sum = F::zero();
            let mut identity_q1_sum = F::zero();

            for i in 0..len {
                left_q0_sum += self.left_operand_ps.debug_Q(0)[i];
                left_q1_sum += self.left_operand_ps.debug_Q(1)[i];
                right_q0_sum += self.right_operand_ps.debug_Q(0)[i];
                right_q1_sum += self.right_operand_ps.debug_Q(1)[i];
                identity_q0_sum += self.identity_ps.debug_Q(0)[i];
                identity_q1_sum += self.identity_ps.debug_Q(1)[i];
            }

            let to_be_hex = |f: &F| -> String {
                let bytes = to_bytes(f);
                let mut s = String::new();
                for b in bytes.iter().rev().take(16) {
                    s.push_str(&format!("{:02x}", b));
                }
                s
            };

            eprintln!("[JOLT_RAF R{}] Q_SUM: left[0]={}, left[1]={}", round, to_be_hex(&left_q0_sum), to_be_hex(&left_q1_sum));
            eprintln!("[JOLT_RAF R{}] Q_SUM: right[0]={}, right[1]={}", round, to_be_hex(&right_q0_sum), to_be_hex(&right_q1_sum));
            eprintln!("[JOLT_RAF R{}] Q_SUM: identity[0]={}, identity[1]={}", round, to_be_hex(&identity_q0_sum), to_be_hex(&identity_q1_sum));
        }

        let [left_0, left_2, right_0, right_2] = (0..len / 2)
            .into_par_iter()
            .map(|b| {
                let (i0, i2) = self.identity_ps.sumcheck_evals(b);
                let (r0, r2) = self.right_operand_ps.sumcheck_evals(b);
                let (l0, l2) = self.left_operand_ps.sumcheck_evals(b);
                [
                    *l0.as_unreduced_ref(),
                    *l2.as_unreduced_ref(),
                    *(i0 + r0).as_unreduced_ref(),
                    *(i2 + r2).as_unreduced_ref(),
                ]
            })
            .fold_with([F::Unreduced::<5>::zero(); 4], |running, new| {
                [
                    running[0] + new[0],
                    running[1] + new[1],
                    running[2] + new[2],
                    running[3] + new[3],
                ]
            })
            .reduce(
                || [F::Unreduced::zero(); 4],
                |running, new| {
                    [
                        running[0] + new[0],
                        running[1] + new[1],
                        running[2] + new[2],
                        running[3] + new[3],
                    ]
                },
            );
        let raf_eval_0 = F::from_montgomery_reduce(
            left_0.mul_trunc::<4, 9>(self.params.gamma.as_unreduced_ref())
                + right_0.mul_trunc::<4, 9>(self.params.gamma_sqr.as_unreduced_ref()),
        );
        let raf_eval_2 = F::from_montgomery_reduce(
            left_2.mul_trunc::<4, 9>(self.params.gamma.as_unreduced_ref())
                + right_2.mul_trunc::<4, 9>(self.params.gamma_sqr.as_unreduced_ref()),
        );
        [raf_eval_0, raf_eval_2]
    }

    /// Read-checking part for address rounds.
    ///
    /// For each lookup table, evaluates Σ P(0)·Q^L, Σ P(2)·Q^L, Σ P(2)·Q^R via
    /// table-specific suffix families, then returns [g(0), g(2)] by the standard
    /// quadratic interpolation trick.
    fn prover_msg_read_checking(&self, j: usize) -> [F; 2] {
        let lookup_tables: Vec<_> = LookupTables::<XLEN>::iter().collect();

        let len = self.suffix_polys[0][0].len();
        let log_len = len.log_2();

        let r_x = if j % 2 == 1 {
            self.r.last().copied()
        } else {
            None
        };

        let [eval_0, eval_2_left, eval_2_right] = (0..len / 2)
            .into_par_iter()
            .flat_map_iter(|b| {
                let b = LookupBits::new(b as u128, log_len - 1);
                let prefixes_c0: Vec<_> = Prefixes::iter()
                    .map(|prefix| {
                        prefix.prefix_mle::<XLEN, F, F::Challenge>(
                            &self.prefix_checkpoints,
                            r_x,
                            0,
                            b,
                            j,
                        )
                    })
                    .collect();
                let prefixes_c2: Vec<_> = Prefixes::iter()
                    .map(|prefix| {
                        prefix.prefix_mle::<XLEN, F, F::Challenge>(
                            &self.prefix_checkpoints,
                            r_x,
                            2,
                            b,
                            j,
                        )
                    })
                    .collect();
                lookup_tables
                    .iter()
                    .zip(self.suffix_polys.iter())
                    .map(move |(table, suffixes)| {
                        let suffixes_left: Vec<_> =
                            suffixes.iter().map(|suffix| suffix[b.into()]).collect();
                        let suffixes_right: Vec<_> = suffixes
                            .iter()
                            .map(|suffix| suffix[usize::from(b) + len / 2])
                            .collect();
                        [
                            table.combine(&prefixes_c0, &suffixes_left),
                            table.combine(&prefixes_c2, &suffixes_left),
                            table.combine(&prefixes_c2, &suffixes_right),
                        ]
                    })
            })
            .fold_with([F::Unreduced::<5>::zero(); 3], |running, new| {
                [
                    running[0] + new[0].as_unreduced_ref(),
                    running[1] + new[1].as_unreduced_ref(),
                    running[2] + new[2].as_unreduced_ref(),
                ]
            })
            .reduce(
                || [F::Unreduced::zero(); 3],
                |running, new| {
                    [
                        running[0] + new[0],
                        running[1] + new[1],
                        running[2] + new[2],
                    ]
                },
            )
            .map(F::from_barrett_reduce);
        let result_eval_2 = eval_2_right + eval_2_right - eval_2_left;

        #[cfg(feature = "zolt-debug")]
        if j == 0 {
            use ark_serialize::CanonicalSerialize;
            // Per-table eval_0 and eval_2 at round 0 (sequential for debugging)
            let num_tables = lookup_tables.len();
            let mut per_table_eval_0 = vec![F::zero(); num_tables];
            let mut per_table_eval_2_left = vec![F::zero(); num_tables];
            let mut per_table_eval_2_right = vec![F::zero(); num_tables];

            for b_idx in 0..len / 2 {
                let b = LookupBits::new(b_idx as u128, log_len - 1);
                let prefixes_c0: Vec<_> = Prefixes::iter()
                    .map(|prefix| {
                        prefix.prefix_mle::<XLEN, F, F::Challenge>(
                            &self.prefix_checkpoints,
                            r_x,
                            0,
                            b,
                            j,
                        )
                    })
                    .collect();
                let prefixes_c2: Vec<_> = Prefixes::iter()
                    .map(|prefix| {
                        prefix.prefix_mle::<XLEN, F, F::Challenge>(
                            &self.prefix_checkpoints,
                            r_x,
                            2,
                            b,
                            j,
                        )
                    })
                    .collect();

                for (t_idx, (table, suffixes)) in lookup_tables.iter().zip(self.suffix_polys.iter()).enumerate() {
                    let suffixes_left: Vec<_> =
                        suffixes.iter().map(|suffix| suffix[b_idx]).collect();
                    let suffixes_right: Vec<_> = suffixes
                        .iter()
                        .map(|suffix| suffix[b_idx + len / 2])
                        .collect();
                    per_table_eval_0[t_idx] += table.combine(&prefixes_c0, &suffixes_left);
                    per_table_eval_2_left[t_idx] += table.combine(&prefixes_c2, &suffixes_left);
                    per_table_eval_2_right[t_idx] += table.combine(&prefixes_c2, &suffixes_right);
                }
            }

            for t_idx in 0..num_tables {
                let e0 = per_table_eval_0[t_idx];
                let e2 = per_table_eval_2_right[t_idx] + per_table_eval_2_right[t_idx] - per_table_eval_2_left[t_idx];
                if e0 != F::zero() || e2 != F::zero() {
                    let mut b0 = [0u8; 32];
                    let mut b2 = [0u8; 32];
                    let mut b2l = [0u8; 32];
                    let mut b2r = [0u8; 32];
                    e0.serialize_compressed(&mut b0[..]).ok();
                    e2.serialize_compressed(&mut b2[..]).ok();
                    per_table_eval_2_left[t_idx].serialize_compressed(&mut b2l[..]).ok();
                    per_table_eval_2_right[t_idx].serialize_compressed(&mut b2r[..]).ok();
                    eprintln!("[JOLT READ_CHECK R0] TABLE {} eval_0={:02x?}, eval_2={:02x?}, eval_2_left={:02x?}, eval_2_right={:02x?}",
                        t_idx, &b0[..16], &b2[..16], &b2l[..16], &b2r[..16]);
                }
            }
        }

        [eval_0, result_eval_2]
    }

    /// Compute per-table flag claims and RAF flag claim using split-eq + unreduced accumulation.
    ///
    /// For each lookup table i, computes: flag_claim[i] = Σ_{j: table[j] == i} eq(r_cycle, j)
    /// For RAF flag: raf_flag_claim = Σ_{j: identity path} eq(r_cycle, j)
    ///
    /// Uses split-eq optimization:
    /// - Split r_cycle into hi/lo halves, compute E_hi and E_lo (each size √T)
    /// - Parallelize over E_hi chunks (c_hi)
    /// - For each c_hi, iterate sequentially over c_lo for cache locality
    /// - Use unreduced 5-limb accumulation within each c_hi block
    #[tracing::instrument(skip_all, name = "ReadRafSumcheckProver::compute_flag_claims")]
    fn compute_flag_claims(&self, r_cycle: &OpeningPoint<BIG_ENDIAN, F>) -> (Vec<F>, F) {
        let T = self.trace.len();
        let num_tables = LookupTables::<XLEN>::COUNT;

        // Split-eq: divide r_cycle into MSB (hi) and LSB (lo) halves
        let log_T = r_cycle.len();
        let lo_bits = log_T / 2;
        let hi_bits = log_T - lo_bits;
        let (r_hi, r_lo) = r_cycle.r.split_at(hi_bits);

        let (E_hi, E_lo) = rayon::join(
            || EqPolynomial::<F>::evals(r_hi),
            || EqPolynomial::<F>::evals(r_lo),
        );

        let in_len = E_lo.len();

        // Parallel over E_hi chunks
        let num_threads = rayon::current_num_threads();
        let out_len = E_hi.len();
        let chunk_size = out_len.div_ceil(num_threads).max(1);

        E_hi.par_chunks(chunk_size)
            .enumerate()
            .map(|(chunk_idx, chunk)| {
                // Partial accumulators for this thread (field elements)
                let mut partial_flags: Vec<F> = vec![F::zero(); num_tables];
                let mut partial_raf: F = F::zero();

                let chunk_start = chunk_idx * chunk_size;
                for (local_idx, &e_hi) in chunk.iter().enumerate() {
                    let c_hi = chunk_start + local_idx;
                    let c_hi_base = c_hi * in_len;

                    // Local unreduced accumulators for this c_hi (5-limb)
                    let mut local_flags: Vec<F::Unreduced<5>> =
                        vec![F::Unreduced::<5>::zero(); num_tables];
                    let mut local_raf: F::Unreduced<5> = F::Unreduced::<5>::zero();

                    // Sequential over c_lo (contiguous cycles for this c_hi)
                    for c_lo in 0..in_len {
                        let j = c_hi_base + c_lo;
                        if j >= T {
                            break;
                        }

                        let cycle = &self.trace[j];
                        let e_lo_unreduced = *E_lo[c_lo].as_unreduced_ref();

                        // Accumulate table flag
                        if let Some(table) = cycle.lookup_table() {
                            let t_idx = LookupTables::<XLEN>::enum_index(&table);
                            local_flags[t_idx] += e_lo_unreduced;
                        }

                        // Accumulate RAF flag (identity = not interleaved)
                        if !cycle
                            .instruction()
                            .circuit_flags()
                            .is_interleaved_operands()
                        {
                            local_raf += e_lo_unreduced;
                        }
                    }

                    // Reduce and scale by e_hi
                    for t_idx in 0..num_tables {
                        let reduced = F::from_barrett_reduce::<5>(local_flags[t_idx]);
                        partial_flags[t_idx] += e_hi * reduced;
                    }
                    let raf_reduced = F::from_barrett_reduce::<5>(local_raf);
                    partial_raf += e_hi * raf_reduced;
                }

                (partial_flags, partial_raf)
            })
            .reduce(
                || (vec![F::zero(); num_tables], F::zero()),
                |(mut a_flags, a_raf), (b_flags, b_raf)| {
                    for (a, b) in a_flags.iter_mut().zip(b_flags.iter()) {
                        *a += *b;
                    }
                    (a_flags, a_raf + b_raf)
                },
            )
    }
}

/// Instruction lookups: batched Read + RAF sumcheck.
///
/// Let K = 2^{LOG_K}, T = 2^{log_T}. For random r_addr ∈ F^{LOG_K}, r_reduction ∈ F^{log_T},
/// this sumcheck proves that the accumulator claims
///   rv + γ·left_op + γ^2·right_op
/// equal the double sum over (j, k):
///   Σ_j Σ_k [ eq(j; r_reduction) · ra(k, j) · (Val_j(k) + γ·RafVal_j(k)) ].
/// It is implemented as: first log(K) address-binding rounds (prefix/suffix condensation), then
/// last log(T) cycle-binding rounds driven by [`GruenSplitEqPolynomial`].
pub struct InstructionReadRafSumcheckVerifier<F: JoltField> {
    params: InstructionReadRafSumcheckParams<F>,
}

impl<F: JoltField> InstructionReadRafSumcheckVerifier<F> {
    pub fn new(
        n_cycle_vars: usize,
        one_hot_params: &OneHotParams,
        opening_accumulator: &VerifierOpeningAccumulator<F>,
        transcript: &mut impl Transcript,
    ) -> Self {
        let params = InstructionReadRafSumcheckParams::new(
            n_cycle_vars,
            one_hot_params,
            opening_accumulator,
            transcript,
        );
        Self { params }
    }
}

impl<F: JoltField, T: Transcript> SumcheckInstanceVerifier<F, T>
    for InstructionReadRafSumcheckVerifier<F>
{
    fn get_params(&self) -> &dyn SumcheckInstanceParams<F> {
        &self.params
    }

    fn expected_output_claim(
        &self,
        accumulator: &VerifierOpeningAccumulator<F>,
        sumcheck_challenges: &[F::Challenge],
    ) -> F {
        // Verifier's RHS reconstruction from virtual claims at r:
        //
        // Computes Val and RafVal contributions at r_address, forms EQ(r_cycle)
        // for InstructionClaimReduction sumcheck, multiplies by ra claim at r_sumcheck,
        // and returns the batched identity RHS to be matched against the LHS input claim.
        #[cfg(feature = "zolt-debug")]
        {
            use ark_serialize::CanonicalSerialize;
            eprintln!("[InstructionReadRaf expected_output_claim] sumcheck_challenges.len={}", sumcheck_challenges.len());
            // Print last 6 challenges (cycle variables)
            let start = if sumcheck_challenges.len() > 6 { sumcheck_challenges.len() - 6 } else { 0 };
            for i in start..sumcheck_challenges.len() {
                let c = sumcheck_challenges[i];
                let c_f: F = c.into();
                let mut bytes = [0u8; 32];
                c_f.serialize_compressed(&mut bytes[..]).ok();
                eprintln!("  sumcheck_challenges[{}] hi16: {:02x?}", i, &bytes[16..]);
            }
        }
        let opening_point = self.params.normalize_opening_point(sumcheck_challenges);
        let (r_address_prime, r_cycle_prime) = opening_point.split_at(LOG_K);
        #[cfg(feature = "zolt-debug")]
        {
            use ark_serialize::CanonicalSerialize;
            eprintln!("[InstructionReadRaf expected_output_claim] after split:");
            eprintln!("  r_cycle_prime.len = {}", r_cycle_prime.len());
            for (i, r) in r_cycle_prime.r.iter().enumerate() {
                let r_f: F = (*r).into();
                let mut bytes = [0u8; 32];
                r_f.serialize_compressed(&mut bytes[..]).ok();
                eprintln!("  r_cycle_prime[{}] hi16: {:02x?}", i, &bytes[16..]);
            }
        }
        let left_operand_eval =
            OperandPolynomial::<F>::new(LOG_K, OperandSide::Left).evaluate(&r_address_prime.r);
        let right_operand_eval =
            OperandPolynomial::<F>::new(LOG_K, OperandSide::Right).evaluate(&r_address_prime.r);
        let identity_poly_eval = IdentityPolynomial::<F>::new(LOG_K).evaluate(&r_address_prime.r);
        let val_evals: Vec<_> = LookupTables::<XLEN>::iter()
            .map(|table| table.evaluate_mle::<F, F::Challenge>(&r_address_prime.r))
            .collect();
        #[cfg(feature = "zolt-debug")]
        {
            use ark_serialize::CanonicalSerialize;

            // Debug: print first few r_address_prime values
            eprintln!("[InstructionReadRaf] r_address_prime challenges (first 4 and last 4):");
            for i in 0..4 {
                let mut bytes = [0u8; 32];
                r_address_prime.r[i].serialize_compressed(&mut bytes[..]).ok();
                eprintln!("  r[{}] = {:02x?}", i, &bytes[16..]);
            }
            for i in 124..128 {
                let mut bytes = [0u8; 32];
                r_address_prime.r[i].serialize_compressed(&mut bytes[..]).ok();
                eprintln!("  r[{}] = {:02x?}", i, &bytes[16..]);
            }

            // Debug: print r[64] and r[65] which are used in first LowerWord update
            eprintln!("[InstructionReadRaf] r[64] and r[65] (first LowerWord update):");
            let mut bytes64 = [0u8; 32];
            r_address_prime.r[64].serialize_compressed(&mut bytes64[..]).ok();
            eprintln!("  r[64] = {:02x?}", &bytes64[16..]);
            let mut bytes65 = [0u8; 32];
            r_address_prime.r[65].serialize_compressed(&mut bytes65[..]).ok();
            eprintln!("  r[65] = {:02x?}", &bytes65[16..]);
            // Also print raw sumcheck_challenges[64]
            let mut raw_bytes64 = [0u8; 32];
            let raw_64: F = sumcheck_challenges[64].into();
            raw_64.serialize_compressed(&mut raw_bytes64[..]).ok();
            eprintln!("  sumcheck_challenges[64] (as F) = {:02x?}", &raw_bytes64[16..]);
            // Print the Challenge directly
            let mut raw_ch_bytes64 = [0u8; 32];
            let ch_64_as_f: F = r_address_prime.r[64].into();
            ch_64_as_f.serialize_compressed(&mut raw_ch_bytes64[..]).ok();
            eprintln!("  r_address_prime.r[64] (as F) = {:02x?}", &raw_ch_bytes64[16..]);

            eprintln!("[InstructionReadRaf] ALL Table MLE evaluations at r_address_prime:");
            for (i, eval) in val_evals.iter().enumerate() {
                let mut bytes = [0u8; 32];
                eval.serialize_compressed(&mut bytes[..]).ok();
                // Only print tables used by collatz: 0, 1, 2, 6, 11, 21, 26
                if i == 0 || i == 1 || i == 2 || i == 6 || i == 11 || i == 21 || i == 26 {
                    eprintln!("  table[{}] val_eval(FULL 32 LE)={:02x?}", i, &bytes);
                }
            }
        }

        let r_reduction = accumulator
            .get_virtual_polynomial_opening(
                VirtualPolynomial::LookupOutput,
                SumcheckId::InstructionClaimReduction,
            )
            .0
            .r;
        #[cfg(feature = "zolt-debug")]
        {
            use ark_serialize::CanonicalSerialize;
            eprintln!("[InstructionReadRaf] r_reduction (all 8 elements):");
            for (i, r) in r_reduction.iter().enumerate() {
                let mut bytes = [0u8; 32];
                r.serialize_compressed(&mut bytes[..]).ok();
                eprintln!("  r_reduction[{}] = {:02x?}", i, &bytes[16..]);
            }
            eprintln!("[InstructionReadRaf] r_cycle_prime (all 8 elements):");
            for (i, r) in r_cycle_prime.r.iter().enumerate() {
                let mut bytes = [0u8; 32];
                r.serialize_compressed(&mut bytes[..]).ok();
                eprintln!("  r_cycle_prime[{}] = {:02x?}", i, &bytes[16..]);
            }
        }
        let eq_eval_r_reduction = EqPolynomial::<F>::mle(&r_reduction, &r_cycle_prime.r);

        let n_virtual_ra_polys = LOG_K / self.params.ra_virtual_log_k_chunk;
        let ra_claims: Vec<F> = (0..n_virtual_ra_polys)
            .map(|i| {
                accumulator
                    .get_virtual_polynomial_opening(
                        VirtualPolynomial::InstructionRa(i),
                        SumcheckId::InstructionReadRaf,
                    )
                    .1
            })
            .collect();
        #[cfg(feature = "zolt-debug")]
        {
            use ark_serialize::CanonicalSerialize;
            eprintln!("[InstructionReadRaf] ra_chunk claims (FULL 32 bytes LE):");
            for (i, claim) in ra_claims.iter().enumerate() {
                let mut bytes = [0u8; 32];
                claim.serialize_compressed(&mut bytes[..]).ok();
                eprintln!("  ra_claims[{}] = {:02x?}", i, &bytes);
            }
        }
        let ra_claim: F = ra_claims.clone().into_iter().product();
        #[cfg(feature = "zolt-debug")]
        {
            use ark_serialize::CanonicalSerialize;
            let mut ra_product_bytes = [0u8; 32];
            ra_claim.serialize_compressed(&mut ra_product_bytes[..]).ok();
            eprintln!("[InstructionReadRaf] ra_claim PRODUCT (FULL 32 bytes LE) = {:02x?}", &ra_product_bytes);
        }

        let table_flag_claims: Vec<F> = (0..LookupTables::<XLEN>::COUNT)
            .map(|i| {
                accumulator
                    .get_virtual_polynomial_opening(
                        VirtualPolynomial::LookupTableFlag(i),
                        SumcheckId::InstructionReadRaf,
                    )
                    .1
            })
            .collect();
        #[cfg(feature = "zolt-debug")]
        {
            use ark_serialize::CanonicalSerialize;
            eprintln!("[InstructionReadRaf] Non-zero table_flag claims:");
            for (i, claim) in table_flag_claims.iter().enumerate() {
                if !claim.is_zero() {
                    let mut bytes = [0u8; 32];
                    claim.serialize_compressed(&mut bytes[..]).ok();
                    eprintln!("  table_flag[{}] = {:02x?}", i, &bytes[16..]);
                }
            }
        }

        let raf_flag_claim = accumulator
            .get_virtual_polynomial_opening(
                VirtualPolynomial::InstructionRafFlag,
                SumcheckId::InstructionReadRaf,
            )
            .1;
        #[cfg(feature = "zolt-debug")]
        {
            use ark_serialize::CanonicalSerialize;
            let mut bytes = [0u8; 32];
            raf_flag_claim.serialize_compressed(&mut bytes[..]).ok();
            eprintln!("[InstructionReadRaf] raf_flag_claim = {:02x?}", &bytes);
        }

        let val_claim = val_evals
            .into_iter()
            .zip(table_flag_claims)
            .map(|(claim, val)| claim * val)
            .sum::<F>();

        let raf_claim = (F::one() - raf_flag_claim)
            * (left_operand_eval + self.params.gamma * right_operand_eval)
            + raf_flag_claim * self.params.gamma * identity_poly_eval;

        #[cfg(feature = "zolt-debug")]
        {
            use ark_serialize::CanonicalSerialize;
            fn to_bytes<F: CanonicalSerialize>(f: &F) -> Vec<u8> {
                let mut buf = vec![];
                f.serialize_compressed(&mut buf).unwrap();
                buf
            }
            eprintln!("InstructionReadRaf expected_output_claim debug:");
            eprintln!("  left_operand_eval:  {:02x?}", &to_bytes(&left_operand_eval)[..16]);
            eprintln!("  right_operand_eval: {:02x?}", &to_bytes(&right_operand_eval)[..16]);
            eprintln!("  identity_poly_eval: {:02x?}", &to_bytes(&identity_poly_eval)[..16]);
            eprintln!("  gamma:              {:02x?}", &to_bytes(&self.params.gamma)[..16]);
            eprintln!("  eq_eval_r_reduction: {:02x?}", &to_bytes(&eq_eval_r_reduction)[..16]);
            eprintln!("  ra_claim:           {:02x?}", &to_bytes(&ra_claim)[..16]);
            eprintln!("  raf_flag_claim:     {:02x?}", &to_bytes(&raf_flag_claim)[..16]);
            eprintln!("  raf_claim:          {:02x?}", &to_bytes(&raf_claim)[..16]);
            eprintln!("  val_claim FULL LE:  {:02x?}", &to_bytes(&val_claim));
            eprintln!("  raf_claim FULL LE:  {:02x?}", &to_bytes(&raf_claim));
            eprintln!("  eq_eval FULL LE:    {:02x?}", &to_bytes(&eq_eval_r_reduction));
            eprintln!("  ra_claim FULL LE:   {:02x?}", &to_bytes(&ra_claim));
            eprintln!("  r_reduction[0]:     {:02x?}", &to_bytes(&r_reduction[0]));
            eprintln!("  r_cycle_prime[0]:   {:02x?}", &to_bytes(&r_cycle_prime.r[0]));
            eprintln!("  r_reduction.len():  {}", r_reduction.len());
            eprintln!("  r_cycle_prime.len():{}", r_cycle_prime.r.len());
            let final_result = eq_eval_r_reduction * ra_claim * (val_claim + self.params.gamma * raf_claim);
            eprintln!("  final_result FULL LE:  {:02x?}", &to_bytes(&final_result));
        }

        eq_eval_r_reduction * ra_claim * (val_claim + self.params.gamma * raf_claim)
    }

    fn cache_openings(
        &self,
        accumulator: &mut VerifierOpeningAccumulator<F>,
        transcript: &mut T,
        sumcheck_challenges: &[F::Challenge],
    ) {
        let r_sumcheck = self.params.normalize_opening_point(sumcheck_challenges);
        // Verifier requests the virtual openings that the prover must provide
        // for this sumcheck (same set as published by the prover-side cache).
        let (r_address, r_cycle) = r_sumcheck.split_at(LOG_K);

        (0..LookupTables::<XLEN>::COUNT).for_each(|i| {
            accumulator.append_virtual(
                transcript,
                VirtualPolynomial::LookupTableFlag(i),
                SumcheckId::InstructionReadRaf,
                r_cycle.clone(),
            );
        });

        for (i, r_address_chunk) in r_address
            .r
            .chunks(self.params.ra_virtual_log_k_chunk)
            .enumerate()
        {
            let opening_point =
                OpeningPoint::<BIG_ENDIAN, F>::new([r_address_chunk, &*r_cycle.r].concat());
            accumulator.append_virtual(
                transcript,
                VirtualPolynomial::InstructionRa(i),
                SumcheckId::InstructionReadRaf,
                opening_point,
            );
        }

        accumulator.append_virtual(
            transcript,
            VirtualPolynomial::InstructionRafFlag,
            SumcheckId::InstructionReadRaf,
            r_cycle.clone(),
        );
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::subprotocols::sumcheck::BatchedSumcheck;
    use crate::transcripts::Blake2bTranscript;
    use ark_bn254::Fr;
    use ark_std::Zero;
    use rand::{rngs::StdRng, RngCore, SeedableRng};
    use strum::IntoEnumIterator;
    use tracer::instruction::Cycle;

    const LOG_T: usize = 8;
    const T: usize = 1 << LOG_T;

    fn random_instruction(rng: &mut StdRng, instruction: &Option<Cycle>) -> Cycle {
        let instruction = instruction.unwrap_or_else(|| {
            let index = rng.next_u64() as usize % Cycle::COUNT;
            Cycle::iter()
                .enumerate()
                .filter(|(i, _)| *i == index)
                .map(|(_, x)| x)
                .next()
                .unwrap()
        });

        match instruction {
            Cycle::ADD(cycle) => cycle.random(rng).into(),
            Cycle::ADDI(cycle) => cycle.random(rng).into(),
            Cycle::AND(cycle) => cycle.random(rng).into(),
            Cycle::ANDN(cycle) => cycle.random(rng).into(),
            Cycle::ANDI(cycle) => cycle.random(rng).into(),
            Cycle::AUIPC(cycle) => cycle.random(rng).into(),
            Cycle::BEQ(cycle) => cycle.random(rng).into(),
            Cycle::BGE(cycle) => cycle.random(rng).into(),
            Cycle::BGEU(cycle) => cycle.random(rng).into(),
            Cycle::BLT(cycle) => cycle.random(rng).into(),
            Cycle::BLTU(cycle) => cycle.random(rng).into(),
            Cycle::BNE(cycle) => cycle.random(rng).into(),
            Cycle::FENCE(cycle) => cycle.random(rng).into(),
            Cycle::JAL(cycle) => cycle.random(rng).into(),
            Cycle::JALR(cycle) => cycle.random(rng).into(),
            Cycle::LUI(cycle) => cycle.random(rng).into(),
            Cycle::LD(cycle) => cycle.random(rng).into(),
            Cycle::MUL(cycle) => cycle.random(rng).into(),
            Cycle::MULHU(cycle) => cycle.random(rng).into(),
            Cycle::OR(cycle) => cycle.random(rng).into(),
            Cycle::ORI(cycle) => cycle.random(rng).into(),
            Cycle::SLT(cycle) => cycle.random(rng).into(),
            Cycle::SLTI(cycle) => cycle.random(rng).into(),
            Cycle::SLTIU(cycle) => cycle.random(rng).into(),
            Cycle::SLTU(cycle) => cycle.random(rng).into(),
            Cycle::SUB(cycle) => cycle.random(rng).into(),
            Cycle::SD(cycle) => cycle.random(rng).into(),
            Cycle::XOR(cycle) => cycle.random(rng).into(),
            Cycle::XORI(cycle) => cycle.random(rng).into(),
            Cycle::VirtualAdvice(cycle) => cycle.random(rng).into(),
            Cycle::VirtualAssertEQ(cycle) => cycle.random(rng).into(),
            Cycle::VirtualAssertHalfwordAlignment(cycle) => cycle.random(rng).into(),
            Cycle::VirtualAssertWordAlignment(cycle) => cycle.random(rng).into(),
            Cycle::VirtualAssertLTE(cycle) => cycle.random(rng).into(),
            Cycle::VirtualAssertValidDiv0(cycle) => cycle.random(rng).into(),
            Cycle::VirtualAssertValidUnsignedRemainder(cycle) => cycle.random(rng).into(),
            Cycle::VirtualMovsign(cycle) => cycle.random(rng).into(),
            Cycle::VirtualMULI(cycle) => cycle.random(rng).into(),
            Cycle::VirtualPow2(cycle) => cycle.random(rng).into(),
            Cycle::VirtualPow2I(cycle) => cycle.random(rng).into(),
            Cycle::VirtualPow2W(cycle) => cycle.random(rng).into(),
            Cycle::VirtualPow2IW(cycle) => cycle.random(rng).into(),
            Cycle::VirtualShiftRightBitmask(cycle) => cycle.random(rng).into(),
            Cycle::VirtualShiftRightBitmaskI(cycle) => cycle.random(rng).into(),
            Cycle::VirtualSRA(cycle) => cycle.random(rng).into(),
            Cycle::VirtualRev8W(cycle) => cycle.random(rng).into(),
            Cycle::VirtualSRAI(cycle) => cycle.random(rng).into(),
            Cycle::VirtualSRL(cycle) => cycle.random(rng).into(),
            Cycle::VirtualSRLI(cycle) => cycle.random(rng).into(),
            Cycle::VirtualZeroExtendWord(cycle) => cycle.random(rng).into(),
            Cycle::VirtualSignExtendWord(cycle) => cycle.random(rng).into(),
            Cycle::VirtualROTRI(cycle) => cycle.random(rng).into(),
            Cycle::VirtualROTRIW(cycle) => cycle.random(rng).into(),
            Cycle::VirtualChangeDivisor(cycle) => cycle.random(rng).into(),
            Cycle::VirtualChangeDivisorW(cycle) => cycle.random(rng).into(),
            Cycle::VirtualAssertMulUNoOverflow(cycle) => cycle.random(rng).into(),
            _ => Cycle::NoOp,
        }
    }

    fn test_read_raf_sumcheck(instruction: Option<Cycle>) {
        let mut rng = StdRng::seed_from_u64(12345);

        let trace: Arc<Vec<_>> = Arc::new(
            (0..T)
                .map(|_| random_instruction(&mut rng, &instruction))
                .collect(),
        );

        let prover_transcript = &mut Blake2bTranscript::new(&[]);
        let mut prover_opening_accumulator = ProverOpeningAccumulator::new(trace.len().log_2());
        let verifier_transcript = &mut Blake2bTranscript::new(&[]);
        let mut verifier_opening_accumulator = VerifierOpeningAccumulator::new(trace.len().log_2());

        let r_cycle: Vec<<Fr as JoltField>::Challenge> =
            prover_transcript.challenge_vector_optimized::<Fr>(LOG_T);
        let _r_cycle: Vec<<Fr as JoltField>::Challenge> =
            verifier_transcript.challenge_vector_optimized::<Fr>(LOG_T);
        let eq_r_cycle = EqPolynomial::<Fr>::evals(&r_cycle);

        let mut rv_claim = Fr::zero();
        let mut left_operand_claim = Fr::zero();
        let mut right_operand_claim = Fr::zero();

        for (i, cycle) in trace.iter().enumerate() {
            let lookup_index = LookupQuery::<XLEN>::to_lookup_index(cycle);
            let table: Option<LookupTables<XLEN>> = cycle.lookup_table();
            if let Some(table) = table {
                rv_claim +=
                    JoltField::mul_u64(&eq_r_cycle[i], table.materialize_entry(lookup_index));
            }

            // Compute left and right operand claims
            let (lo, ro) = LookupQuery::<XLEN>::to_lookup_operands(cycle);
            left_operand_claim += JoltField::mul_u64(&eq_r_cycle[i], lo);
            right_operand_claim += JoltField::mul_u128(&eq_r_cycle[i], ro);
        }

        prover_opening_accumulator.append_virtual(
            prover_transcript,
            VirtualPolynomial::LookupOutput,
            SumcheckId::InstructionClaimReduction,
            OpeningPoint::new(r_cycle.clone()),
            rv_claim,
        );
        prover_opening_accumulator.append_virtual(
            prover_transcript,
            VirtualPolynomial::LeftLookupOperand,
            SumcheckId::InstructionClaimReduction,
            OpeningPoint::new(r_cycle.clone()),
            left_operand_claim,
        );
        prover_opening_accumulator.append_virtual(
            prover_transcript,
            VirtualPolynomial::RightLookupOperand,
            SumcheckId::InstructionClaimReduction,
            OpeningPoint::new(r_cycle.clone()),
            right_operand_claim,
        );
        prover_opening_accumulator.append_virtual(
            prover_transcript,
            VirtualPolynomial::LookupOutput,
            SumcheckId::SpartanProductVirtualization,
            OpeningPoint::new(r_cycle.clone()),
            rv_claim,
        );

        let one_hot_params = OneHotParams::new(trace.len().log_2(), 100, 100);

        let params = InstructionReadRafSumcheckParams::new(
            trace.len().log_2(),
            &one_hot_params,
            &prover_opening_accumulator,
            prover_transcript,
        );
        let mut prover_sumcheck =
            InstructionReadRafSumcheckProver::initialize(params, Arc::clone(&trace));

        let (proof, r_sumcheck) = BatchedSumcheck::prove(
            vec![&mut prover_sumcheck],
            &mut prover_opening_accumulator,
            prover_transcript,
        );

        // Take claims
        for (key, (_, value)) in &prover_opening_accumulator.openings {
            let empty_point = OpeningPoint::<BIG_ENDIAN, Fr>::new(vec![]);
            verifier_opening_accumulator
                .openings
                .insert(*key, (empty_point, *value));
        }

        verifier_opening_accumulator.append_virtual(
            verifier_transcript,
            VirtualPolynomial::LookupOutput,
            SumcheckId::InstructionClaimReduction,
            OpeningPoint::new(r_cycle.clone()),
        );
        verifier_opening_accumulator.append_virtual(
            verifier_transcript,
            VirtualPolynomial::LeftLookupOperand,
            SumcheckId::InstructionClaimReduction,
            OpeningPoint::new(r_cycle.clone()),
        );
        verifier_opening_accumulator.append_virtual(
            verifier_transcript,
            VirtualPolynomial::RightLookupOperand,
            SumcheckId::InstructionClaimReduction,
            OpeningPoint::new(r_cycle.clone()),
        );
        verifier_opening_accumulator.append_virtual(
            verifier_transcript,
            VirtualPolynomial::LookupOutput,
            SumcheckId::SpartanProductVirtualization,
            OpeningPoint::new(r_cycle.clone()),
        );

        let mut verifier_sumcheck = InstructionReadRafSumcheckVerifier::new(
            trace.len().log_2(),
            &one_hot_params,
            &verifier_opening_accumulator,
            verifier_transcript,
        );

        let r_sumcheck_verif = BatchedSumcheck::verify(
            &proof,
            vec![&mut verifier_sumcheck],
            &mut verifier_opening_accumulator,
            verifier_transcript,
        )
        .unwrap();

        assert_eq!(r_sumcheck, r_sumcheck_verif);
    }

    #[test]
    fn test_random_instructions() {
        test_read_raf_sumcheck(None);
    }

    #[test]
    fn test_add() {
        test_read_raf_sumcheck(Some(Cycle::ADD(Default::default())));
    }

    #[test]
    fn test_addi() {
        test_read_raf_sumcheck(Some(Cycle::ADDI(Default::default())));
    }

    #[test]
    fn test_and() {
        test_read_raf_sumcheck(Some(Cycle::AND(Default::default())));
    }

    #[test]
    fn test_andn() {
        test_read_raf_sumcheck(Some(Cycle::ANDN(Default::default())));
    }

    #[test]
    fn test_andi() {
        test_read_raf_sumcheck(Some(Cycle::ANDI(Default::default())));
    }

    #[test]
    fn test_auipc() {
        test_read_raf_sumcheck(Some(Cycle::AUIPC(Default::default())));
    }

    #[test]
    fn test_beq() {
        test_read_raf_sumcheck(Some(Cycle::BEQ(Default::default())));
    }

    #[test]
    fn test_bge() {
        test_read_raf_sumcheck(Some(Cycle::BGE(Default::default())));
    }

    #[test]
    fn test_bgeu() {
        test_read_raf_sumcheck(Some(Cycle::BGEU(Default::default())));
    }

    #[test]
    fn test_blt() {
        test_read_raf_sumcheck(Some(Cycle::BLT(Default::default())));
    }

    #[test]
    fn test_bltu() {
        test_read_raf_sumcheck(Some(Cycle::BLTU(Default::default())));
    }

    #[test]
    fn test_bne() {
        test_read_raf_sumcheck(Some(Cycle::BNE(Default::default())));
    }

    #[test]
    fn test_fence() {
        test_read_raf_sumcheck(Some(Cycle::FENCE(Default::default())));
    }

    #[test]
    fn test_jal() {
        test_read_raf_sumcheck(Some(Cycle::JAL(Default::default())));
    }

    #[test]
    fn test_jalr() {
        test_read_raf_sumcheck(Some(Cycle::JALR(Default::default())));
    }

    #[test]
    fn test_lui() {
        test_read_raf_sumcheck(Some(Cycle::LUI(Default::default())));
    }

    #[test]
    fn test_ld() {
        test_read_raf_sumcheck(Some(Cycle::LD(Default::default())));
    }

    #[test]
    fn test_mul() {
        test_read_raf_sumcheck(Some(Cycle::MUL(Default::default())));
    }

    #[test]
    fn test_mulhu() {
        test_read_raf_sumcheck(Some(Cycle::MULHU(Default::default())));
    }

    #[test]
    fn test_or() {
        test_read_raf_sumcheck(Some(Cycle::OR(Default::default())));
    }

    #[test]
    fn test_ori() {
        test_read_raf_sumcheck(Some(Cycle::ORI(Default::default())));
    }

    #[test]
    fn test_slt() {
        test_read_raf_sumcheck(Some(Cycle::SLT(Default::default())));
    }

    #[test]
    fn test_slti() {
        test_read_raf_sumcheck(Some(Cycle::SLTI(Default::default())));
    }

    #[test]
    fn test_sltiu() {
        test_read_raf_sumcheck(Some(Cycle::SLTIU(Default::default())));
    }

    #[test]
    fn test_sltu() {
        test_read_raf_sumcheck(Some(Cycle::SLTU(Default::default())));
    }

    #[test]
    fn test_sub() {
        test_read_raf_sumcheck(Some(Cycle::SUB(Default::default())));
    }

    #[test]
    fn test_sd() {
        test_read_raf_sumcheck(Some(Cycle::SD(Default::default())));
    }

    #[test]
    fn test_xor() {
        test_read_raf_sumcheck(Some(Cycle::XOR(Default::default())));
    }

    #[test]
    fn test_xori() {
        test_read_raf_sumcheck(Some(Cycle::XORI(Default::default())));
    }

    #[test]
    fn test_advice() {
        test_read_raf_sumcheck(Some(Cycle::VirtualAdvice(Default::default())));
    }

    #[test]
    fn test_asserteq() {
        test_read_raf_sumcheck(Some(Cycle::VirtualAssertEQ(Default::default())));
    }

    #[test]
    fn test_asserthalfwordalignment() {
        test_read_raf_sumcheck(Some(Cycle::VirtualAssertHalfwordAlignment(
            Default::default(),
        )));
    }

    #[test]
    fn test_assertwordalignment() {
        test_read_raf_sumcheck(Some(Cycle::VirtualAssertWordAlignment(Default::default())));
    }

    #[test]
    fn test_assertlte() {
        test_read_raf_sumcheck(Some(Cycle::VirtualAssertLTE(Default::default())));
    }

    #[test]
    fn test_assertvaliddiv0() {
        test_read_raf_sumcheck(Some(Cycle::VirtualAssertValidDiv0(Default::default())));
    }

    #[test]
    fn test_assertvalidunsignedremainder() {
        test_read_raf_sumcheck(Some(Cycle::VirtualAssertValidUnsignedRemainder(
            Default::default(),
        )));
    }

    #[test]
    fn test_movsign() {
        test_read_raf_sumcheck(Some(Cycle::VirtualMovsign(Default::default())));
    }

    #[test]
    fn test_muli() {
        test_read_raf_sumcheck(Some(Cycle::VirtualMULI(Default::default())));
    }

    #[test]
    fn test_pow2() {
        test_read_raf_sumcheck(Some(Cycle::VirtualPow2(Default::default())));
    }

    #[test]
    fn test_pow2i() {
        test_read_raf_sumcheck(Some(Cycle::VirtualPow2I(Default::default())));
    }

    #[test]
    fn test_pow2w() {
        test_read_raf_sumcheck(Some(Cycle::VirtualPow2W(Default::default())));
    }

    #[test]
    fn test_pow2iw() {
        test_read_raf_sumcheck(Some(Cycle::VirtualPow2IW(Default::default())));
    }

    #[test]
    fn test_shiftrightbitmask() {
        test_read_raf_sumcheck(Some(Cycle::VirtualShiftRightBitmask(Default::default())));
    }

    #[test]
    fn test_shiftrightbitmaski() {
        test_read_raf_sumcheck(Some(Cycle::VirtualShiftRightBitmaskI(Default::default())));
    }

    #[test]
    fn test_virtualrotri() {
        test_read_raf_sumcheck(Some(Cycle::VirtualROTRI(Default::default())));
    }

    #[test]
    fn test_virtualrotriw() {
        test_read_raf_sumcheck(Some(Cycle::VirtualROTRIW(Default::default())));
    }

    #[test]
    fn test_virtualsra() {
        test_read_raf_sumcheck(Some(Cycle::VirtualSRA(Default::default())));
    }

    #[test]
    fn test_virtualsrai() {
        test_read_raf_sumcheck(Some(Cycle::VirtualSRAI(Default::default())));
    }

    #[test]
    fn test_virtualrev8w() {
        test_read_raf_sumcheck(Some(Cycle::VirtualRev8W(Default::default())));
    }

    #[test]
    fn test_virtualsrl() {
        test_read_raf_sumcheck(Some(Cycle::VirtualSRL(Default::default())));
    }

    #[test]
    fn test_virtualsrli() {
        test_read_raf_sumcheck(Some(Cycle::VirtualSRLI(Default::default())));
    }

    #[test]
    fn test_virtualextend() {
        test_read_raf_sumcheck(Some(Cycle::VirtualZeroExtendWord(Default::default())));
    }

    #[test]
    fn test_virtualsignextend() {
        test_read_raf_sumcheck(Some(Cycle::VirtualSignExtendWord(Default::default())));
    }

    #[test]
    fn test_virtualchangedivisor() {
        test_read_raf_sumcheck(Some(Cycle::VirtualChangeDivisor(Default::default())));
    }

    #[test]
    fn test_virtualchangedivisorw() {
        test_read_raf_sumcheck(Some(Cycle::VirtualChangeDivisorW(Default::default())));
    }

    #[test]
    fn test_virtualassertmulnooverflow() {
        test_read_raf_sumcheck(Some(Cycle::VirtualAssertMulUNoOverflow(Default::default())));
    }
}
