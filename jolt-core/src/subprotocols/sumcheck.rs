#![allow(clippy::too_many_arguments)]
#![allow(clippy::type_complexity)]

use crate::field::JoltField;
use crate::poly::opening_proof::{ProverOpeningAccumulator, VerifierOpeningAccumulator};
use crate::poly::unipoly::{CompressedUniPoly, UniPoly};
use crate::subprotocols::sumcheck_prover::SumcheckInstanceProver;
use crate::subprotocols::sumcheck_verifier::SumcheckInstanceVerifier;
use crate::transcripts::{AppendToTranscript, Transcript};
use crate::utils::errors::ProofVerifyError;
#[cfg(not(target_arch = "wasm32"))]
use crate::utils::profiling::print_current_memory_usage;

use ark_serialize::*;
use std::marker::PhantomData;

/// Implements the standard technique for batching parallel sumchecks to reduce
/// verifier cost and proof size.
///
/// For details, refer to Jim Posen's ["Perspectives on Sumcheck Batching"](https://hackmd.io/s/HyxaupAAA).
/// We do what they describe as "front-loaded" batch sumcheck.
pub enum BatchedSumcheck {}
impl BatchedSumcheck {
    pub fn prove<F: JoltField, ProofTranscript: Transcript>(
        mut sumcheck_instances: Vec<&mut dyn SumcheckInstanceProver<F, ProofTranscript>>,
        opening_accumulator: &mut ProverOpeningAccumulator<F>,
        transcript: &mut ProofTranscript,
    ) -> (SumcheckInstanceProof<F, ProofTranscript>, Vec<F::Challenge>) {
        let max_num_rounds = sumcheck_instances
            .iter()
            .map(|sumcheck| sumcheck.num_rounds())
            .max()
            .unwrap();

        // Append input claims to transcript
        sumcheck_instances.iter().for_each(|sumcheck| {
            let input_claim = sumcheck.input_claim(opening_accumulator);
            transcript.append_scalar(&input_claim);
        });

        let batching_coeffs: Vec<F> = transcript.challenge_vector(sumcheck_instances.len());

        // To see why we may need to scale by a power of two, consider a batch of
        // two sumchecks:
        //   claim_a = \sum_x P(x)             where x \in {0, 1}^M
        //   claim_b = \sum_{x, y} Q(x, y)     where x \in {0, 1}^M, y \in {0, 1}^N
        // Then the batched sumcheck is:
        //   \sum_{x, y} A * P(x) + B * Q(x, y)  where A and B are batching coefficients
        //   = A * \sum_y \sum_x P(x) + B * \sum_{x, y} Q(x, y)
        //   = A * \sum_y claim_a + B * claim_b
        //   = A * 2^N * claim_a + B * claim_b
        let mut individual_claims: Vec<F> = sumcheck_instances
            .iter()
            .map(|sumcheck| {
                let num_rounds = sumcheck.num_rounds();
                let input_claim = sumcheck.input_claim(opening_accumulator);
                input_claim.mul_pow_2(max_num_rounds - num_rounds)
            })
            .collect();

        #[cfg(test)]
        let mut batched_claim: F = individual_claims
            .iter()
            .zip(batching_coeffs.iter())
            .map(|(claim, coeff)| *claim * coeff)
            .sum();

        let mut r_sumcheck: Vec<F::Challenge> = Vec::with_capacity(max_num_rounds);
        let mut compressed_polys: Vec<CompressedUniPoly<F>> = Vec::with_capacity(max_num_rounds);
        let two_inv = F::from_u64(2).inverse().unwrap();

        for round in 0..max_num_rounds {
            #[cfg(not(target_arch = "wasm32"))]
            {
                let label = format!("Sumcheck round {round}");
                print_current_memory_usage(label.as_str());
            }

            let univariate_polys: Vec<UniPoly<F>> = sumcheck_instances
                .iter_mut()
                .zip(individual_claims.iter())
                .map(|(sumcheck, previous_claim)| {
                    let num_rounds = sumcheck.num_rounds();
                    let offset = sumcheck.round_offset(max_num_rounds);
                    let active = round >= offset && round < offset + num_rounds;
                    if active {
                        sumcheck.compute_message(round - offset, *previous_claim)
                    } else {
                        // Variable is "dummy" for this instance: polynomial is independent of it,
                        // so the round univariate is constant with H(0)=H(1)=previous_claim/2.
                        UniPoly::from_coeff(vec![*previous_claim * two_inv])
                    }
                })
                .collect();

            // Linear combination of individual univariate polynomials
            let batched_univariate_poly: UniPoly<F> =
                univariate_polys.iter().zip(&batching_coeffs).fold(
                    UniPoly::from_coeff(vec![]),
                    |mut batched_poly, (poly, &coeff)| {
                        batched_poly += &(poly * coeff);
                        batched_poly
                    },
                );

            let compressed_poly = batched_univariate_poly.compress();

            // append the prover's message to the transcript
            compressed_poly.append_to_transcript(transcript);
            let r_j = transcript.challenge_scalar_optimized::<F>();
            r_sumcheck.push(r_j);

            // Cache individual claims for this round
            individual_claims
                .iter_mut()
                .zip(univariate_polys.into_iter())
                .for_each(|(claim, poly)| *claim = poly.evaluate(&r_j));

            #[cfg(test)]
            {
                // Sanity check
                let h0 = batched_univariate_poly.evaluate::<F>(&F::zero());
                let h1 = batched_univariate_poly.evaluate::<F>(&F::one());
                assert_eq!(
                    h0 + h1,
                    batched_claim,
                    "round {round}: H(0) + H(1) = {h0} + {h1} != {batched_claim}"
                );
                batched_claim = batched_univariate_poly.evaluate(&r_j);
            }

            for sumcheck in sumcheck_instances.iter_mut() {
                let num_rounds = sumcheck.num_rounds();
                let offset = sumcheck.round_offset(max_num_rounds);
                let active = round >= offset && round < offset + num_rounds;
                if active {
                    sumcheck.ingest_challenge(r_j, round - offset);
                }
            }

            compressed_polys.push(compressed_poly);
        }

        // Allow each sumcheck instance to perform any end-of-protocol work (e.g. flushing
        // delayed bindings) after the final challenge has been ingested and before we cache
        // openings.
        for sumcheck in sumcheck_instances.iter_mut() {
            sumcheck.finalize();
        }

        let max_num_rounds = sumcheck_instances
            .iter()
            .map(|sumcheck| sumcheck.num_rounds())
            .max()
            .unwrap();

        for sumcheck in sumcheck_instances.iter() {
            // Instance-local slice can start at a custom global offset.
            let offset = sumcheck.round_offset(max_num_rounds);
            let r_slice = &r_sumcheck[offset..offset + sumcheck.num_rounds()];

            // Cache polynomial opening claims, to be proven using either an
            // opening proof or sumcheck (in the case of virtual polynomials).
            sumcheck.cache_openings(opening_accumulator, transcript, r_slice);
        }

        (SumcheckInstanceProof::new(compressed_polys), r_sumcheck)
    }

    pub fn verify<F: JoltField, ProofTranscript: Transcript>(
        proof: &SumcheckInstanceProof<F, ProofTranscript>,
        sumcheck_instances: Vec<&dyn SumcheckInstanceVerifier<F, ProofTranscript>>,
        opening_accumulator: &mut VerifierOpeningAccumulator<F>,
        transcript: &mut ProofTranscript,
    ) -> Result<Vec<F::Challenge>, ProofVerifyError> {
        let max_degree = sumcheck_instances
            .iter()
            .map(|sumcheck| sumcheck.degree())
            .max()
            .unwrap();
        let max_num_rounds = sumcheck_instances
            .iter()
            .map(|sumcheck| sumcheck.num_rounds())
            .max()
            .unwrap();

        // Append input claims to transcript
        #[cfg(feature = "zolt-debug")]
        if max_num_rounds == 8 {
            eprintln!("Stage 3 verifier - appending input claims (max_num_rounds={}):", max_num_rounds);
        }
        #[cfg(feature = "zolt-debug")]
        if max_num_rounds == 15 {
            eprintln!("Stage 4 verifier - appending input claims:");
        }
        #[cfg(feature = "zolt-debug")]
        if max_num_rounds == 24 {
            eprintln!("Stage 2 verifier - appending input claims (max_num_rounds={}):", max_num_rounds);
        }
        #[cfg(feature = "zolt-debug")]
        if max_num_rounds == 136 {
            eprintln!("Stage 5 verifier - appending input claims (max_num_rounds={}):", max_num_rounds);
        }
        sumcheck_instances.iter().enumerate().for_each(|(idx, sumcheck)| {
            let input_claim = sumcheck.input_claim(opening_accumulator);
            #[cfg(feature = "zolt-debug")]
            if max_num_rounds == 8 || max_num_rounds == 15 || max_num_rounds == 24 || max_num_rounds == 136 {
                use ark_serialize::CanonicalSerialize;
                let mut claim_bytes = [0u8; 32];
                input_claim.serialize_compressed(&mut claim_bytes[..]).ok();
                eprintln!("  instance[{}] input_claim: {:02x?}", idx, &claim_bytes);
            }
            transcript.append_scalar(&input_claim);
        });

        let batching_coeffs: Vec<F> = transcript.challenge_vector(sumcheck_instances.len());
        #[cfg(feature = "zolt-debug")]
        if max_num_rounds == 8 || max_num_rounds == 15 || max_num_rounds == 24 || max_num_rounds == 136 {
            let stage = if max_num_rounds == 8 { "Stage 3" } else if max_num_rounds == 15 { "Stage 4" } else { "Stage 2" };
            eprintln!("{} verifier - batching coeffs:", stage);
            for (idx, coeff) in batching_coeffs.iter().enumerate() {
                use ark_serialize::CanonicalSerialize;
                let mut coeff_bytes = [0u8; 32];
                coeff.serialize_compressed(&mut coeff_bytes[..]).ok();
                eprintln!("  coeff[{}]: {:02x?}", idx, &coeff_bytes);
            }
        }

        // To see why we may need to scale by a power of two, consider a batch of
        // two sumchecks:
        //   claim_a = \sum_x P(x)             where x \in {0, 1}^M
        //   claim_b = \sum_{x, y} Q(x, y)     where x \in {0, 1}^M, y \in {0, 1}^N
        // Then the batched sumcheck is:
        //   \sum_{x, y} A * P(x) + B * Q(x, y)  where A and B are batching coefficients
        //   = A * \sum_y \sum_x P(x) + B * \sum_{x, y} Q(x, y)
        //   = A * \sum_y claim_a + B * claim_b
        //   = A * 2^N * claim_a + B * claim_b
        let claim: F = sumcheck_instances
            .iter()
            .zip(batching_coeffs.iter())
            .map(|(sumcheck, coeff)| {
                let num_rounds = sumcheck.num_rounds();
                let input_claim = sumcheck.input_claim(opening_accumulator);
                input_claim.mul_pow_2(max_num_rounds - num_rounds) * coeff
            })
            .sum();

        #[cfg(feature = "zolt-debug")]
        {
            use ark_serialize::CanonicalSerialize;
            let mut claim_bytes = [0u8; 32];
            claim.serialize_compressed(&mut claim_bytes[..]).ok();
            eprintln!("Sumcheck starting (max_rounds={}):", max_num_rounds);
            eprintln!("  initial_claim: {:02x?}", &claim_bytes);
            eprintln!("  max_num_rounds: {}", max_num_rounds);
            eprintln!("  max_degree: {}", max_degree);
            // Print first 3 rounds' coefficients for Stage 3 (8 rounds)
            if max_num_rounds == 8 {
                for (round_idx, poly) in proof.compressed_polys.iter().take(3).enumerate() {
                    eprintln!("  Stage3 round {} coeffs_except_linear:", round_idx);
                    for (i, c) in poly.coeffs_except_linear_term.iter().enumerate() {
                        let mut bytes = [0u8; 32];
                        c.serialize_compressed(&mut bytes[..]).ok();
                        eprintln!("    [{}]: {:02x?}", i, &bytes);
                    }
                }
            }
            if let Some(first_poly) = proof.compressed_polys.first() {
                eprintln!("  first round coeffs_except_linear:");
                for (i, c) in first_poly.coeffs_except_linear_term.iter().enumerate() {
                    let mut bytes = [0u8; 32];
                    c.serialize_compressed(&mut bytes[..]).ok();
                    eprintln!("    [{}]: {:02x?}", i, &bytes);
                }
            }
        }

        let (output_claim, r_sumcheck) =
            proof.verify(claim, max_num_rounds, max_degree, transcript)?;

        let expected_output_claim = sumcheck_instances
            .iter()
            .zip(batching_coeffs.iter())
            .enumerate()
            .map(|(idx, (sumcheck, coeff))| {
                let offset = sumcheck.round_offset(max_num_rounds);
                let r_slice = &r_sumcheck[offset..offset + sumcheck.num_rounds()];

                // Cache polynomial opening claims, to be proven using either an
                // opening proof or sumcheck (in the case of virtual polynomials).
                sumcheck.cache_openings(opening_accumulator, transcript, r_slice);
                let claim = sumcheck.expected_output_claim(opening_accumulator, r_slice);

                #[cfg(feature = "zolt-debug")]
                if max_num_rounds == 24 || max_num_rounds == 15 || max_num_rounds == 136 {
                    use ark_serialize::CanonicalSerialize;
                    let stage = if max_num_rounds == 24 { "Stage 2" } else if max_num_rounds == 15 { "Stage 4" } else { "Stage 5" };
                    let mut claim_bytes = [0u8; 32];
                    claim.serialize_compressed(&mut claim_bytes[..]).ok();
                    let mut coeff_bytes = [0u8; 32];
                    coeff.serialize_compressed(&mut coeff_bytes[..]).ok();
                    let contribution = claim * coeff;
                    let mut contrib_bytes = [0u8; 32];
                    contribution.serialize_compressed(&mut contrib_bytes[..]).ok();
                    eprintln!("{} Instance {} expected_output_claim:", stage, idx);
                    eprintln!("  claim: {:02x?}", &claim_bytes);
                    eprintln!("  coeff: {:02x?}", &coeff_bytes);
                    eprintln!("  claim*coeff: {:02x?}", &contrib_bytes);
                }

                claim * coeff
            })
            .sum();

        if output_claim != expected_output_claim {
            #[cfg(feature = "zolt-debug")]
            {
                use ark_serialize::CanonicalSerialize;
                let mut out_bytes = [0u8; 32];
                let mut exp_bytes = [0u8; 32];
                output_claim.serialize_compressed(&mut out_bytes[..]).ok();
                expected_output_claim.serialize_compressed(&mut exp_bytes[..]).ok();
                eprintln!("Sumcheck verification failed!");
                eprintln!("  output_claim:   {:02x?}", &out_bytes);
                eprintln!("  expected_claim: {:02x?}", &exp_bytes);
            }
            return Err(ProofVerifyError::SumcheckVerificationError);
        }

        Ok(r_sumcheck)
    }
}

#[derive(CanonicalSerialize, CanonicalDeserialize, Debug, Clone)]
pub struct SumcheckInstanceProof<F: JoltField, ProofTranscript: Transcript> {
    pub compressed_polys: Vec<CompressedUniPoly<F>>,
    _marker: PhantomData<ProofTranscript>,
}

impl<F: JoltField, ProofTranscript: Transcript> SumcheckInstanceProof<F, ProofTranscript> {
    pub fn new(
        compressed_polys: Vec<CompressedUniPoly<F>>,
    ) -> SumcheckInstanceProof<F, ProofTranscript> {
        SumcheckInstanceProof {
            compressed_polys,
            _marker: PhantomData,
        }
    }

    /// Verify this sumcheck proof.
    /// Note: Verification does not execute the final check of sumcheck protocol: g_v(r_v) = oracle_g(r),
    /// as the oracle is not passed in. Expected that the caller will implement.
    ///
    /// Params
    /// - `claim`: Claimed evaluation
    /// - `num_rounds`: Number of rounds of sumcheck, or number of variables to bind
    /// - `degree_bound`: Maximum allowed degree of the combined univariate polynomial
    /// - `transcript`: Fiat-shamir transcript
    ///
    /// Returns (e, r)
    /// - `e`: Claimed evaluation at random point
    /// - `r`: Evaluation point
    pub fn verify(
        &self,
        claim: F,
        num_rounds: usize,
        degree_bound: usize,
        transcript: &mut ProofTranscript,
    ) -> Result<(F, Vec<F::Challenge>), ProofVerifyError> {
        let mut e = claim;
        let mut r: Vec<F::Challenge> = Vec::new();

        // verify that there is a univariate polynomial for each round
        assert_eq!(self.compressed_polys.len(), num_rounds);
        for i in 0..self.compressed_polys.len() {
            // verify degree bound
            if self.compressed_polys[i].degree() > degree_bound {
                return Err(ProofVerifyError::InvalidInputLength(
                    degree_bound,
                    self.compressed_polys[i].degree(),
                ));
            }

            // append the prover's message to the transcript
            self.compressed_polys[i].append_to_transcript(transcript);

            //derive the verifier's challenge for the next round
            let r_i: F::Challenge = transcript.challenge_scalar_optimized::<F>();
            r.push(r_i);

            // evaluate the claimed degree-ell polynomial at r_i using the hint
            e = self.compressed_polys[i].eval_from_hint(&e, &r_i);

            #[cfg(feature = "zolt-debug")]
            {
                use ark_serialize::CanonicalSerialize;
                // Print Stage 2 (24 rounds) - rounds 16-23 for InstructionClaimReduction
                if num_rounds == 24 && i >= 16 {
                    let mut r_bytes = [0u8; 32];
                    let r_field: F = r_i.into();
                    r_field.serialize_compressed(&mut r_bytes[..]).ok();
                    eprintln!("  Stage2 Round {} challenge: {:02x?}", i, &r_bytes);
                }
                // Print Stage 3 (8 rounds) - first 3 rounds only
                if num_rounds == 8 && i < 3 {
                    let mut r_bytes = [0u8; 32];
                    let r_field: F = r_i.into();
                    r_field.serialize_compressed(&mut r_bytes[..]).ok();
                    eprintln!("  Stage3 Round {} challenge: {:02x?}", i, &r_bytes[0..16]);
                }
                // Print Stage 4 (15 rounds) all iterations
                if num_rounds == 15 {
                    // Print round polynomial coefficients
                    eprintln!("  Round {} coeffs:", i);
                    for (j, c) in self.compressed_polys[i].coeffs_except_linear_term.iter().enumerate() {
                        let mut c_bytes = [0u8; 32];
                        c.serialize_compressed(&mut c_bytes[..]).ok();
                        eprintln!("    [{}]: {:02x?}", j, &c_bytes[0..16]);
                    }
                    let mut e_bytes = [0u8; 32];
                    let mut r_bytes = [0u8; 32];
                    e.serialize_compressed(&mut e_bytes[..]).ok();
                    let r_field: F = r_i.into();
                    r_field.serialize_compressed(&mut r_bytes[..]).ok();
                    eprintln!("  Round {} challenge: {:02x?}", i, &r_bytes);
                    eprintln!("  Round {} new_claim: {:02x?}", i, &e_bytes);
                }
                // Print Stage 5 (136 rounds) - last 8 rounds (cycle vars)
                if num_rounds == 136 && i >= 128 {
                    let mut r_bytes = [0u8; 32];
                    let r_field: F = r_i.into();
                    r_field.serialize_compressed(&mut r_bytes[..]).ok();
                    let mut e_bytes = [0u8; 32];
                    e.serialize_compressed(&mut e_bytes[..]).ok();
                    eprintln!("  Stage5 Round {} (cycle var {}): challenge: {:02x?}", i, i - 128, &r_bytes[0..16]);
                    eprintln!("  Stage5 Round {} new_claim: {:02x?}", i, &e_bytes[0..16]);
                }
            }
        }

        Ok((e, r))
    }
}
