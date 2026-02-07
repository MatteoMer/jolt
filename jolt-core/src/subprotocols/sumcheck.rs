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

            #[cfg(feature = "zolt-debug")]
            {
                // Print individual instance polynomials for Stage 5 rounds
                if max_num_rounds > 130 && (round < 3 || round == 128 || round == 129) {
                    use ark_serialize::CanonicalSerialize;
                    eprintln!("[JOLT PROVER BATCHED] Round {}:", round);
                    for (i, poly) in univariate_polys.iter().enumerate() {
                        eprintln!("  Instance {} poly coeffs (degree={}):", i, poly.degree());
                        for (j, coeff) in poly.coeffs.iter().enumerate() {
                            let mut bytes = [0u8; 32];
                            coeff.serialize_compressed(&mut bytes[..]).ok();
                            eprintln!("    c[{}]: {:02x?}", j, &bytes[..16]);
                        }
                    }
                    eprintln!("  Batched poly coeffs:");
                    for (j, coeff) in batched_univariate_poly.coeffs.iter().enumerate() {
                        let mut bytes = [0u8; 32];
                        coeff.serialize_compressed(&mut bytes[..]).ok();
                        eprintln!("    c[{}]: {:02x?}", j, &bytes[..16]);
                    }
                }
            }

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

            #[cfg(feature = "zolt-debug")]
            {
                // Track individual claims for Stage 5 at ALL rounds
                if max_num_rounds > 130 {
                    use ark_serialize::CanonicalSerialize;
                    // Print Instance 2 claim at every round (critical for debugging chain divergence)
                    if individual_claims.len() > 2 {
                        let mut claim_bytes = [0u8; 32];
                        individual_claims[2].serialize_compressed(&mut claim_bytes[..]).ok();
                        eprintln!("[S5 INST2 CLAIM R{}] {:02x?}", round, &claim_bytes[..16]);
                    }
                    // Print full details for first 3 and last few rounds
                    if round < 3 || round >= 127 {
                        let r_j_as_f: F = r_j.into();
                        let mut r_bytes = [0u8; 32];
                        r_j_as_f.serialize_compressed(&mut r_bytes[..]).ok();
                        eprintln!("[S5 CLAIM TRACKING R{}] challenge = {:02x?}", round, &r_bytes[..16]);
                        for (idx, claim) in individual_claims.iter().enumerate() {
                            let mut claim_bytes = [0u8; 32];
                            claim.serialize_compressed(&mut claim_bytes[..]).ok();
                            eprintln!("[S5 CLAIM TRACKING R{}] instance[{}]: {:02x?}", round, idx, &claim_bytes);
                        }
                    }
                }
            }

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
        if max_num_rounds == 8 || max_num_rounds == 6 {
            eprintln!("Stage 3 verifier - appending input claims (max_num_rounds={}):", max_num_rounds);
        }
        #[cfg(feature = "zolt-debug")]
        if max_num_rounds == 15 {
            eprintln!("Stage 4 verifier - appending input claims:");
            eprintln!("Stage 4 verifier - transcript state BEFORE input claims: {:02x?}", &transcript.debug_state());
        }
        #[cfg(feature = "zolt-debug")]
        if max_num_rounds == 24 {
            eprintln!("Stage 2 verifier - appending input claims (max_num_rounds={}):", max_num_rounds);
        }
        #[cfg(feature = "zolt-debug")]
        if max_num_rounds == 136 || max_num_rounds == 134 {
            eprintln!("Stage 5 verifier - appending input claims (max_num_rounds={}):", max_num_rounds);
        }
        sumcheck_instances.iter().enumerate().for_each(|(idx, sumcheck)| {
            let input_claim = sumcheck.input_claim(opening_accumulator);
            #[cfg(feature = "zolt-debug")]
            if max_num_rounds == 15 {
                use ark_serialize::CanonicalSerialize;
                let mut claim_bytes = [0u8; 32];
                input_claim.serialize_compressed(&mut claim_bytes[..]).ok();
                eprintln!("  Stage 4 instance[{}] input_claim: {:02x?}", idx, &claim_bytes);
                eprintln!("  Stage 4 transcript BEFORE appending instance[{}]: {:02x?}", idx, &transcript.debug_state()[0..8]);
            }
            #[cfg(feature = "zolt-debug")]
            if max_num_rounds == 8 || max_num_rounds == 6 || max_num_rounds == 24 || max_num_rounds == 136 || max_num_rounds == 134 {
                use ark_serialize::CanonicalSerialize;
                let mut claim_bytes = [0u8; 32];
                input_claim.serialize_compressed(&mut claim_bytes[..]).ok();
                eprintln!("  instance[{}] input_claim: {:02x?}", idx, &claim_bytes);
            }
            transcript.append_scalar(&input_claim);
            #[cfg(feature = "zolt-debug")]
            if max_num_rounds == 15 {
                eprintln!("  Stage 4 transcript AFTER appending instance[{}]: {:02x?}", idx, &transcript.debug_state()[0..8]);
            }
        });

        let batching_coeffs: Vec<F> = transcript.challenge_vector(sumcheck_instances.len());
        #[cfg(feature = "zolt-debug")]
        if max_num_rounds == 8 || max_num_rounds == 6 || max_num_rounds == 15 || max_num_rounds == 24 || max_num_rounds == 136 || max_num_rounds == 134 {
            let stage = if max_num_rounds == 8 || max_num_rounds == 6 { "Stage 3" } else if max_num_rounds == 15 { "Stage 4" } else if max_num_rounds == 136 || max_num_rounds == 134 { "Stage 5" } else { "Stage 2" };
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
            .enumerate()
            .map(|(idx, (sumcheck, coeff))| {
                let num_rounds = sumcheck.num_rounds();
                let input_claim = sumcheck.input_claim(opening_accumulator);
                let scaled = input_claim.mul_pow_2(max_num_rounds - num_rounds);
                let contribution = scaled * coeff;
                #[cfg(feature = "zolt-debug")]
                if max_num_rounds == 136 {
                    use ark_serialize::CanonicalSerialize;
                    let mut ic_bytes = [0u8; 32];
                    let mut sc_bytes = [0u8; 32];
                    let mut co_bytes = [0u8; 32];
                    input_claim.serialize_compressed(&mut ic_bytes[..]).ok();
                    scaled.serialize_compressed(&mut sc_bytes[..]).ok();
                    contribution.serialize_compressed(&mut co_bytes[..]).ok();
                    eprintln!("[S5 INIT] instance[{}]: num_rounds={}, scale_by={}", idx, num_rounds, max_num_rounds - num_rounds);
                    eprintln!("  input_claim: {:02x?}", &ic_bytes);
                    eprintln!("  scaled:      {:02x?}", &sc_bytes);
                    eprintln!("  contribution (scaled*coeff): {:02x?}", &co_bytes);
                }
                contribution
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
            // For Stage 5 (136 rounds), print first 3 rounds' coefficients
            if max_num_rounds == 136 || max_num_rounds == 134 {
                eprintln!("[STAGE5 DEBUG] First 3 rounds' coefficients:");
                for (round_idx, poly) in proof.compressed_polys.iter().take(3).enumerate() {
                    eprintln!("  Round {} (degree {}):", round_idx, poly.degree());
                    for (i, c) in poly.coeffs_except_linear_term.iter().enumerate() {
                        let mut bytes = [0u8; 32];
                        c.serialize_compressed(&mut bytes[..]).ok();
                        eprintln!("    coeff[{}] FULL: {:02x?}", i, &bytes);
                    }
                }
            }
        }

        #[cfg(feature = "zolt-debug")]
        if max_num_rounds == 13 {
            use ark_serialize::CanonicalSerialize;
            let mut claim_bytes = [0u8; 32];
            claim.serialize_compressed(&mut claim_bytes[..]).ok();
            eprintln!("[Stage 6] initial_claim: {:02x?}", &claim_bytes);
            eprintln!("[Stage 6] max_num_rounds={}, max_degree={}", max_num_rounds, max_degree);
            eprintln!("[Stage 6] num instances: {}", sumcheck_instances.len());
            for (i, si) in sumcheck_instances.iter().enumerate() {
                let ic = si.input_claim(opening_accumulator);
                let mut ic_bytes = [0u8; 32];
                ic.serialize_compressed(&mut ic_bytes[..]).ok();
                eprintln!("[Stage 6] instance[{}]: num_rounds={}, degree={}, input_claim={:02x?}",
                    i, si.num_rounds(), si.degree(), &ic_bytes);
            }
        }

        let (output_claim, r_sumcheck) =
            proof.verify(claim, max_num_rounds, max_degree, transcript)?;

        #[cfg(feature = "zolt-debug")]
        if max_num_rounds == 24 {
            use ark_serialize::CanonicalSerialize;
            eprintln!("[Stage 2] proof.verify returned r_sumcheck.len() = {}", r_sumcheck.len());
            for (i, r) in r_sumcheck.iter().enumerate() {
                // Print raw Challenge bytes (for comparison with cache_openings debug)
                let mut c_bytes = [0u8; 32];
                r.serialize_compressed(&mut c_bytes[..]).ok();
                eprintln!("[Stage 2] r_sumcheck[{}] as Challenge: {:02x?}", i, &c_bytes);
            }
            eprintln!("[Stage 2] About to call cache_openings for each instance...");
        }

        let expected_output_claim: F = sumcheck_instances
            .iter()
            .zip(batching_coeffs.iter())
            .enumerate()
            .map(|(idx, (sumcheck, coeff))| {
                let offset = sumcheck.round_offset(max_num_rounds);
                let r_slice = &r_sumcheck[offset..offset + sumcheck.num_rounds()];

                // Cache polynomial opening claims, to be proven using either an
                // opening proof or sumcheck (in the case of virtual polynomials).
                #[cfg(feature = "zolt-debug")]
                if max_num_rounds == 8 || max_num_rounds == 6 {
                    eprintln!("[Stage 3 cache_openings] Instance {} about to call cache_openings", idx);
                }
                sumcheck.cache_openings(opening_accumulator, transcript, r_slice);
                #[cfg(feature = "zolt-debug")]
                if max_num_rounds == 8 || max_num_rounds == 6 {
                    eprintln!("[Stage 3 cache_openings] Instance {} done", idx);
                }
                #[cfg(feature = "zolt-debug")]
                if (max_num_rounds == 136 || max_num_rounds == 134) && idx == 2 {
                    use ark_serialize::CanonicalSerialize;
                    eprintln!("[Stage5 Instance 2] r_slice len = {}", r_slice.len());
                    if r_slice.len() > 127 {
                        let mut r64_bytes = [0u8; 32];
                        let r64_as_f: F = r_slice[64].into();
                        r64_as_f.serialize_compressed(&mut r64_bytes[..]).ok();
                        eprintln!("[Stage5 Instance 2] r_slice[64] (as F) FULL = {:02x?}", &r64_bytes);

                        let mut r127_bytes = [0u8; 32];
                        let r127_as_f: F = r_slice[127].into();
                        r127_as_f.serialize_compressed(&mut r127_bytes[..]).ok();
                        eprintln!("[Stage5 Instance 2] r_slice[127] (as F) FULL = {:02x?}", &r127_bytes);
                    }
                }
                let claim = sumcheck.expected_output_claim(opening_accumulator, r_slice);

                #[cfg(feature = "zolt-debug")]
                if max_num_rounds == 24 || max_num_rounds == 15 || max_num_rounds == 136 || max_num_rounds == 134 || max_num_rounds == 6 || max_num_rounds == 13 {
                    use ark_serialize::CanonicalSerialize;
                    let stage = if max_num_rounds == 6 { "Stage 3" } else if max_num_rounds == 24 { "Stage 2" } else if max_num_rounds == 15 { "Stage 4" } else if max_num_rounds == 13 { "Stage 6" } else { "Stage 5" };
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

                #[cfg(feature = "zolt-debug")]
                if max_num_rounds == 9 {
                    use ark_serialize::CanonicalSerialize;
                    let mut claim_bytes = [0u8; 32];
                    claim.serialize_compressed(&mut claim_bytes[..]).ok();
                    let mut coeff_bytes = [0u8; 32];
                    coeff.serialize_compressed(&mut coeff_bytes[..]).ok();
                    let contribution = claim * coeff;
                    let mut contrib_bytes = [0u8; 32];
                    contribution.serialize_compressed(&mut contrib_bytes[..]).ok();
                    eprintln!("[Stage 1] Instance {} expected_output_claim:", idx);
                    eprintln!("  claim: {:02x?}", &claim_bytes);
                    eprintln!("  coeff: {:02x?}", &coeff_bytes);
                    eprintln!("  claim*coeff: {:02x?}", &contrib_bytes);
                }

                let contribution = claim * coeff;
                #[cfg(feature = "zolt-debug")]
                if max_num_rounds == 136 {
                    use ark_serialize::CanonicalSerialize;
                    let mut contrib_bytes2 = [0u8; 32];
                    contribution.serialize_compressed(&mut contrib_bytes2[..]).ok();
                    eprintln!("  [SUM DEBUG] instance {} contribution (for sum): {:02x?}", idx, &contrib_bytes2);
                }
                contribution
            })
            .sum();

        #[cfg(feature = "zolt-debug")]
        if max_num_rounds == 136 {
            use ark_serialize::CanonicalSerialize;
            use ark_serialize::CanonicalDeserialize;
            let mut sum_bytes = [0u8; 32];
            expected_output_claim.serialize_compressed(&mut sum_bytes[..]).ok();
            eprintln!("[SUM DEBUG] expected_output_claim (sum of all): {:02x?}", &sum_bytes);

            // Manually reconstruct from bytes and sum
            let i0_bytes: [u8; 32] = [0x76, 0xd5, 0x0f, 0xf8, 0x80, 0x91, 0xd8, 0x08, 0xce, 0x44, 0xd7, 0x5d, 0x70, 0x96, 0x2d, 0x0c, 0x30, 0xd0, 0x57, 0x1b, 0xdd, 0xe3, 0xe3, 0x35, 0xa0, 0xc4, 0x5a, 0x8e, 0x59, 0x84, 0xe3, 0x2a];
            let i1_bytes: [u8; 32] = [0x14, 0x8e, 0x95, 0xfa, 0xfd, 0x48, 0x49, 0xfd, 0x45, 0x90, 0x1f, 0x79, 0x27, 0x1e, 0xb0, 0x3b, 0x6f, 0xaf, 0x12, 0x18, 0x5a, 0xe3, 0x75, 0xe5, 0xc3, 0xc7, 0x5a, 0x4a, 0xed, 0xfb, 0x65, 0x2e];
            let i2_bytes: [u8; 32] = [0xeb, 0x4d, 0x04, 0x58, 0xef, 0xe9, 0xa4, 0x3c, 0x6e, 0x9c, 0xaf, 0x27, 0xdb, 0xb4, 0xe1, 0x82, 0x99, 0xb3, 0x94, 0xf6, 0x57, 0x76, 0xc3, 0x16, 0x8b, 0x1c, 0x97, 0xe6, 0xac, 0xa4, 0xc4, 0x0c];
            let f0 = F::deserialize_compressed(&i0_bytes[..]).unwrap();
            let f1 = F::deserialize_compressed(&i1_bytes[..]).unwrap();
            let f2 = F::deserialize_compressed(&i2_bytes[..]).unwrap();
            let manual_sum = f0 + f1 + f2;
            let mut manual_bytes = [0u8; 32];
            manual_sum.serialize_compressed(&mut manual_bytes[..]).ok();
            eprintln!("[SUM DEBUG] manual f0+f1+f2: {:02x?}", &manual_bytes);
        }

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

        #[cfg(feature = "zolt-debug")]
        {
            use ark_serialize::CanonicalSerialize;
            if num_rounds == 9 {
                let mut e_bytes = [0u8; 32];
                e.serialize_compressed(&mut e_bytes[..]).ok();
                eprintln!("[JOLT_VERIFIER] Stage1 initial_claim: {:02x?}", &e_bytes);
            }
            if num_rounds == 15 {
                let mut e_bytes = [0u8; 32];
                e.serialize_compressed(&mut e_bytes[..]).ok();
                eprintln!("[JOLT_VERIFIER] Stage4 initial_claim (hint for round 0): {:02x?}", &e_bytes);
            }
            if num_rounds == 136 || num_rounds == 134 {
                let mut e_bytes = [0u8; 32];
                e.serialize_compressed(&mut e_bytes[..]).ok();
                eprintln!("[S5V] initial_claim (e before R0): {:02x?}", &e_bytes[0..16]);
            }
        }

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
                // Print Stage 1 (9 rounds) - all iterations
                if num_rounds == 9 {
                    let mut e_bytes = [0u8; 32];
                    let mut r_bytes = [0u8; 32];
                    e.serialize_compressed(&mut e_bytes[..]).ok();
                    let r_field: F = r_i.into();
                    r_field.serialize_compressed(&mut r_bytes[..]).ok();
                    eprintln!("[JOLT_VERIFIER] Stage1 Round {} challenge: {:02x?}", i, &r_bytes);
                    eprintln!("[JOLT_VERIFIER] Stage1 Round {} new_claim: {:02x?}", i, &e_bytes);
                    // Also print round polynomial coefficients
                    for (j, c) in self.compressed_polys[i].coeffs_except_linear_term.iter().enumerate() {
                        let mut c_bytes = [0u8; 32];
                        c.serialize_compressed(&mut c_bytes[..]).ok();
                        eprintln!("[JOLT_VERIFIER] Stage1 Round {} coeff[{}]: {:02x?}", i, j, &c_bytes);
                    }
                }
                // Print Stage 2 (24 rounds) - rounds 16-23 for InstructionClaimReduction
                if num_rounds == 24 && i >= 16 {
                    let mut r_bytes = [0u8; 32];
                    let r_field: F = r_i.into();
                    r_field.serialize_compressed(&mut r_bytes[..]).ok();
                    eprintln!("  Stage2 Round {} challenge: {:02x?}", i, &r_bytes);
                }
                // Print Stage 3 (6 rounds) - all rounds
                if num_rounds == 6 {
                    eprintln!("[JOLT_VERIFIER] Stage3 Round {} coeffs:", i);
                    for (j, c) in self.compressed_polys[i].coeffs_except_linear_term.iter().enumerate() {
                        let mut c_bytes = [0u8; 32];
                        c.serialize_compressed(&mut c_bytes[..]).ok();
                        eprintln!("[JOLT_VERIFIER] Stage3 Round {} coeff[{}]: {:02x?}", i, j, &c_bytes);
                    }
                    let mut r_bytes = [0u8; 32];
                    let r_field: F = r_i.into();
                    r_field.serialize_compressed(&mut r_bytes[..]).ok();
                    eprintln!("[JOLT_VERIFIER] Stage3 Round {} challenge: {:02x?}", i, &r_bytes);
                    let mut e_bytes = [0u8; 32];
                    e.serialize_compressed(&mut e_bytes[..]).ok();
                    eprintln!("[JOLT_VERIFIER] Stage3 Round {} new_claim: {:02x?}", i, &e_bytes);
                }
                // Print Stage 4 (15 rounds) all iterations
                if num_rounds == 15 {
                    // Print round polynomial coefficients
                    eprintln!("  Round {} coeffs:", i);
                    for (j, c) in self.compressed_polys[i].coeffs_except_linear_term.iter().enumerate() {
                        let mut c_bytes = [0u8; 32];
                        c.serialize_compressed(&mut c_bytes[..]).ok();
                        eprintln!("    [{}]: {:02x?}", j, &c_bytes);
                    }
                    let mut e_bytes = [0u8; 32];
                    let mut r_bytes = [0u8; 32];
                    e.serialize_compressed(&mut e_bytes[..]).ok();
                    let r_field: F = r_i.into();
                    r_field.serialize_compressed(&mut r_bytes[..]).ok();
                    eprintln!("  Round {} challenge: {:02x?}", i, &r_bytes);
                    eprintln!("  Round {} new_claim: {:02x?}", i, &e_bytes);
                }
                // Print Stage 6 (13 rounds) - ALL rounds for debugging
                if num_rounds == 13 {
                    let mut r_bytes = [0u8; 32];
                    let r_field: F = r_i.into();
                    r_field.serialize_compressed(&mut r_bytes[..]).ok();
                    let mut e_bytes = [0u8; 32];
                    e.serialize_compressed(&mut e_bytes[..]).ok();
                    eprintln!("  [S6V] R{} challenge={:02x?} new_e={:02x?} degree={}", i, &r_bytes[0..8], &e_bytes[0..8], self.compressed_polys[i].degree());
                    // Print the coefficients
                    for (j, c) in self.compressed_polys[i].coeffs_except_linear_term.iter().enumerate() {
                        let mut c_bytes = [0u8; 32];
                        c.serialize_compressed(&mut c_bytes[..]).ok();
                        eprintln!("    coeff[{}]={:02x?}", j, &c_bytes[0..8]);
                    }
                }
                // Print Stage 5 (136 rounds) - ALL rounds for claim tracking
                if num_rounds == 136 || num_rounds == 134 {
                    let mut r_bytes = [0u8; 32];
                    let r_field: F = r_i.into();
                    r_field.serialize_compressed(&mut r_bytes[..]).ok();
                    let mut e_bytes = [0u8; 32];
                    e.serialize_compressed(&mut e_bytes[..]).ok();
                    // Print hint (e before eval) and new claim for EVERY round
                    eprintln!("  [S5V] R{} challenge={:02x?} new_e={:02x?} degree={}", i, &r_bytes[0..16], &e_bytes[0..16], self.compressed_polys[i].degree());
                }
            }
        }

        #[cfg(feature = "zolt-debug")]
        {
            use ark_serialize::CanonicalSerialize;
            let mut e_bytes = [0u8; 32];
            e.serialize_compressed(&mut e_bytes[..]).ok();
            eprintln!("[SUMCHECK VERIFY] output_claim after {} rounds: {:02x?}", num_rounds, &e_bytes);
        }

        Ok((e, r))
    }
}
