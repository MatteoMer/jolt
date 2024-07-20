// SPDX-License-Identifier: MIT
pragma solidity ^0.8.21;

import {Jolt} from "../src/reference/JoltTypes.sol";
import {Fr, FrLib, sub, MODULUS} from "../src/reference/Fr.sol";
import {Transcript, FiatShamirTranscript} from "../src/subprotocols/FiatShamirTranscript.sol";
import {UniPoly, UniPolyLib} from "../src/reference/UniPoly.sol";
import {GrandProductArgument} from "../src/reference/JoltVerifier.sol";
import {TestBase} from "./base/TestBase.sol";

import "forge-std/console.sol";

error GrandProductArgumentFailed();
error SumcheckFailed();

contract GrandProductArgumentGasWrapper is TestBase {
    using FiatShamirTranscript for Transcript;

    function verifySumcheckLayer(
        Jolt.BatchedGrandProductLayerProof memory layer,
        Transcript memory transcript,
        Fr claim,
        uint256 degree_bound,
        uint256 num_rounds
    ) public view returns (Fr, Fr[] memory) {
        return GrandProductArgument.verifySumcheckLayer(layer, transcript, claim, degree_bound, num_rounds);
    }


    function buildEqEval(Fr[] memory rGrandProduct, Fr[] memory rSumcheck) public view returns (Fr eqEval) {
        return GrandProductArgument.buildEqEval(rGrandProduct, rSumcheck);
    }


    function verifySumcheckClaim(
        Jolt.BatchedGrandProductLayerProof[] memory layerProofs,
        uint256 layerIndex,
        Fr[] memory coeffs,
        Fr sumcheckClaim,
        Fr eqEval,
        Fr[] memory claims,
        Fr[] memory rGrandProduct,
        Transcript memory transcript
    ) public view returns (Fr[] memory newClaims, Fr[] memory newRGrandProduct) {
        return GrandProductArgument.verifySumcheckClaim(layerProofs, layerIndex, coeffs, sumcheckClaim, eqEval, claims, rGrandProduct, transcript);
    }


    /* Here I'm doing the same thing as in the library, 
     * but wrapping the internal functions to external ones so they appears in foundry's gas-report */
    function verify(
        Jolt.BatchedGrandProductProof memory proof,
        Fr[] memory claims,
        Transcript memory transcript
    ) external view returns (Fr[] memory) {
        Fr[] memory rGrandProduct = new Fr[](0);

        for (uint256 i = 0; i < proof.layers.length; i++) {
            uint256[] memory loaded = transcript.challenge_scalars(claims.length, MODULUS);
            Fr[] memory coeffs;
            assembly {
                coeffs := loaded
            }

            Fr joined_claim = Fr.wrap(0);
            for (uint256 k = 0; k < claims.length; k++) {
                joined_claim = joined_claim + (claims[k] * coeffs[k]);
            }

            if (
                claims.length != proof.layers[i].leftClaims.length
                    || claims.length != proof.layers[i].rightClaims.length
            ) {
                revert GrandProductArgumentFailed();
            }

            // What's commented below works 
            (Fr sumcheckClaim, Fr[] memory rSumcheck) =
                verifySumcheckLayer(proof.layers[i], transcript, joined_claim, 3, i);

            /* But the code below crashes because of a Sumcheck claim mismatch.
             *   
             * I'm using `this.verifySumcheckLayer` to use it as an external function
             * I also tried to make the wrapper functions fully external and call the verify code in the test directly 
             * and got same results */
            /* (Fr sumcheckClaim, Fr[] memory rSumcheck) =
                this.verifySumcheckLayer(proof.layers[i], transcript, joined_claim, 3, i); */

            if (rSumcheck.length != rGrandProduct.length) {
                revert GrandProductArgumentFailed();
            }

            for (uint256 l = 0; l < proof.layers[l].leftClaims.length; l++) {
                transcript.append_scalar(Fr.unwrap(proof.layers[i].leftClaims[l]));
                transcript.append_scalar(Fr.unwrap(proof.layers[i].rightClaims[l]));
            }

            // works
            Fr eqEval = buildEqEval(rGrandProduct, rSumcheck); 

            // crashes
            /* Fr eqEval = this.buildEqEval(rGrandProduct, rSumcheck); */

            rGrandProduct = new Fr[](rSumcheck.length);
            for (uint256 l = 0; l < rSumcheck.length; l++) {
                rGrandProduct[l] = rSumcheck[rSumcheck.length - 1 - l];
            }


            // works
            (claims, rGrandProduct) =
                verifySumcheckClaim(proof.layers, i, coeffs, sumcheckClaim, eqEval, claims, rGrandProduct, transcript); 

            // crashes
            /* (claims, rGrandProduct) =
                this.verifySumcheckClaim(proof.layers, i, coeffs, sumcheckClaim, eqEval, claims, rGrandProduct, transcript); */

            /* From my investigations, it seems that the coeffs (that I'm getting via the fiat-shamir lib) are not valid when
             * using external functions ? */
        }

        return rGrandProduct;
    }
}
