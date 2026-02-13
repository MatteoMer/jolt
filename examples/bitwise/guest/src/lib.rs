#![cfg_attr(feature = "guest", no_std)]
use core::hint::black_box;

/// Count number of set bits (popcount)
fn popcount(mut n: u32) -> u32 {
    let mut count: u32 = 0;
    while n != 0 {
        count += n & 1; // Uses AND
        n >>= 1;        // Uses SRLI
    }
    count
}

/// Bit manipulation operations combining AND, OR, XOR, shifts
fn bit_ops(a: u32, b: u32) -> u32 {
    let x = a & b;   // AND
    let y = a | b;   // OR
    let z = a ^ b;   // XOR
    let w = a << 4;  // SLLI
    let v = b >> 2;  // SRLI

    // Combine results with wrapping arithmetic
    x.wrapping_add(y).wrapping_add(z).wrapping_add(w).wrapping_add(v)
}

#[jolt::provable(memory_size = 32768, max_trace_length = 65536)]
fn bitwise(a: u32, b: u32) -> u32 {
    let a = black_box(a);
    let b = black_box(b);

    // popcount(0xF0F0F0F0) = 16
    let pop = popcount(a);

    // bit_ops computes combined result
    let ops = bit_ops(a, b);

    // Return low bits of combined result
    pop.wrapping_add(ops & 0xFF)
}
