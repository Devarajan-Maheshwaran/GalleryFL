package com.fgt.galleryfl.data.network

import java.security.SecureRandom

/**
 * Android-side counterpart of the server's `generate_access_code`
 * (server/security.py). Produces the same simple 8-character alphanumeric
 * access code: uppercase letters + digits, excluding visually ambiguous
 * characters (0/O, 1/I, L). Used both to validate a pasted code and to offer
 * a generated suggestion, keeping the contract identical on both sides.
 */
object AccessCodeGenerator {
    private const val ALPHABET = "ABCDEFGHJKLMNPQRSTUVWXYZ23456789"

    fun generate(length: Int = 8): String {
        val rng = SecureRandom()
        return (1..length).joinToString("") { ALPHABET[rng.nextInt(ALPHABET.length)].toString() }
    }

    fun isValid(code: String): Boolean {
        if (code.length != 8) return false
        return code.all { it in ALPHABET }
    }
}
