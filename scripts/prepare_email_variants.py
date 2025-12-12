#!/usr/bin/env python3
"""
Prepare Email Dataset - LeCo-like format with truncation

According to LeCo paper Section 4.1:
- email: 30K email addresses (host reversed) with an average string length of 15 bytes

Since our 64-bit encoding with 7-bit charset only supports ~9 characters,
we'll create the dataset and let the compression truncate as needed.

This is a realistic test showing the limitation of 64-bit encoding
and demonstrating that 128-bit support would be needed for full email strings.
"""

import sys
import random

def reverse_host(email):
    """Reverse the host part of an email address."""
    if '@' not in email:
        return email

    parts = email.split('@')
    if len(parts) != 2:
        return email

    user = parts[0]
    host = parts[1]

    host_parts = host.split('.')
    reversed_host = '.'.join(reversed(host_parts))

    return f"{reversed_host}@{user}"

def main():
    input_file = "/root/autodl-tmp/test/data/sosd/strings/email.txt"
    output_dir = "/root/autodl-tmp/test/data/sosd/strings/"
    random_seed = 42

    # Read original emails
    with open(input_file, 'r') as f:
        all_emails = [line.strip().lower() for line in f if line.strip()]

    print(f"Original dataset: {len(all_emails)} emails")

    # Create the 30K host-reversed dataset (LeCo spec)
    random.seed(random_seed)
    sampled = random.sample(all_emails, 30000)
    transformed = [reverse_host(e) for e in sampled]
    transformed.sort()

    output_file = output_dir + "email_leco_30k.txt"
    with open(output_file, 'w') as f:
        for e in transformed:
            f.write(e + '\n')
    print(f"Created: {output_file} ({len(transformed)} entries)")

    # Also create a short-prefix version for testing 64-bit encoding
    # Use just the first 12 characters (5-bit encoding limit)
    short_transformed = []
    for e in transformed:
        # Take first 10 chars to fit in 64-bit with 6-bit encoding
        short_transformed.append(e[:10] if len(e) > 10 else e)
    short_transformed = sorted(set(short_transformed))  # Remove dups from truncation

    output_file_short = output_dir + "email_leco_30k_truncated.txt"
    with open(output_file_short, 'w') as f:
        for e in short_transformed:
            f.write(e + '\n')
    print(f"Created: {output_file_short} ({len(short_transformed)} entries)")

    # Create lowercase-only version (enables 5-bit encoding, 12 chars)
    # Keep only a-z, 0-9, and essential chars
    allowed_chars = set('abcdefghijklmnopqrstuvwxyz0123456789@._-')
    clean_transformed = []
    for e in transformed:
        clean = ''.join(c for c in e if c in allowed_chars)
        if len(clean) >= 5:  # Skip very short ones
            clean_transformed.append(clean[:12])  # Truncate to 12 chars

    clean_transformed = sorted(set(clean_transformed))

    output_file_clean = output_dir + "email_leco_30k_clean.txt"
    with open(output_file_clean, 'w') as f:
        for e in clean_transformed:
            f.write(e + '\n')
    print(f"Created: {output_file_clean} ({len(clean_transformed)} entries)")

    # Stats
    for name, data in [("Full 30K", transformed),
                        ("Truncated 10char", short_transformed),
                        ("Clean 12char", clean_transformed)]:
        if data:
            avg_len = sum(len(e) for e in data) / len(data)
            print(f"\n{name}:")
            print(f"  Count: {len(data)}")
            print(f"  Avg length: {avg_len:.1f}")
            charset = set()
            for e in data:
                charset.update(e)
            print(f"  Charset size: {len(charset)}")

if __name__ == "__main__":
    main()
