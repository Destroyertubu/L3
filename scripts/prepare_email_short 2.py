#!/usr/bin/env python3
"""
Prepare Short Email Dataset for L3 String Compression

Creates a subset of emails that fit within 64-bit encoding limits.
With 7-bit encoding (charset 91), max 9 chars fit in 64 bits.
With 6-bit encoding (charset 64), max 10 chars fit in 64 bits.
With 5-bit encoding (charset 32), max 12 chars fit in 64 bits.

This script creates a lowercase-only version for better compression.
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

def normalize_email(email):
    """
    Normalize email for better compression:
    - Convert to lowercase
    - Remove special characters that aren't needed
    """
    # Convert to lowercase
    email = email.lower()

    # Keep only alphanumeric, @, ., _, -
    normalized = ''.join(c for c in email if c.isalnum() or c in '@._-')

    return normalized

def main():
    input_file = "/root/autodl-tmp/test/data/sosd/strings/email.txt"
    output_file = "/root/autodl-tmp/test/data/sosd/strings/email_leco_30k_short.txt"
    target_count = 30000
    max_length = 12  # For 5-bit encoding (lowercase letters only)
    random_seed = 42

    print(f"LeCo Short Email Dataset Preparation")
    print(f"=====================================")
    print(f"Input:  {input_file}")
    print(f"Output: {output_file}")
    print(f"Target: {target_count} emails (max length: {max_length})")
    print()

    # Read original emails
    with open(input_file, 'r') as f:
        emails = [line.strip() for line in f if line.strip()]

    print(f"Original dataset: {len(emails)} emails")

    # Transform and filter
    transformed = []
    for email in emails:
        # Normalize to lowercase
        email = email.lower()

        # Reverse host
        rev = reverse_host(email)

        # Only keep if short enough
        if len(rev) <= max_length:
            transformed.append(rev)

    print(f"After filtering (len <= {max_length}): {len(transformed)} emails")

    # If not enough, try extracting just the domain part
    if len(transformed) < target_count:
        print("Not enough short emails, using domain-only extraction...")
        transformed = []
        for email in emails:
            email = email.lower()
            if '@' in email:
                user, host = email.split('@', 1)
                # Reverse host
                host_parts = host.split('.')
                rev_host = '.'.join(reversed(host_parts))

                # Use just the reversed host if it fits
                if len(rev_host) <= max_length:
                    transformed.append(rev_host)
                # Or just the TLD and first domain part
                elif len(host_parts) >= 2:
                    short = f"{host_parts[-1]}.{host_parts[-2]}"
                    if len(short) <= max_length:
                        transformed.append(short)

    # Remove duplicates and sort
    transformed = sorted(set(transformed))
    print(f"After dedup: {len(transformed)} unique entries")

    # Sample if needed
    if len(transformed) > target_count:
        random.seed(random_seed)
        transformed = random.sample(transformed, target_count)
        transformed.sort()
        print(f"Sampled: {len(transformed)} entries")

    # Calculate statistics
    total_length = sum(len(e) for e in transformed)
    avg_length = total_length / len(transformed) if transformed else 0
    min_length = min(len(e) for e in transformed) if transformed else 0
    max_len_actual = max(len(e) for e in transformed) if transformed else 0

    print(f"\nFinal dataset statistics:")
    print(f"  Count: {len(transformed)}")
    print(f"  Average length: {avg_length:.1f} bytes")
    print(f"  Min length: {min_length} bytes")
    print(f"  Max length: {max_len_actual} bytes")
    print(f"  Total size: {total_length} bytes")

    # Analyze character set
    all_chars = set()
    for s in transformed:
        all_chars.update(s)
    print(f"  Character set size: {len(all_chars)}")
    print(f"  Characters: {''.join(sorted(all_chars))}")

    # Show samples
    print(f"\nFirst 10 entries:")
    for i, s in enumerate(transformed[:10]):
        print(f"  [{i}] {s}")

    # Write output
    with open(output_file, 'w') as f:
        for s in transformed:
            f.write(s + '\n')

    print(f"\nOutput written to: {output_file}")

if __name__ == "__main__":
    main()
