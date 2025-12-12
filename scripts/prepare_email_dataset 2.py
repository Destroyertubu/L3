#!/usr/bin/env python3
"""
Prepare Email Dataset for LeCo String Compression

According to LeCo paper Section 4.1:
- email: 30K email addresses (host reversed) with an average string length of 15 bytes

Transformation:
1. Sample 30K emails from the original dataset
2. Reverse the host (domain) part: user@yahoo.com -> com.yahoo@user
3. Sort the result for optimal compression (serial correlation)
"""

import sys
import random

def reverse_host(email):
    """
    Reverse the host part of an email address.
    Example: user@yahoo.com -> com.yahoo@user
    """
    if '@' not in email:
        return email

    parts = email.split('@')
    if len(parts) != 2:
        return email

    user = parts[0]
    host = parts[1]

    # Reverse the host parts (split by '.')
    host_parts = host.split('.')
    reversed_host = '.'.join(reversed(host_parts))

    # Return reversed format: reversed_host@user
    return f"{reversed_host}@{user}"

def main():
    input_file = "/root/autodl-tmp/test/data/sosd/strings/email.txt"
    output_file = "/root/autodl-tmp/test/data/sosd/strings/email_leco_30k.txt"
    target_count = 30000
    random_seed = 42

    # Parse arguments
    if len(sys.argv) > 1:
        output_file = sys.argv[1]
    if len(sys.argv) > 2:
        target_count = int(sys.argv[2])

    print(f"LeCo Email Dataset Preparation")
    print(f"==============================")
    print(f"Input:  {input_file}")
    print(f"Output: {output_file}")
    print(f"Target: {target_count} emails")
    print()

    # Read original emails
    with open(input_file, 'r') as f:
        emails = [line.strip() for line in f if line.strip()]

    print(f"Original dataset: {len(emails)} emails")

    # Sample if needed
    if len(emails) > target_count:
        random.seed(random_seed)
        emails = random.sample(emails, target_count)
        print(f"Sampled: {len(emails)} emails")
    elif len(emails) < target_count:
        print(f"Warning: Only {len(emails)} emails available (target: {target_count})")

    # Transform: reverse host
    transformed = []
    for email in emails:
        transformed.append(reverse_host(email))

    # Sort for serial correlation (improves compression)
    transformed.sort()

    # Calculate statistics
    total_length = sum(len(e) for e in transformed)
    avg_length = total_length / len(transformed)
    min_length = min(len(e) for e in transformed)
    max_length = max(len(e) for e in transformed)

    print(f"\nTransformed dataset statistics:")
    print(f"  Count: {len(transformed)}")
    print(f"  Average length: {avg_length:.1f} bytes")
    print(f"  Min length: {min_length} bytes")
    print(f"  Max length: {max_length} bytes")
    print(f"  Total size: {total_length} bytes")

    # Show sample transformations
    print(f"\nSample transformations:")
    random.seed(random_seed)
    sample_indices = random.sample(range(len(emails)), min(5, len(emails)))
    for i in sample_indices:
        orig = emails[i]
        trans = reverse_host(orig)
        print(f"  {orig}")
        print(f"    -> {trans}")

    # Show first few sorted entries
    print(f"\nFirst 10 sorted entries:")
    for i, email in enumerate(transformed[:10]):
        print(f"  [{i}] {email}")

    # Write output
    with open(output_file, 'w') as f:
        for email in transformed:
            f.write(email + '\n')

    print(f"\nOutput written to: {output_file}")
    print(f"Total bytes: {total_length + len(transformed)} (with newlines)")

if __name__ == "__main__":
    main()
