"""In this example, we perform encrypted steganography using FIMEC.
Doing encrypted steganography in this way is "perfectly secure" in the sense
that the stegotext and covertext are statistically indistinguishable.
"""

import re
import secrets

from mec import FIMEC

from random_string import RandomString
from gpt2 import GPT2Marginal

# For encrypted steganography, the sender and receiver share a private key.
cipher_len = 15
shared_private_key = secrets.token_bytes(cipher_len)

# The plaintext is the message the sender wants to communicate.
plaintext = "Attack at dawn!"

# This is a representation of the plaintext as a sequence of bytes.
bytetext = plaintext.encode("utf-8")
assert len(bytetext) == cipher_len

# The ciphertext is the plaintext encrypted with the shared private key.
# It is always distributed uniformly, since the private key is random.
ciphertext = [a ^ b for a, b in zip(bytetext, shared_private_key)]
ciphertext_dist = RandomString(num_chars=2**8, string_len=cipher_len)

# The covertext distribution is a distribution over innocuous content.
# We use GPT-2 here to make the example easy to run.
# Use a better language model for real applications.
covertext_dist = GPT2Marginal(
    prompt="Good evening.",
    max_len=45,
    temperature=1.0,
    k=50,
)

# FIMEC defines the communication protocol between the sender and receiver.
imec = FIMEC(ciphertext_dist, covertext_dist)

# The stegotext is some innocuous content with the ciphertext hidden inside.
# The sender communicates the stegotext to the receiver over a public channel.
stegotext, _ = imec.sample_y_given_x(ciphertext)

# The estimated ciphertext is the receiver's "best guess" of the ciphertext,
# given the stegotext.
estimated_ciphertext, _ = imec.estimate_x_given_y(stegotext)

# The estimated bytetext is a decryption of the estimated ciphertext using the
# shared private key.
estimated_bytetext = bytes(
    [a ^ b for a, b in zip(estimated_ciphertext, shared_private_key)]
)

# The estimated plaintext is the estimated bytetext decoded as a string.
try:
    estimated_plaintext = estimated_bytetext.decode("utf-8")
except UnicodeDecodeError:
    estimated_plaintext = "Estimated bytetext is not valid UTF-8."

# Summarize the results.
print(f"Plaintext: {plaintext}\n{'-' * 80}")
print(f"Bytetext: {bytetext!r}\n{'-' * 80}")
# Format stegotext for better readability.
formatted_stegotext = re.sub(
    " {2,}", "\n", covertext_dist.decode(stegotext).replace("\n", " ")
).strip()
print(f"Stegotext: {formatted_stegotext}\n{'-' * 80}")
print(f"Estimated bytetext: {estimated_bytetext!r}\n{'-' * 80}")
print(f"Estimated plaintext: {estimated_plaintext}\n{'-' * 80}")
if plaintext == estimated_plaintext:
    print("The secret message was successfully communicated.")
else:
    print("The secret message was not successfully communicated.")
