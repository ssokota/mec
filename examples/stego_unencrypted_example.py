"""In this example, we perform unencrypted steganography using ARIMEC.
Compared to encrypted steganography, unencrypted steganography is able to achieve
higher data rates because it can leverage priors about the plaintext. However, it
is not perfectly secure.
"""

import re

from mec import ARIMEC

from gpt2 import GPT2Marginal

# The plaintext prior is a prior on secret messages we may want to send.
plaintext_len = 10
plaintext_prior = GPT2Marginal(
    prompt="Here's a secret message:", max_len=plaintext_len, temperature=1.0, k=2_000
)

# The covertext distribution is a distribution over innocuous content.
# We use GPT-2 here to make the example easy to run.
# Use a better language model for real applications.
covertext_dist = GPT2Marginal(prompt="Good evening.", max_len=25, temperature=1.0, k=50)

# ARIMEC defines the communication protocol between the sender and receiver.
imec = ARIMEC(plaintext_prior, covertext_dist)

# This is the message the sender wants to communicate.
plaintext = "Meet in the park at 3:30pm."

plaintext_tokens = plaintext_prior.encode(plaintext)
# Assert non-zero probability under top-k sampling
assert plaintext_tokens is not None
# Assert required length
assert len(plaintext_tokens) == plaintext_len

# The stegotext is some innocuous content with the plaintext hidden inside.
# The sender communicates the stegotext to the receiver over a public channel.
stegotext, _ = imec.sample_y_given_x(plaintext_tokens)

# The estimated plaintext is the receiver's "best guess" of the plaintext,
# given the stegotext.
estimated_plaintext_tokens, _ = imec.estimate_x_given_y(stegotext)
estimated_plaintext = plaintext_prior.decode(estimated_plaintext_tokens)

# Summarize the results.
print(f"Plaintext: {plaintext}\n{'-' * 80}")
# Format stegotext for better readability.
formatted_stegotext = re.sub(
    " {2,}", "\n", covertext_dist.decode(stegotext).replace("\n", " ")
).strip()
print(f"Stegotext: {formatted_stegotext}\n{'-' * 80}")
print(f"Estimated plaintext: {estimated_plaintext}\n{'-' * 80}")
if plaintext == estimated_plaintext:
    print("The secret message was successfully communicated.")
else:
    print("The secret message was not successfully communicated.")
