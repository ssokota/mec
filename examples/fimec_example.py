"""In this example, we couple a bytetext distribution with a natural language 
distribution using FIMEC. We estimate the joint entropy of the coupling using samples.
"""

import numpy as np

from mec import FIMEC

from random_string import RandomString
from gpt2 import GPT2Marginal

bytetext_dist = RandomString(num_chars=256, string_len=10)
gpt2_dist = GPT2Marginal(prompt="Here's some text:", max_len=25, temperature=1.0, k=50)
imec = FIMEC(bytetext_dist, gpt2_dist)
# Entropy of bytetext is easy to compute because it's a uniform distribution
bytetext_entropy = -bytetext_dist.ll
gpt2_nlls = []
imec_nlls = []
num_samples = 5
for i in range(num_samples):
    (x, y), ll = imec.sample()
    imec_nlls.append(-ll)
    x, ll = gpt2_dist.sample()
    gpt2_nlls.append(-ll)
imec_ent = np.mean(imec_nlls)
imec_se = np.std(imec_nlls) / np.sqrt(num_samples)
gpt2_ent = np.mean(gpt2_nlls)
gpt2_se = np.std(gpt2_nlls) / np.sqrt(num_samples)
# Estimates below are reasonable b/c bytetext entropy is constant & less than gpt2 entropy.
print(
    f"Estimate of lower bound on joint entropy: {gpt2_ent:0.2f} +/- {gpt2_se:0.2f} nats"
)
print(f"Estimate of joint entropy: {imec_ent:0.2f} +/- {imec_se:0.2f} nats")
print(
    f"Estimate of upper bound on joint entropy: {gpt2_ent + bytetext_entropy:0.2f} +/- {gpt2_se:0.2f} nats"
)
