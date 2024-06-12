"""In this example, we couple a two natural language distributions using ARIMEC.
We estimate the joint entropy of the coupling using samples.
"""

import numpy as np

from mec import ARIMEC

from gpt2 import GPT2Marginal


gpt2_dist = GPT2Marginal(prompt="Here's some text:", max_len=10, temperature=1.0, k=50)
other_gpt2_dist = GPT2Marginal(
    prompt="Here's some other text:", max_len=25, temperature=1.0, k=50
)
imec = ARIMEC(gpt2_dist, other_gpt2_dist)
nld_nlls = []
onld_nlls = []
coupling_nlls = []
num_samples = 5
for i in range(num_samples):
    (x, y), ll = imec.sample()
    coupling_nlls.append(-ll)
    ll = gpt2_dist.evaluate(x)
    nld_nlls.append(-ll)
    ll = other_gpt2_dist.evaluate(y)
    onld_nlls.append(-ll)
coupling_ent = np.mean(coupling_nlls)
coupling_se = np.std(coupling_nlls) / np.sqrt(num_samples)
nld_ent = np.mean(nld_nlls)
onld_ent = np.mean(onld_nlls)
upper_bound_se = np.sqrt(np.std(nld_nlls) ** 2 + np.std(onld_nlls) ** 2) / np.sqrt(
    num_samples
)
print(
    f"Estimate of lower bound on joint entropy: {max(nld_ent, onld_ent):0.2f} +/- {upper_bound_se:0.2f} nats"
)
print(f"Estimate of joint entropy: {coupling_ent:0.2f} +/- {coupling_se:0.2f} nats")
print(
    f"Estimate of upper bound on joint entropy: {nld_ent + onld_ent:0.2f} +/- {upper_bound_se:0.2f} nats"
)
