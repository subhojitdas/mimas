REINFORCE
   |
   |  (Problem: extremely high variance, noisy learning)
   v
REINFORCE + Baseline
   |
   |  (Idea: subtract V(s) to reduce variance)
   |  (Still Monte Carlo, still episodic)
   v
Actor–Critic (A2C)
   |
   |  (Idea: bootstrap with TD learning)
   |  (Advantage = r + γV(s') − V(s))
   |
   |  Problems that remain:
   |    - Policy can change too much
   |    - Training unstable
   |    - Samples used once
   v
A2C + GAE
   |
   |  (Idea: multi-step advantage estimation)
   |  (Bias–variance control with λ)
   |
   |  Still broken:
   |    - Large destructive updates
   |    - No trust region
   v
TRPO (conceptual, not practical)
   |
   |  (Idea: constrain KL divergence)
   |  (Trust region guarantees monotonic improvement)
   |
   |  Problem:
   |    - Second-order optimization
   |    - Very complex, slow
   v
PPO
   |
   |  (Idea: approximate trust region with clipping)
   |  (Reuse data safely with multiple epochs)
   |
   v
Stable, scalable deep RL
