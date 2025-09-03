# NiceODE

# Introduction

NiceODE is a modern toolkit for solving and optimizing parameters of ordinary differential equation (ODE) systems. It provides a containerized environment and a unified Python interface for tackling these problems from multiple perspectives:

- Frequentist NLME: Using JAX and SciPy to implement established methods like First-Order (FO) and First-Order Conditional Estimation (FOCE).

- Bayesian Hierarchical Modeling: Using state-of-the-art MCMC samplers from PyMC and Numpyro.

The project is built as a self-contained development environment featuring experiment tracking with MLFlow and a Poetry-managed package, allowing for rapid iteration on both the core NiceODE library and analysis scripts.

# Purpose
The primary purpose of this project was to serve as a rigorous testbed for exploring advanced scientific computing concepts. The specific goals were to:

- Identify the capabilities and limitations of using an LLM as a pair programming partner for complex mathematical and software engineering tasks.

- Develop a deeper, intuitive understanding of the methodologies used to estimate parameters of differential equations.

- Expand my practical Python toolbox to include JAX for high-performance, gradient-based optimization.

- Sharpen my skills in Bayesian hierarchical modeling with PyMC and Numpyro.

A core motivation behind this work is the need for interpretable models in healthcare. While complex machine learning models are powerful, my experience is that their insights can often be reformulated into simpler, mechanistic models (like systems of ODEs). These models are better understood, trusted, and adopted by clinical stakeholders. NiceODE is an exploration of the tools needed to build these understandable models effectively.

# Methodology
## Frequentist
### Objective
Frequentist objective function modeling is performed using approximations of `2*negative log-likelihood` following from the laplacian approximation. Namely the relaxations of the laplacian known in the PK field as the First-Order (FO) and First-Oder w/ Conditional Estimation (FOCE) methods, and the forumlation of the laplacian approxmation known as First-Order Estimation w/ Conditional Estimation and Interaction (Kim: http://dx.doi.org/10.12793/tcp.2015.23.1.1, Bae:http://dx.doi.org/10.12793/tcp.2016.24.4.161, Wang: http://dx.doi.org/10.1007/s10928-007-9060-6). The implementation of these methods and their gradients in a modern automatic differentiation framework like JAX formed the central and most challenging part of this project.
### Error Model
Model error is currently implemented as additive error only. 

## Bayesian
### Objective
Bayesian estimation is performed using non-centered heirarchical modeling of the subject level effects. Presently only one level of 'mixed' effect is supported at the subject-level, but the flexbility of the MCMC estimation means that extending the code which constructs the heirarchical model to include additional levels is trivial (for example modeling subject-level effects and dosing route level effects in one model).
### Error Model
Model error implemented as additive, proportional, or combined. 
### ODE Format
My experiments with bayesian methods suggest that in order to produce a model which samples effectively, the differential equations used to model a process should be non-dimensionalized. 

## Shared Methods
Bayesian and Frequentist methods both estimate the model parameters on the log scale. The parameters are transformed to the true scale before IVP solving to generate the predictions which are evaluated against the modeled data.


# Applicability Beyond PK
The package was developed for use with pharmacokinetic (PK) differential equations such
as those seen in `src/diffeqs.py`, but the framework provided should work for any ODE's which can be expressed in the `scipy` or `sympy` functional format.  

# Viability of Jax for NLME
Established R methodologies for performing nlme modeling (`nlmixr2`: https://nlmixr2.org/) are performant, but do not use 'modern' jit-compiled broadly applicable AD methods such as Jax. Instead, `nlmixr2` utilizes a fixed set of rules for constructing gradient and loss functions under the assumption that the user is working with diffeq's aligned with the PK context. At high level `nlmixr2` works by parsing a parameterization of a given ODE, then per parameter, given that it 'knows' how that parameter interacts with the others in the nlme PK context, it constructs bespoke c++ code which is used to analyltically determine the gradient and loss at each loss function evaluation. 

In contrast, JAX takes a more general approach by using reverse-mode automatic differentiation (jax.grad) to trace a function's execution and compute its gradient. This is incredibly flexible, but for the FOCE method, it creates an extremely challenging technical problem. FOCE and FOCEi require a bi-level optimization: an outer optimization of population parameters, and an inner optimization to find the per-subject random effects (b_i). This is a "grad-of-a-grad" problem.

The journey to a working JAX-based FOCE gradient revealed several layers of complexity:

1) The Manual VJP + Optax: The first approach was to manually implement the Implicit Function Theorem using jax.custom_vjp where the IFT described  the working of a 'manually' defined optax optmizer. This proved to be exceptionally difficult due to the complexity of the matrix calculus and extreme sensitivity to subtle mathematical and numerical errors. Even in a very stripped back Minimal Reproducible Example (MRE) context without an IVP on the inner optmizer, these numerical errors made it difficult to recover a gradient which matched the one estimated by finite differences.

2) The jaxopt Approach: The recommended JAX-native solution is to use a library like jaxopt, whose solvers are designed to be differentiable. However, this approach failed due to a fundamental incompatibility when nesting VJPs. The jaxopt solver's VJP, when trying to differentiate an inner loss function that itself contained a diffrax ODE solver with its own adjoint VJP, created a "VJP-of-a-VJP" problem that triggered low-level errors in JAX's AD machinery.

3) The Hybrid Approach: The most promising architecture combined a robust jaxopt optimizer for the forward pass with our manual VJP for the backward pass. This avoided the nested VJP issue. However, even with a Minimal Reproducible Example (MRE), this approach failed to produce correct gradients, proving that subtle, unidentified numerical bugs remained in the manual VJP implementation.

The conclusion is that obtaining a fast, stable, and accurate gradient for the FOCE objective in JAX is a frontier problem. While theoretically possible, it requires a level of numerical and mathematical precision in the manual VJP implementation that is exceptionally difficult to achieve without my doing additional graduate+ level research/knowledge accumulation specific to this problem. With such knowledge of 'exactly how to discuss the issue at hand' in the mind of the human user, it may be that model like Gemini 2.5 Pro could construct the required vjp following precise prompting and feedback. Without such human knowledge however, at the time of writing it does seem like I have identified an edge of Gemini 2.5 Pro's capabilities which interestingly intersects with the edge of what is possible with Jax. Given the current prevailing notion that LLM's have difficulty 'pushing the frontier of knowlege', the end state of this project is perhaps not so surprising.   

# Project Learnings & Key Takeaways
This project was a deep dive into the practical realities of advanced scientific computing in JAX. The key lessons learned were:

1) Differentiating Through Optimizers is Hard: The "grad-of-a-grad" problem is the core challenge. While jaxopt provides a powerful solution, it can be confounded by the presence of other custom VJPs (like diffrax's adjoint) in the objective function.

2) The Manual VJP is a Last Resort: Manually implementing the Implicit Function Theorem is a powerful technique but is incredibly fragile. The backward pass must be a perfect, numerically stable implementation of the analytical derivatives.

3) The Indispensable Adjoint: For reverse-mode AD (jax.grad) to work on any function containing an adaptive ODE solve, the diffrax solver must be configured with an adjoint method. The various adjoints are applicable in forward or reverse mode, but none of them go to the second derivative level.   

4) Domain-Specific vs. General-Purpose Trade-offs: The speed of nlmixr2 comes from its specialized, generative approach. The power of JAX comes from its generality. This project demonstrates that applying a general tool to a highly specialized problem can expose the deepest and most challenging edge cases of the framework.

# The Debugging Journey: The Search for a Workable Gradient

The initial implementation of the FOCE gradient in JAX did not match the ground truth provided by finite differences. This kicked off an extensive debugging process to find the source of the error. Given the complexity of the system I created of a series of Minimal Reproducible Examples (MREs), each designed to isolate and test a specific component. The MREs are in the `fix-vjp-calc` branch, which is not merged into the main branch. 

## A Note on Pair Programming with an LLM

This MRE process became a core part of my goal to test the limits of LLM pair programming.

- The LLM (Gemini) served as a Socratic partner and a mathematical engine. It proposed debugging strategies (like the "toy problem" and the final "hybrid VJP"), derived the complex matrix calculus for the Implicit Function Theorem on the fly, and generated the boilerplate code for each MRE.

- I was responsible for the high-level strategy, executing the code, and—most critically—identifying the contradictions when the results didn't match the theory. This allowed me to guide the process and ask the specific questions needed to uncover the next layer of the problem.

## MRE 1: Demonstration of Core Problem

- `debug/inner_opt_vjp/gemini_ivp_mre.py` 

Demonstrates that an objective function containing an inner optmizer which itself contains an IVP solve (`diffrax.diffeqsolve`) fails because the custom vjp's utilized by `diffrax` and `jaxopt` do not go to the second derivative level.

## MRE 2: Less MRE More Consolidated Code

- `debug/inner_opt_vjp/debug_customvjp.py` 

This file contains all of the logic from the core package, with the `diffrax.diffeq` solve replaced by a matrix multiplcation. This file was further simplified into MRE 3 and 4. 

## MRE 3: Linear Model Simplification with jaxopt IFT

- `debug/inner_opt_vjp/jaxopt_mre_gemini.py`

This file simplifies MRE 2 and demonstrates that the jaxopt IFT implementation generates the same gradient as finite differences. Critically, this match to finite differences is generated when the inner optmizer DOES NOT contain a `diffrax.diffeqsolve`, which we know fails based on MRE 1. 

It seems therefore that if one can construct a custom vjp for MRE 3 it would then be possible to put the diffrax solver back into the inner optimizer and arrive at a solution to the issue demonstrated in MRE 1. 

## MRE 4: Linear Model Simplification with Custom VJP

- `debug/inner_opt_vjp/gemini_mre_vjp_debug.py`

This file attempts to address the solution pathway hypothesized in the discussion of MRE 3 above. The script contains many control paths for controlling how various elements of the vjp are estimated. For example, how the hessian is constructed and how the quantities in the reverse pass are estimated. 

### The Good News
Ultimately the custom VJP was able to closesly match the finite differences gradients associated with the 'easy' to fit model parameters (the population level effects for the ODE parameters and model error) using analytical approximations of the IFT and exactly match the finite differences gradients when AD was used in the reverse pass. However, it is important to note that using AD in the reverse pass is not viable when the loss function being AD's contains a `diffrax.diffeqsolve`, so the exact AD match is more of a confirmation of the IFT approximation rather than the solution. 

### The Bad News
In contrast my IFT implementation was NOT able to match the finite differences gradients for the components of the between subject variability matrix (`omega` in PK jargon). Strikingly, the AD estimation of gradients matched the results generated by my IFT implementation, but DID NOT match the finite differences estimation of the omega gradient (or the gradient generated by jaxopt which we know matches finite differences based on MRE 3). 

## MRE Conclusion
Resolving the difference between my VJP IFT estimate of the omega gradient in MRE 4 is thus the outstanding unresolved issue which must be resolved in order to address the core problem demonstrated in MRE 1. I believe the solution may lie in an advanced implementation of the IFT which differentiates the optmality condition of the inner loss rather than the inner loss itself (the diff of the inner loss itself being the 'textbook' IFT implementation which clearly does not provide the appropriate resolution for estimating the omega gradient).   


# Present State of the Project
- Frequentist FO Objective: 
