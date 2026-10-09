"""microbe2metabolite: predict stool metabolite profiles from microbial composition.

Kept import-free on purpose: the MGM environment (torch 2.0.1, pandas 2.0.3) imports
m2m.models and m2m.train, so importing the package must not pull in the main env's stack.
"""

__version__ = "0.1.0"
