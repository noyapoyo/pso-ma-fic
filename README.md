# LaTeX Bibliography (BibTeX) Guide

This project uses **IEEEtran** style. Here is how to handle citations and references without errors.

---

## 💡 How it Works (The 4-Step Build)
When you add a `\cite{...}` in `main.tex`, LaTeX needs **4 steps** to update the PDF references:

1. `xelatex main` $\rightarrow$ Finds `\cite{}` keys and writes them to `main.aux`.
2. `bibtex main` $\rightarrow$ Reads `main.aux`, fetches data from `references.bib`, and creates `main.bbl`.
3. `xelatex main` $\rightarrow$ Draws the Reference list at the end of the paper.
4. `xelatex main` $\rightarrow$ Fixes the numbers (e.g., `[1]`) inside the text.

> ⚠️ **CRITICAL ERROR WARNING**
> If you have **NO** `\cite{...}` commands in your text, `bibtex` will generate an empty file, and LaTeX will crash with this error:
> `! LaTeX Error: Something's wrong--perhaps a missing \item.`
> **Fix:** Make sure you have at least one `\cite{...}` in your paper!

---

## 🛠️ How to Update When You Add a New Citation

Whenever you add a new `\cite{}` or update `references.bib`, run these commands in your terminal:

```bash
# Step 1: Clean old cache files to prevent errors
rm -f main.aux main.bbl main.blg main.log main.out

# Step 2: Run the 4-step compilation
xelatex main
bibtex main
xelatex main
xelatex main