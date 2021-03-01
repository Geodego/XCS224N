# XCS229-Handouts

This repository contains .tex and .pdf along with relevant files for Handouts used in both XCS229i: Machine Learning and XCS229ii: Machine Learning Strategy and Intro to Reinforcement Learning. Below is a snapshot of where each handout belongs within the scope of these two classes:

### XCS229i: Machine Learning Handouts
- Supervised Learning
  - Supervised Learning and Discriminative Algorithms (Further reading: Binary Classification and Logistic Regression, Perceptron)
  - Generative Learning Algorithms
  - Kernels and Support Vector Machines (Further Reading: Representer Functions)
  - Decision Trees (Further Reading: Boosting, Ensemble Methods)
- Unsupervised Learning
  - K-Means Clustering
  - Mixture of Gaussians
  - Expectation Maximization Algorithm
  - Factor Analysis
  - Independent Component Analysis
- Deep Learning
  - Deep Learning (Further Reading: Backpropagation)

### XCS229ii: Machine Learning Strategy and Intro to Reinforcement Learning
- Learning Strategy
  - Learning Theory (Further Reading: Bias-Variance, Error Analysis)
  - Regularization and Model Selection
- Reinforcement Learning
  - Reinforcement Learning and Control
  - LQR, DDP, and LQG

# Compiling Latex

Each subdirectory within this folder has, at a minimum, a file titled
`latexmkrc`.  This is the settings file for latexmk, which will handle
juggling the various latex engines preferred by the course staff.  A basic
`latexmkrc` file (e.g., for `pdflatex`) might have the following contents
([`latexmk` documentation](https://mirror.las.iastate.edu/tex-archive/support/latexmk/latexmk.pdf)):
```
@default_files = ("main.tex");   # Set the root tex file for the output document
$pdf_mode = 1;                   # tex -> PDF
$auto_rc_use = 1;                # Do not read further latexmkrc files
$warnings_as_errors = 1;         # Elevates warnings to errors.  Enforces cleaner code.
$pdflatex = "pdflatex -halt-on-error -interaction=batchmode %O %S";
                                 # Forces latexmk to stop and quit if it encounters an error
$jobname = "output_name";        # This is the name of the output PDF file
$silent = 1                      # For quieter output on the terminal.
```
Feel free to customize this as you desire, including adding more files,
directories, and media.  There is only one requirements:

**IT MUST BE POSSIBLE TO COMPILE EACH DOCUMENT USING ONLY THE FOLLOWING COMMAND:**
```
$ latexmk
```
A properly setup `latexmkrc` file can handle any special compilation options you
may require.  Put those options in the `latexmkrc` file so that other course
staff can compile your document with the command above.

Other commands that might be helpful include:
- `$ latexmk -pvc`:  (preview continuously) This will run `latexmk`
continuously, allowing you to immediately view changes to your output document
as you save source files.
- `$ latexmk -c`:  This will remove all auxiliary files other than the final
output PDF.
- `$ latexmk -C`:  This will remove all output files (including the final output
PDF).
