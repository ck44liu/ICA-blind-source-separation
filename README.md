# ICA-blind-source-separation
This project uses Independent Component Analysis to achieve blind source separation. The detailed theoretical derivation can be found in Andrew Ng's [lecture notes](http://cs229.stanford.edu/notes2020spring/cs229-notes11.pdf). In general, the prerequisite to use ICA is that the data is not Gaussian, and the key idea is to use maximum likelihood estimation to find the unmixing matrix W.
## Files
- `icaTest.mat` is a small mixing data used for experimenting the idea of ICA. The code is in `small_dat.py`.
- `sounds.mat` is the mixed sound file we aim to separate, which contains 5 rows and 44000 columns. The code is in `large_dat.py`. It turns out that choosing the appropriate learning rate is important for both datasets, otherwise the gradient ascent update will lead to overflow.
