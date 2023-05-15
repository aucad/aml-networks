# Referential Result

This directory contains results of experiments presented in the paper.

- `query` directory contains experiment results on different attack iterations. 
- `sample` directory contains experiment results on limited size, sampled input.

To generate plots of these results, run from the repository root:

1. Query-limited experiment results:

    ```
    make plot RESDIR=result/query
    ```
   
2. Sampled-input experiment results:

    ```
    make plot RESDIR=result/sample
    ```