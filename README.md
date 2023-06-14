# ModuleRefinement

## Description

ModuleRefinement is a package for gene co-expression module refinement. This package provides an API for reproducing the results in the gene co-expression module refinement paper.

See `examples/small_example.ipynb` for a notebook that details simple usage of ModuleRefinement along with PyWGCNA. 

## How to Cite Our Work

If you like this work, please consider citing us

```
@article{mankovich2023module,
  title={Module representatives for refining gene co-expression modules},
  author={Mankovich, Nathan and Andrews-Polymenis, Helene and Threadgill, David and Kirby, Michael},
  journal={Physical Biology},
  volume={20},
  number={4},
  pages={045001},
  year={2023},
  publisher={IOP Publishing}
}
```

## Getting Started

### Dependencies

See `requirements.txt`. `ModuleRefinement` was built with Python 3.8.8.

### Quick Start

1. Initialize conda environment

    ```
    conda create --name module_refinement python=3.8.8
    conda activate module_refinement
    ```

1. Install requirements

    ```
    pip install -r requirements.txt
    ```

1. Install `ModuleRefinement` package

    ```
    python -m pip install --index-url https://test.pypi.org/simple/ --no-deps ModuleRefinement==0.0.14
    ```

1. Open `./examples/small_example.ipynb` and run it within the`module_refinement` environment.

`small_example.ipynb` shows how to compute:

* WGCNA modules
* Refined modules using subspace LBG clustering
* Relative gain in GO signiciance
* Relative gain in classification BSR

<!-- ### Executing program

* How to run the program
* Step-by-step bullets
```
code blocks for commands
```

## Help

Any advise for common problems or issues.
```
command to run if program contains helper info
``` -->

## Authors

Nathan Mankovich: [nmank@colostate.edu](mailto:nmank@colostate.edu)

## Version History

* 0.8
    * Initial functional release

## License

See the `LICENSE.md` file for details

## Acknowledgments

Big thank you to [EricStern](https://github.com/estern95) for his help packaging the repository and ensuring reproducibility. Another thanks to [EricKehoe](https://github.com/ekehoe32) for his work on the code base for Pathway Expression Analysis which was used to run some of these examples.
