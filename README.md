# ModuleRefinement

## Description

ModuleRefinement is a package for gene co-expression module refinement. This package provides an API for reproducing the results in the gene co-expression module refinement paper.

See `examples/small_example.ipynb` for a notebook that details simple usage of ModuleRefinement along with PyWGCNA. 

## Getting Started

### Dependencies

See `requirements.txt`. `ModuleRefinement` was built with Python 3.8.8.

### Quick Start

1. Initialize conda environment

    ```
    conda create --name module_refinement
    conda activate module_refinement
    ```

1. Install requirements

    ```
    pip install -r requirements.txt
    ```

1. Install `ModuleRefinement` package

    ```
    python -m pip install --index-url https://test.pypi.org/simple/ --no-deps ModuleRefinement==0.0.8
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

<!-- ## Acknowledgments

Inspiration, code snippets, etc.
* [awesome-readme](https://github.com/matiassingers/awesome-readme)
* [PurpleBooth](https://gist.github.com/PurpleBooth/109311bb0361f32d87a2)
* [dbader](https://github.com/dbader/readme-template)
* [zenorocha](https://gist.github.com/zenorocha/4526327)
* [fvcproductions](https://gist.github.com/fvcproductions/1bfc2d4aecb01a834b46) -->