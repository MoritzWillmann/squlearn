# sQUlearn 0.2.0

**_Note:_** This is an early access version! Not everything that is described is already working 100%.

sQUlearn is a novel Python package introducing quantum machine learning (QML) capabilities to 
traditional machine learning pipelines via a 'high-level / low-level' design approach. The software
features an array of algorithms including Quantum Support Vector Machines, Regression,
Gaussian Processes, Kernel Ridge Regression, and a Quantum Neural Network (QNN), all designed to
seamlessly integrate with sklearn. The QNN engine facilitates efficient gradient computation and
automated training with non-linear parametrized circuits. Users can further customize their QNN
models, enhancing flexibility and potential outcomes. sQUlearn's kernel engines are designed to
meet various needs, with fidelity kernels and projected quantum kernels, the latter leveraging
the QNN engine for optimization. A feature map tool allows for efficient layer-wise design
based on strings, encouraging innovation beyond standard implementations. Lastly, backend integration 
with IBM quantum computers is handled with a custom execution engine that optimizes session management
on quantum backends and simulators, ensuring optimal use
of quantum resources and creating an accessible environment for QML experimentation.

## Prerequisites

The package requires **at least Python 3.9**.
## Installation

### Stable Release

To install the stable release version of sQUlearn, run the following command:
```bash
pip install squlearn
```

Alternatively, you can install sQUlearn directly from GitHub via
```bash
pip install git+ssh://git@github.com:sQUlearn/squlearn.git
```

## Examples
There are several more elaborate examples available in the folder ``./examples`` which display the features of this package.
Tutorials for beginners can be found at ``./examples/tutorials``.

To install the required packages, run
```bash
pip install .[examples]
```

## Contribution
Thanks for considering to contribute to sQUlearn! Please read our [contribution guidelines](./.github/CONTRIBUTING.md) before you submit a pull request.

---

## License

[Apache License 2.0](https://github.com/sQUlearn/squlearn/blob/main/LICENSE.txt)

## Imprint
This project is maintained by the quantum computing group at the Fraunhofer Institute for Manufacturing Engineering and Automation IPA. It started as a collection of implementations of quantum machine learning methods.

http://www.ipa.fraunhofer.de/quantum

---
