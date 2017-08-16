# probability_GAN
> This work was introduced in the summer research program supervised by Prof. Vijayakumar Bhagavatula hosted by Carnegie Mellon University after I finish my second year study in computer science department.

Currently, generative adversarial net is widely used in domain adaptation, image to image translation, adversarial training and etc. A bunch of different GANs are proposed to solve these problems, most of them proposed a new loss function and experiment on image datasets. But nearly none of them explain GAN back to the probability view. In this project, I explore the insight of GAN, simGAN and cycleGAN in distribution level.

## Installation

Pytorch with cpu Version

## Usage

mixGau-simGAN.py    generated mixture Gaussian data from mixture Gaussian Distribution using simGAN [1]

mixGau_GAN.py    generated mixture Gaussian data from mixture Gaussian Distribution using GAN [2]

mixGau_cycleGAN.py    generated mixture Gaussian data from mixture Gaussian Distribution using cycleGAN [3]

uniMix_GAN.py     generated mixture Gaussian data from uniform Distribution using GAN

uniNor_cycleGAN.py     generated mixture Gaussian data from uniform Distribution using GAN

```sh
python uniMix_GAN.py

```

## Experiment Result

Copare of converging speed of S dataset.

![alt tag](https://raw.githubusercontent.com/MaureenZOU/i-CPCL-Algorithm/master/sample.png)



CPU

![alt tag](https://raw.githubusercontent.com/MaureenZOU/i-CPCL-Algorithm/master/cpu.png)

Epoch

![alt tag](https://raw.githubusercontent.com/MaureenZOU/i-CPCL-Algorithm/master/epoch.png)

Comparison of i-CPCL, l-CPCL and s-CPCL algorithm with 10 Gaussian clusters.

![alt tag](https://raw.githubusercontent.com/MaureenZOU/i-CPCL-Algorithm/master/compare.png)

## Release History

* 0.2.1
    * CHANGE: Update docs (module code remains unchanged)
* 0.2.0
    * CHANGE: Remove `setDefaultXYZ()`
    * ADD: Add `init()`
* 0.1.1
    * FIX: Crash when calling `baz()` (Thanks @GenerousContributorName!)
* 0.1.0
    * The first proper release
    * CHANGE: Rename `foo()` to `bar()`
* 0.0.1
    * Work in progress

## Meta

Your Name – [@YourTwitter](https://twitter.com/dbader_org) – YourEmail@example.com

Distributed under the XYZ license. See ``LICENSE`` for more information.

[https://github.com/yourname/github-link](https://github.com/dbader/)

## Contributing

1. Fork it (<https://github.com/yourname/yourproject/fork>)
2. Create your feature branch (`git checkout -b feature/fooBar`)
3. Commit your changes (`git commit -am 'Add some fooBar'`)
4. Push to the branch (`git push origin feature/fooBar`)
5. Create a new Pull Request

<!-- Markdown link & img dfn's -->
[npm-image]: https://img.shields.io/npm/v/datadog-metrics.svg?style=flat-square
[npm-url]: https://npmjs.org/package/datadog-metrics
[npm-downloads]: https://img.shields.io/npm/dm/datadog-metrics.svg?style=flat-square
[travis-image]: https://img.shields.io/travis/dbader/node-datadog-metrics/master.svg?style=flat-square
[travis-url]: https://travis-ci.org/dbader/node-datadog-metrics
[wiki]: https://github.com/yourname/yourproject/wiki
