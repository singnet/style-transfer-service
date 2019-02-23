[issue-template]: ../../issues/new?template=BUG_REPORT.md
[feature-template]: ../../issues/new?template=FEATURE_REQUEST.md

<a href="https://singularitynet.io/">
<img align="right" src="./docs/assets/logo/singularityNETblue.png" alt="drawing" width="160"/>
</a>

# Style Transfer

> Repository for the style transfer service on the SingularityNET.

[![Github Issues](http://githubbadges.herokuapp.com/singnet/style-transfer-service/issues.svg?style=flat-square)](https://github.com/singnet/style-transfer-service/issues/) 
[![Pending Pull-Requests](http://githubbadges.herokuapp.com/singnet/style-transfer-service/pulls.svg?style=flat-square)](https://github.com/singnet/style-transfer-service/pulls) 
[![GitHub license](https://img.shields.io/github/license/Naereen/StrapDown.js.svg)](https://github.com/Naereen/StrapDown.js/blob/master/LICENSE)
[![CircleCI](https://circleci.com/gh/singnet/style-transfer-service.svg?style=svg)](https://circleci.com/gh/singnet/style-transfer-service)

This service uses convolutional neural networks to transfer the style of a "style image" to a "content image".

This repository was forked from [xunhuang1995/AdaIN-style](https://github.com/xunhuang1995/AdaIN-style). The original code is written in Lua (using torch). It has been integrated into the SingularityNET using Python 3.6.

Refer to:
- [The User's Guide](https://singnet.github.io/style-transfer-service/): for information about how to use this code as a SingularityNET service;
- [The Original Repository](https://github.com/xunhuang1995/AdaIN-style): for up-to-date information on [xunhuang1995](https://github.com/xunhuang1995)'s implementation of this code.
- [SingularityNET Wiki](https://github.com/singnet/wiki): for information and tutorials on how to use the SingularityNET and its services.

## Contributing and Reporting Issues

Please read our [guidelines](https://github.com/singnet/wiki/blob/master/guidelines/CONTRIBUTING.md#submitting-an-issue) before submitting an issue. If your issue is a bug, please use the bug template pre-populated [here][issue-template]. For feature requests and queries you can use [this template][feature-template].

## Authors

* **Ramon Durães** - *Maintainer* - [SingularityNET](https://www.singularitynet.io)

## Licenses

This project is licensed under the MIT License. The original repository is also licensed under the MIT License. See the [LICENSE](LICENSE) file for details. 