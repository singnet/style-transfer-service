[issue-template]: ../../../issues/new?template=BUG_REPORT.md
[feature-template]: ../../../issues/new?template=FEATURE_REQUEST.md

<!--
<a href="https://singularitynet.io/">
<img align="right" src="../assets/logo/singularityNETblue.png" alt="drawing" width="160"/>
</a>
-->

# Style Transfer

This service uses convolutional neural networks to transfer the artistic style of a "style" image to a "content" image.

It is part of SingularityNET's third party services, [originally implemented by xunhuang1995](https://github.com/xunhuang1995/AdaIN-style).

### Welcome

The service takes as input:
- Required parameters:
    - content (string): the image of interest. Takes a URL or base64-encoded image;
    - style (string): the image from which the artistic style will be transferred. Takes a URL or base64-encoded image;
- Other (optional) parameters:
    - alpha (float): a value between 0 and 1 interpreted as the content-style trade-off. Defaults to **1**, which means that the style will be fully transferred to the content image. Smaller values cause the content image to keep more of its original colors and shapes;
    - preserveColor (bool): whether or not to keep the original (content image's) colors (instead of applying the same colors of the style image). Defaults to **false**;
    - crop (bool): whether or not to use (and output) only the central (square-shaped) crop of both images. Defaults to **false**;
    - contentSize (integer): the output size of the content image. Will apply this value to the smallest dimension of the image and adjust the other to keep proportions;
    - styleSize (integer): the output size of the style image. Will apply this value to the smallest dimension of the image and adjust the other to keep proportions;

Since the service can easily consume all the GPU memory, for SNET dAPP both the content and style output sizes have been fixed at **640**. For SNET CLI, it will still try to use the original dimensions of the input/content and style images.

### What’s the point?

This service can be used to generate artistic images by copying another image's style (i.e.: cubist painting, cartoon, pencil drawing, etc.).

### How does it work?

To get a response from the service, the only required inputs are the content and style images. The remaining inputs will be set to their default values.

You can use this service at [SingularityNET DApp](http://beta.singularitynet.io/) by clicking on `snet/style-transfer`.

You can also call the service from SingularityNET CLI:

```
$ snet client call transfer_image_style '{"content": "CONTENT_IMAGE_URL", "style": "STYLE_IMAGE_URL"}'
```

Go to [this tutorial](https://dev.singularitynet.io/tutorials/publish/) to learn more about publishing, using and deleting a service.

### What to expect from this service?

Example:

**Input**

- content: "https://raw.githubusercontent.com/singnet/style-transfer-service/master/docs/assets/input/content/newyork.jpg"
- style: "https://raw.githubusercontent.com/singnet/style-transfer-service/master/docs/assets/input/style/brushstrokes.jpg"
- preserveColor: true

Content Image                        | Style Image
:-----------------------------------:|:-------------------------:
<img width="100%" src="assets/input/content/newyork.jpg"> | <img width="100%" src="assets/input/style/brushstrokes.jpg">

**Output**

<img width="100%" src="assets/examples/newyork_brushstrokes_preservecolor.jpg">
