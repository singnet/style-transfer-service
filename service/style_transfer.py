from __future__ import print_function

import numpy as np
from scipy import optimize as opt
import cntk as C
from PIL import Image
import requests
import h5py
import os
from io import BytesIO
import errno
import base64
import matplotlib.pyplot as plt
import cntk.tests.test_utils
cntk.tests.test_utils.set_device_from_pytest_env() # (only needed for our build system)
C.cntk_py.set_fixed_random_seed(1) # fix a random seed for CNTK components

def download(url, filename):
    response = requests.get(url, stream=True)
    with open(filename, 'wb') as handle:
        for data in response.iter_content(chunk_size=2**20):
            if data: handle.write(data)
                
class style_transfer_model:
    """Transfers the style from one image to another."""
    
    
    def __init__(self):
        
        # Initializing parameters
        self.START_FROM_RANDOM = bool
        self.OPTIMIZATION_ROUNDS = int
        self.OPTIMIZATION_ITERATIONS = int
        self.SIZE = int
        
        self.SAVE_SNAPSHOTS = False
        
        self.CONTENT_WEIGHT = 5.0
        self.STYLE_WEIGHT = 1.0
        self.DECAY = 0.5
        
        self.MODEL_PATH = 'dnn_models/vgg16_weights.bin'
        self.TEMP_IMAGES = {}
        self.TEMP_IMAGES['content'] = 'images/content_temp.jpg'
        self.TEMP_IMAGES['style'] = 'images/style_temp.jpg'

        # Image shift parameters
        self.SHIFT = np.reshape([103.939, 116.779, 123.68], (3, 1, 1)).astype('f')
        
        # Loads the VGG network
        self.MODEL = self._load_model()
    
    def _load_model(self):
        """Loads the VGG network to perform style transfer."""
        # Checks if model is available.
        if not os.path.exists(self.MODEL_PATH):
            raise os.error("Unable to load pre-trained model: file not available. Download it from https://cntk.ai/jup/models/vgg16_weights.bin.")
        f = h5py.File(self.MODEL_PATH)
        layers = []
        for k in range(f.attrs['nb_layers']):
            g = f['layer_{}'.format(k)]
            n = g.attrs['nb_params']
            layers.append([g['param_{}'.format(p)][:] for p in range(n)])
        f.close()
        print('Loaded VGG model.')
        return layers
    
    def _vggblock(self, x, arrays, layer_map, name):
        """A convolutional layer in the VGG network."""
        f = arrays[0]
        b = arrays[1]
        k = C.constant(value=f)
        t = C.constant(value=np.reshape(b, (-1, 1, 1)))
        y = C.relu(C.convolution(k, x, auto_padding=[False, True, True]) + t)
        layer_map[name] = y
        return y
    
    def _vggpool(self, x):
        """A pooling layer in the VGG network."""
        return C.pooling(x, C.AVG_POOLING, (2, 2), (2, 2))
    
    def _model(self, x):
        """Build the graph for the VGG network (excluding fully connected layers)."""
        model_layers = {}
        def convolutional(z): return len(z) == 2 and len(z[0].shape) == 4
        conv = [layer for layer in self.MODEL if convolutional(layer)]
        cnt = 0
        num_convs = {1: 2, 2: 2, 3: 3, 4: 3, 5: 3}
        for outer in range(1,6):
            for inner in range(num_convs[outer]):
                x = self._vggblock(x, conv[cnt], model_layers, 'conv%d_%d' % (outer, 1+inner))
                cnt += 1
            x = self._vggpool(x)

        return x, C.combine([model_layers[k] for k in sorted(model_layers.keys())])
    
    def _flatten(self, x):
        """Flattens an array."""
        assert len(x.shape) >= 3
        return C.reshape(x, (x.shape[-3], x.shape[-2] * x.shape[-1]))

    def _gram(self, x):
        """Calculates the gram matrix (i.e. style matrix) for a layer."""
        features = C.minus(self._flatten(x), C.reduce_mean(x))
        return C.times_transpose(features, features)

    def _npgram(self, x):
        """Calculates the gram matrix (i.e. style matrix) for a layer using numpy."""
        features = np.reshape(x, (-1, x.shape[-2]*x.shape[-1])) - np.mean(x)
        return features.dot(features.T)

    def _style_loss(self, a, b):
        """Defines the style loss."""
        channels, x, y = a.shape
        assert x == y
        A = self._gram(a)
        B = self._npgram(b)
        return C.squared_error(A, B)/(channels**2 * x**4)
    
    def _content_loss(self, a, b):
        """Defines the content loss."""
        channels, x, y = a.shape
        return C.squared_error(a, b)/(channels*x*y)

    def _total_variation_loss(self, x):
        """Defines the total variation loss."""
        xx = C.reshape(x, (1,)+x.shape)
        delta = np.array([-1, 1], dtype=np.float32)
        kh = C.constant(value=delta.reshape(1, 1, 1, 1, 2))
        kv = C.constant(value=delta.reshape(1, 1, 1, 2, 1))
        dh = C.convolution(kh, xx, auto_padding=[False])
        dv = C.convolution(kv, xx, auto_padding=[False])
        avg = 0.5 * (C.reduce_mean(C.square(dv)) + C.reduce_mean(C.square(dh)))
        return avg

    def _save_image(self, img, path):
        """Saves an image in the given path."""
        sanitized_img = np.maximum(0, np.minimum(255, img + self.SHIFT))
        pic = Image.fromarray(np.uint8(np.transpose(sanitized_img, (1, 2, 0))))
        pic.save(path)
    
    def _ordered_outputs(self, f, binding):
        """Orders outputs."""
        _, output_dict = f.forward(binding, f.outputs)
        return [np.squeeze(output_dict[out]) for out in f.outputs]
    
    def _vec2img(self, x):
        """Utility to convert a vector to an image."""
        d = np.round(np.sqrt(x.size / 3)).astype('i')
        return np.reshape(x.astype(np.float32), (3, d, d))

    def _img2vec(self, img):
        """Utility to convert an image to a vector."""
        return img.flatten().astype(np.float64)
    
    def _value_and_grads(self, f, binding):
        """Utility to compute the value and the gradient of f at a particular place defined by binding."""
        if len(f.outputs) != 1:
            raise ValueError('function must return a single tensor')
        df, valdict = f.forward(binding, [f.output], set([f.output]))
        value = list(valdict.values())[0]
        grads = f.backward(df, {f.output: np.ones_like(value)}, set(binding.keys()))
        return value, grads

    def _objfun(self, x, loss):
        """The objective function."""
        y = self._vec2img(x)
        v, g = self._value_and_grads(loss, {loss.arguments[0]: [[y]]})
        v = np.reshape(v, (1,))
        g = self._img2vec(list(g.values())[0])
        return v, g
    
    def _optimize(self, loss, x0):
        """The optimization procedure."""
        bounds = [(-np.min(self.SHIFT), 255-np.max(self.SHIFT))]*x0.size
        for i in range(self.OPTIMIZATION_ROUNDS):
            s = opt.minimize(self._objfun, self._img2vec(x0), args=(loss,), method='L-BFGS-B', 
                             bounds=bounds, options={'maxiter': self.OPTIMIZATION_ITERATIONS}, jac=True)
            print('objective : %s' % s.fun[0])
            x0 = self._vec2img(s.x)
            # Saves a snapshot of the output for every optimization round (OPTIMIZATION_ROUNDS)
            if self.SAVE_SNAPSHOTS:
                path = 'output_%d.jpg' % i
                self._save_image(x0, path)
        return x0
    
    def _load_image(self, path):
        """Loads an image from the given path."""
        #with Image.open(path) as pic:
        pic = Image.open(path)
        
        # Checks if image is PNG, convert to JPG is needed
        if pic.format == 'PNG':
            with BytesIO() as f:
                pic.save(f, format='JPEG')
                f.seek(0)
                pic = Image.open(f)
                hw = pic.size[0] / 2
                hh = pic.size[1] / 2
                mh = min(hw,hh)
                cropped = pic.crop((hw - mh, hh - mh, hw + mh, hh + mh))
                array = np.array(cropped.resize((self.SIZE, self.SIZE), Image.BICUBIC), 
                                 dtype=np.float32)
                return np.ascontiguousarray(np.transpose(array, (2,0,1))) - self.SHIFT
        else:
            hw = pic.size[0] / 2
            hh = pic.size[1] / 2
            mh = min(hw,hh)
            cropped = pic.crop((hw - mh, hh - mh, hw + mh, hh + mh))
            array = np.array(cropped.resize((self.SIZE, self.SIZE), Image.BICUBIC), 
                             dtype=np.float32)
        return np.ascontiguousarray(np.transpose(array, (2,0,1))) - self.SHIFT

    def _load_images(self, content_path, style_path):
        """Loads images or download them if they are not available locally."""
        content_img = content_path
        style_img = style_path
        if not os.path.exists(content_path):
            download('%s' % content_path, self.TEMP_IMAGES['content'])
            content_img = self.TEMP_IMAGES['content']
        if not os.path.exists(style_path):
            download('%s' % style_path, self.TEMP_IMAGES['style'])
            style_img = self.TEMP_IMAGES['style']
        # Load the images
        content = self._load_image(content_img)
        style   = self._load_image(style_img)
        
        return content, style
    
    def _push_images(self, content, style):
        """Push the images through the VGG network """
        
        # First define the input and the output
        y = C.input_variable((3, self.SIZE, self.SIZE), needs_gradient=True)
        z, intermediate_layers = self._model(y)
        # Now get the activations for the two images
        content_activations = self._ordered_outputs(intermediate_layers, {y: [[content]]})
        style_activations = self._ordered_outputs(intermediate_layers, {y: [[style]]})
        style_output = np.squeeze(z.eval({y: [[style]]}))

        # Finally define the loss
        n = len(content_activations)
        total = (1 - self.DECAY ** (n+1)) / (1 - self.DECAY) # makes sure that changing the decay does not affect the magnitude of content/style
        loss = (1.0/total * self.CONTENT_WEIGHT * self._content_loss(y, content) 
                 + 1.0/total * self.STYLE_WEIGHT * self._style_loss(z, style_output) 
                 + self._total_variation_loss(y))

        for i in range(n):
            loss = (loss 
                + self.DECAY ** (i+1) / total * self.CONTENT_WEIGHT * self._content_loss(intermediate_layers.outputs[i], content_activations[i])
                + self.DECAY ** (n-i) / total * self.STYLE_WEIGHT * self._style_loss(intermediate_layers.outputs[i], style_activations[i]))
        return loss

    def _clear_images(self):
        """Deletes temporary image files."""
        # Deletes temporary images
        for img in list(self.TEMP_IMAGES.values()):
            try:
                os.remove(img)
            except OSError as e: # this would be "except OSError, e:" before Python 2.6
                if e.errno != errno.ENOENT: # errno.ENOENT = no such file or directory
                    raise # re-raise exception if a different error occurred
    
    @staticmethod
    def print_image(img):
        """Prints the image on the screen."""
        plt.imshow(img)
    
    @staticmethod
    def img_to_base64(ndarrayimg):
        """Base64 encoding for an image (actually a numpy.ndarray)."""
        contiguous_image = np.ascontiguousarray(ndarrayimg, dtype=np.uint8)
        base64img = base64.b64encode(contiguous_image)
        return base64img      
    
    @staticmethod
    def base64_to_img(base64img, size):
        """Base64 decoding for an image (actually a numpy.ndarray)."""
        base64string = base64.decodebytes(base64img)
        ndarrayimg = np.frombuffer(base64string, dtype=np.uint8).reshape(size, size, 3)
        return ndarrayimg
    
    def transfer_style(self, 
                       content_image_path, 
                       style_image_path,
                       start_from_random = False, 
                       optimization_rounds = 10,
                       optimization_iterations = 20,
                       output_image_size = 300):
        """Transfers the style from a style image to a content image."""
        self.START_FROM_RANDOM = start_from_random
        self.OPTIMIZATION_ROUNDS = optimization_rounds
        self.OPTIMIZATION_ITERATIONS = optimization_iterations
        self.SIZE = output_image_size
        
        # Loads images
        content_image, style_image = self._load_images(content_image_path, style_image_path)
        # Transfers the style
        loss = self._push_images(content_image, style_image)
        np.random.seed(98052)
        if self.START_FROM_RANDOM:
            x0 = np.random.randn(3, self.SIZE, self.SIZE).astype(np.float32)
        else:
            x0 = content_image
        output_image = self._optimize(loss, x0)
        output_image = np.asarray(np.transpose(output_image + self.SHIFT, (1, 2, 0)), dtype=np.uint8)
        
        # Closes images
        self._clear_images()
        
        return output_image
    