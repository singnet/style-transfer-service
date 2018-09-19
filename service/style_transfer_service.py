import sys
import logging
import os 

import grpc
import concurrent.futures as futures

from service import default_args
import service.common
import service.style_transfer as style_transfer

# Importing the generated codes from buildproto.sh
import service.model.style_transfer_rpc_pb2_grpc as style_transfer_rpc_pb2_grpc
from service.model.style_transfer_rpc_pb2 import image

logging.basicConfig(
    level=10, format="%(asctime)s - [%(levelname)8s] - %(name)s - %(message)s")
log = logging.getLogger(os.path.basename(__file__))


'''
Neural Artistic Style Transfer Service. To use this service please provide

e.g:
With dApp:  'method': mul
            'params': {"a": 12.0, "b": 77.0}
Resulting:  response:
                value: 924.0


Full snet-cli cmd:
$ snet client call mul '{"a":12.0, "b":77.0}'

Result:
(Transaction info)
Signing job...

Read call params from cmdline...

Calling service...

    response:
        value: 924.0
'''


# Create a class to be added to the gRPC server
# derived from the protobuf codes.
class StyleTransferServicer(style_transfer_rpc_pb2_grpc.StyleTransferServicer):

    def __init__(self):
        # Just for debugging purpose.
        log.debug("StyleTransferServicer created")
        
        # Default input values
        self.output_image_size =  300
        self.start_from_random = False
        self.optimization_rounds = 10
        self.optimization_iterations = 10

    # The method that will be exposed to the snet-cli call command.
    # request: incoming data
    # context: object that provides RPC-specific information (timeout, etc).
    def transfer_style(self, request, context):
        
        def dump_object(obj):
            for descriptor in obj.DESCRIPTOR.fields:
                value = getattr(obj, descriptor.name)
                if descriptor.type == descriptor.TYPE_MESSAGE:
                    if descriptor.label == descriptor.LABEL_REPEATED:
                        map(dump_object, value)
                    else:
                        dump_object(value)
                elif descriptor.type == descriptor.TYPE_ENUM:
                    enum_name = descriptor.enum_type.values[value].name
                    print ("{}: {}".format(descriptor.full_name, enum_name))
                else:
                    print ("{}: {}".format(descriptor.full_name, value))
        dump_object(request)
        
        # Get information from request (defined in .proto file)
        self.content_path = request.content_path
        print("Content path " + self.content_path)
        self.style_path = request.style_path
        print("Style path " + self.style_path)
        self.output_image_size =  request.output_image_size if request.output_image_size  != 0 else self.output_image_size
        print("output_image_size " + str(self.output_image_size))
        self.start_from_random = request.start_from_random
        print("start_from_random " + str(self.start_from_random))
        self.optimization_rounds = request.optimization_rounds if request.optimization_rounds  != 0 else self.optimization_rounds
        print("optimization_rounds " + str(self.optimization_rounds))
        self.optimization_iterations = request.optimization_iterations if request.optimization_iterations  != 0 else self.optimization_iterations
        print("optimization_iterations " + str(self.optimization_iterations))
        
        # Calls transfer_style
        st_model = style_transfer.style_transfer_model()
        output_img = st_model.transfer_style(content_image_path = self.content_path, 
                                             style_image_path = self.style_path,
                                             start_from_random = self.start_from_random,
                                             optimization_rounds = self.optimization_rounds,
                                             optimization_iterations = self.optimization_iterations,
                                             output_image_size = self.output_image_size)
        
        # Creates an image() object (from .proto file) to respond
        self.output_image = image()
        self.output_image.size = self.output_image_size
        #self.output_image.data = style_transfer.style_transfer_model.img_to_base64(output_img)
        self.output_image.data = st_model.npimg_to_base64jpg(output_img)
        log.debug('Style transfer successful!')
        return self.output_image

# The gRPC serve function.
#
# Params:
# max_workers: pool of threads to execute calls asynchronously
# port: gRPC server port
#
# Add all your classes to the server here.
# (from generated .py files by protobuf compiler)
def serve(max_workers=10, port=7777):
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=max_workers))
    style_transfer_rpc_pb2_grpc.add_StyleTransferServicer_to_server(
        StyleTransferServicer(), server)
    server.add_insecure_port('[::]:{}'.format(port))
    return server


if __name__ == '__main__':
    '''
    Runs the gRPC server to communicate with the Snet Daemon.
    '''
    parser = service.common.common_parser(__file__)
    args = parser.parse_args(sys.argv[1:])
    service.common.main_loop(serve, args)
