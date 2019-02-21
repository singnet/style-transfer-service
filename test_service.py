import grpc

# import the generated classes
import service.service_spec.style_transfer_pb2_grpc as grpc_bt_grpc
import service.service_spec.style_transfer_pb2 as grpc_bt_pb2

from service import registry, base64_to_jpg, clear_file

if __name__ == "__main__":

    try:

        # open a gRPC channel
        endpoint = "localhost:{}".format(registry["style_transfer_service"]["grpc"])
        channel = grpc.insecure_channel("{}".format(endpoint))
        print("Opened channel")

        grpc_method = "transfer_image_style"
        content = \
            "https://www.gettyimages.ie/gi-resources/images/Homepage/Hero/UK/CMS_Creative_164657191_Kingfisher.jpg"
        style = "docs/assets/input/style/mondrian.jpg"
        contentSize = 0
        styleSize = 0
        preserveColor = False
        alpha = 0.378
        crop = True
        saveExt = "jpg"

        # create a stub (client)
        stub = grpc_bt_grpc.StyleTransferStub(channel)
        print("Stub created.")

        # create a valid request message
        request = grpc_bt_pb2.TransferImageStyleRequest(content=content,
                                                        style=style,
                                                        contentSize=contentSize,
                                                        styleSize=styleSize,
                                                        preserveColor=preserveColor,
                                                        alpha=alpha,
                                                        crop=crop,
                                                        saveExt=saveExt)
        # make the call
        response = stub.transfer_image_style(request)
        print("Response received: {}".format(response))

        # et voilà
        output_file_path = "./style_transfer_test_output.jpg"
        if response.data:
            base64_to_jpg(response.data, output_file_path)
            clear_file(output_file_path)
            print("Service completed!")
        else:
            print("Service failed! No data received.")

    except Exception as e:
        print(e)
