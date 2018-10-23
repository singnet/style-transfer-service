import grpc

# import the generated classes
import service.service_spec.style_transfer_pb2_grpc as grpc_bt_grpc
import service.service_spec.style_transfer_pb2 as grpc_bt_pb2

from service import registry, jpg_to_base64, base64_to_jpg

if __name__ == "__main__":

    try:
        # open a gRPC channel
        endpoint = "localhost:7027"
        channel = grpc.insecure_channel("{}".format(endpoint))
        print("opened channel")

        grpc_method = "transfer_image_style"

        content = "https://www.gettyimages.ie/gi-resources/images/Homepage/Hero/UK/CMS_Creative_164657191_Kingfisher.jpg"
        style = "input/style/mondrian.jpg"
        contentSize = 0
        styleSize = 0
        preserveColor = False
        alpha = 0.378
        crop = True
        saveExt = "jpg"

        if grpc_method == "transfer_image_style":
            # create a stub (client)
            stub = grpc_bt_grpc.StyleTransferStub(channel)
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

            # et voilà
            base64_to_jpg(response.data, "/Shared/style_transfer_output_image_new.jpg")
            print("Service completed!")
        else:
            print("Invalid method!")

    except Exception as e:
        print(e)
