import models

def get_model(args, image_shape):

    num_channels, W, H = image_shape

    print("num channels: ", num_channels)

    if args.dataset == 'mnist' or args.dataset == 'affnist':
        num_classes = 10
    elif args.dataset == 'imagenet' or args.dataset == 'rimagenet':
        num_classes = 200 

    # This is the one used in the paper
    model = models.MMAMLConvNet3(num_channels, n_context_channels=args.n_context_channels,
             num_classes=num_classes, support_size=args.support_size, use_context=args.use_context,
                                     prediction_net=args.prediction_net,
                                     pretrained=args.pretrained, context_net=args.context_net, context_num=args.context_num)

    return model
