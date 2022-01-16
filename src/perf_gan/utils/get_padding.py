def get_padding(kernel_size, stride: int, dilation: int) -> int:
    """Return size of the padding needed

    Args:
        kernel_size ([type]): kernel size of the convolutional layer
        stride (int): stride of the convolutional layer
        dilation (int): dilation of the convolutional layer

    Returns:
        int: padding
    """
    full_kernel = (kernel_size - 1) * dilation + 1
    return full_kernel // 2