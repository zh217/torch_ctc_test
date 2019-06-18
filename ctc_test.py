import sys

import torch
import torch.nn


def run_test(use_cuda):
    test_data = torch.load('ctc_test_data.pt')
    inp = test_data['inp']
    inp_len = test_data['inp_len']
    tar = test_data['tar']
    tar_len = test_data['tar_len']

    if use_cuda:
        inp = inp.cuda().detach()
        inp_len = inp_len.cuda()

    print('use_cuda:', use_cuda)
    print('inp:', inp.shape, inp.dtype, inp.device)
    print('tar:', tar.shape, tar.dtype, tar.device)
    print('inp_len:', inp_len)
    print('tar_len:', tar_len)
    print('verify that sum(exp(inp)) == 1:', bool(torch.all((inp.exp().sum(dim=-1) - 1).abs() < 1e-5).item()))

    inp.requires_grad = True

    loss_fn = torch.nn.CTCLoss()

    loss = loss_fn(inp, tar, inp_len, tar_len)

    loss.backward()

    grad_sum = inp.grad.sum()

    print('grad_sum:', grad_sum)


if __name__ == '__main__':
    print('python version:', sys.version)
    print('torch version:', torch.__version__)
    print('GPU:', torch.cuda.get_device_name())
    print()
    run_test(False)
    print()
    run_test(True)
