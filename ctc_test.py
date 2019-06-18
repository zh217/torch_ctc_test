import sys

import torch
import torch.nn


def run_test(ctc_type, data_path, zero_inf):
    test_data = torch.load(data_path)
    inp = test_data['inp']
    inp_len = torch.tensor((inp.size(0),) * inp.size(1), dtype=torch.int32)
    tar = test_data['tar']
    tar_len = test_data['tar_len']

    if ctc_type == 'cudnn':
        inp = inp.cuda().detach()
        inp_len = inp_len.cuda()
    elif ctc_type == 'plain_cuda':
        inp = inp.double().cuda().detach()
        inp_len = inp_len.long().cuda()
        tar = tar.long().cuda()
        tar_len = tar_len.long().cuda()
    else:
        inp = inp.double().detach()

    assert bool(torch.all((inp.exp().sum(dim=-1) - 1).abs() < 1e-5).item())
    inp.requires_grad = True

    loss_fn = torch.nn.CTCLoss(reduction='none', zero_infinity=zero_inf)

    loss = loss_fn(inp, tar, inp_len, tar_len)

    loss.sum().backward()

    grad_sum = inp.grad.sum()
    grad_abs_sum = inp.grad.abs().sum()
    print(f'{ctc_type:11} '
          f'tar_len: {tar_len.tolist()}  '
          f'loss:  {loss[0].item():.10f}, {loss[1].item():.10f}  '
          f'grad_sum: {grad_sum.item():.10f}  '
          f'grad_abs_sum: {grad_abs_sum.item():.10f}')


if __name__ == '__main__':
    print('python version:', sys.version)
    print('torch version:', torch.__version__)
    print('GPU:', torch.cuda.get_device_name())
    for i in range(3):
        data_path = f'ctc_test_data_{i}.pt'
        print()
        for zero_inf in [True, False]:
            print(f'[{data_path}] zero_inf={zero_inf}')
            run_test('cpu', data_path, zero_inf)
            run_test('plain_cuda', data_path, zero_inf)
            run_test('cudnn', data_path, zero_inf)
