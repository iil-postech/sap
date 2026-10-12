"""Fisher importance of attention-head and FFN-channel masks."""
import torch

from Codes import bin_to_sign


def collect_mask_grads_ecc_revised(model, head_mask, neuron_mask, pc_mtx, dataloader, device):
    was_training = model.training
    model.eval()
    head_mask.requires_grad_(True)
    neuron_mask.requires_grad_(True)
    handles = []
    head_scores = torch.zeros_like(head_mask)
    neuron_scores = torch.zeros_like(neuron_mask)
    parity = pc_mtx.to(device)
    try:
        for index, layer in enumerate(model.decoder.layers):
            def mask_ffn(module, inputs, index=index):
                return (inputs[0] * neuron_mask[index],)
            handles.append(layer.feed_forward.w_2.register_forward_pre_hook(mask_ffn))
        for x, y, magnitude, syndrome in dataloader:
            x, y, magnitude, syndrome = (t.to(device) for t in (x, y, magnitude, syndrome))
            _, loss, _ = model(magnitude, syndrome, parity, y * bin_to_sign(x), y, parity,
                               head_masks=head_mask)
            head_grad, neuron_grad = torch.autograd.grad(loss.mean(), (head_mask, neuron_mask))
            head_scores += head_grad.detach().square()
            neuron_scores += neuron_grad.detach().square()
    finally:
        for handle in handles:
            handle.remove()
        head_mask.requires_grad_(False)
        neuron_mask.requires_grad_(False)
        model.train(was_training)
    return head_scores, neuron_scores
