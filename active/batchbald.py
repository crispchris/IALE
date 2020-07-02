import gc

import torch
from torch import jit
from torch.utils.data import DataLoader

import properties as prop
from active.strategy import Strategy


@jit.script
def logit_mean(logits, dim: int, keepdim: bool = False):
    r"""Computes $\log \left ( \frac{1}{n} \sum_i p_i \right ) =
    \log \left ( \frac{1}{n} \sum_i e^{\log p_i} \right )$.
    We pass in logits.
    """
    return torch.logsumexp(logits, dim=dim, keepdim=keepdim) - math.log(logits.shape[dim])


@jit.script
def entropy(logits, dim: int, keepdim: bool = False):
    return -torch.sum((torch.exp(logits) * logits).double(), dim=dim, keepdim=keepdim)


@jit.script
def mutual_information(logits_B_K_C):
    sample_entropies_B_K = entropy(logits_B_K_C, dim=-1)
    entropy_mean_B = torch.mean(sample_entropies_B_K, dim=1)

    logits_mean_B_C = logit_mean(logits_B_K_C, dim=1)
    mean_entropy_B = entropy(logits_mean_B_C, dim=-1)

    mutual_info_B = mean_entropy_B - entropy_mean_B
    return mutual_info_B


def bald_acquisition_function(logits_b_K_C):
    return mutual_information(logits_b_K_C)


def split_tensors(output, input, chunk_size):
    assert len(output) == len(input)
    return list(zip(output.split(chunk_size), input.split(chunk_size)))


def compute_scores(logits_B_K_C, data_loader, device):

    B, K, C = logits_B_K_C.shape

    # We need to sample the predictions from the bayesian_model n times and store them.
    with torch.no_grad():
        scores_B = torch.empty((B,), dtype=torch.float64)

        if device.type == "cuda":
            gc.collect()
            torch.cuda.empty_cache()
            KC_memory = K * C * 8
            available_memory = torch.cuda.get_device_properties(0).total_memory - torch.cuda.memory_cached()
            batch_size = min(available_memory // KC_memory, 8192)
        else:
            batch_size = 4096

        assert len(scores_B) == len(logits_B_K_C)
        scores_list = list(zip(scores_B.split(batch_size), logits_B_K_C.split(batch_size)))

        previous_b = torch.empty(dtype=torch.float64)
        for scores_b, logits_b_K_C in scores_list:
            scores_b.copy_(mutual_information(logits_b_K_C.to(device)), non_blocking=True)
            previous_b = torch.cat(previous_b, scores_b)

    return previous_b


class BatchBALDSampling(Strategy):
    name = 'batchBALD'

    def __init__(self, dataset_pool, valid_dataset, test_dataset, device='cuda:0'):
        super(BatchBALDSampling, self).__init__(dataset_pool, [], valid_dataset, test_dataset)
        self.min_distances = None

    def query(self, n, model, train_dataset, pool_dataset):
        device = model.state_dict()['softmax.bias'].device

        # -- Reducing the size of the pool dataset
        # initial_length = len(pool_dataset)
        # initial_percentage = 100
        reduce_percentage = 0
        # initial_split_length = initial_length * initial_percentage // 100
        # subset_split.acquire(torch.randperm(initial_length)[initial_split_length:])

        # Empty embeddings
        logits_B_K_C = None

        num_inference = 10  # number of inference / MC Dropout samples
        num_acquisition = n  # number of acquisition samples
        num_classes = 10  # C

        k_lower = 0
        gc.collect()
        torch.cuda.empty_cache()
        chunk_size = 512 if device.type == "cuda" else 32

        with torch.no_grad():
            pool_len = len(self.dataset_pool)  # B
            while k_lower < num_inference:
                try:
                    k_upper = min(k_lower + chunk_size, num_inference)  # K

                    old_logits = logits_B_K_C
                    logits_B_K_C = torch.empty((pool_len, k_upper, num_classes), dtype=torch.float64)
                    if k_lower > 0:
                        logits_B_K_C[:, 0:k_lower, :].copy_(old_logits)

                    # Acquire embeddings
                    model.mode_training = True
                    model.eval()
                    pool_dataloader = DataLoader(pool_dataset, batch_size=prop.VAL_BATCH, shuffle=False, num_workers=0)
                    for i, data in enumerate(pool_dataloader):
                        pool_lower_bound = i * prop.VAL_BATCH
                        pool_upper_upper = min(pool_lower_bound + prop.VAL_BATCH, pool_len)

                        inputs, labels = data[0].float().to(device), data[1].long().to(device)
                        outputs = model(inputs, k_upper - k_lower)
                        logits_B_K_C[pool_lower_bound:pool_upper_upper, k_lower:k_upper].copy_(outputs.double(), non_blocking=True)

                    model.mode_training = False
                    model.train()

                except RuntimeError as e:
                    if isinstance(e, RuntimeError) and "CUDA out of memory." in e.args[0]:
                        if chunk_size <= 1:
                            raise
                        chunk_size = chunk_size // 2
                        gc.collect()
                        torch.cuda.empty_cache()
                    else:
                        raise
                else:
                    if k_upper == num_inference:
                        next_size = num_acquisition
                    elif k_upper < 50:
                        next_size = pool_len
                    else:
                        next_size = max(num_acquisition, pool_len * (100 - reduce_percentage) // 100)

                if next_size < pool_len or k_upper == num_inference:
                    scores_B = compute_scores(logits_B_K_C, pool_dataloader, device)
                else:
                    scores_B = None

                if next_size < pool_len:
                    # Reducing size
                    sorted_indices = torch.argsort(scores_B, descending=True)
                    new_indices = torch.sort(sorted_indices[:next_size], descending=True)
                    pool_len = next_size
                    logits_B_K_C = logits_B_K_C[new_indices]

                    if k_upper == num_inference:
                        logits_B_K_C = logits_B_K_C.clone().detach()
                    scores_B = scores_B[new_indices].clone().detach()

                    subset_split.acquire(sorted_indices[next_size:])

                k_lower += chunk_size

        return