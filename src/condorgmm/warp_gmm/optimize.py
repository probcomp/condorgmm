import warp as wp
from .adam import Adam
from .state import State
import condorgmm.warp_gmm.kernels
from tqdm import tqdm


@wp.kernel
def add_noise_kernel(
    param: wp.array(dtype=wp.float32),
):
    tid = wp.tid()
    r = wp.rand_init(tid)
    param[tid] = param[tid] + wp.randn(r) * 0.001


def add_noise(param):
    wp.launch(add_noise_kernel, dim=len(param), inputs=[param])


def optimize_params(
    params_to_optimize,
    frame_warp,
    warp_gmm_state: State,
    num_timesteps,
    lr,
    storing_stuff=False,
    use_tqdm=False,
):
    if not isinstance(lr, list):
        lr = [lr]

    optimizer = Adam(params_to_optimize, lr=lr, betas=(0.9, 0.9995))
    # optimizer = SGD(params_to_optimize, lr=lr)

    params_over_time = []
    likelihood_over_time = []
    gradients_over_time = []

    if storing_stuff:
        params_over_time.append([x.numpy() for x in params_to_optimize])

    pbar = tqdm(range(num_timesteps)) if use_tqdm else range(num_timesteps)
    for step in pbar:
        tape = wp.Tape()
        with tape:
            condorgmm.warp_gmm.kernels.warp_gmm_forward(
                frame_warp,
                warp_gmm_state,
            )

        if use_tqdm:
            pbar.set_description(
                f"Likelihood: {warp_gmm_state.log_score_image.numpy().sum()}"
            )

        if storing_stuff:
            likelihood_over_time.append(warp_gmm_state.log_score_image.numpy().sum())

        tape.backward(
            grads={
                warp_gmm_state.log_score_image: wp.ones_like(
                    warp_gmm_state.log_score_image
                )
            }
        )

        if storing_stuff:
            gradients_over_time.append([x.grad.numpy() for x in params_to_optimize])

        optimizer.step([x.grad for x in params_to_optimize])

        if storing_stuff:
            params_over_time.append([x.numpy() for x in params_to_optimize])

        # if step % 50 == 0:
        #     for param in params_to_optimize:
        #         add_noise(param)

        tape.zero()
        wp.synchronize()

    condorgmm.warp_gmm.kernels.warp_gmm_forward(
        frame_warp,
        warp_gmm_state,
    )

    if storing_stuff:
        likelihood_over_time.append(warp_gmm_state.log_score_image.numpy().sum())

    return {
        "params": params_over_time,
        "likelihoods": likelihood_over_time,
        "gradients": gradients_over_time,
    }
